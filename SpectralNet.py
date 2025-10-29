import torch
import torch.nn as nn
import timm
import numpy as np
import os
import glob
from datetime import datetime,timedelta
from itertools import tee
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset,DataLoader,Dataset,random_split
from torch.optim.lr_scheduler import StepLR
from timm.models.layers import DropPath
from typing import Optional
from einops import rearrange
from typing import Tuple, List
from tqdm import tqdm
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import EMInverseDataset, dms_to_decimal

class PositionAwareBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.attn(x))

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FreqDriftModel(nn.Module):
    def __init__(self, input_dim=4, embed_dim=64, layers=6, num_classes=34):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.depthwise_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.pointwise_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        self.encoder = nn.Sequential(*[
            PositionAwareBlock(embed_dim) for _ in range(layers)
        ])

        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x_embed = self.input_proj(x)  
        x_embed = self.pos_encoding(x_embed)

        x_conv = self.pointwise_conv(self.depthwise_conv(x_embed.transpose(1, 2))) 
        x_embed = x_embed + x_conv.transpose(1, 2)
        x_encoded = self.encoder(x_embed)
        logits = self.output_head(x_encoded)
        return logits
    

def total_loss(pred,target,alpha=1.0,beta=1.0,gamma=2.0):
    mse_loss = F.mse_loss(pred,target)
    target_diff = target[:,1:] - target[:,:-1]
    pred_diff = pred[:,1:] - pred[:,:-1]
    diff_jump_loss = F.mse_loss(target_diff,pred_diff)
    return mse_loss*alpha + beta * diff_jump_loss


def freq_to_class(freq_tensor):
    flat_freq = freq_tensor.view(-1)
    indices = []
    for f in flat_freq:
        f_val = f.item()
        if f_val in freq2idx:
            idx = freq2idx[f_val]
        else:
            nearest_freq = min(freq2idx.keys(), key=lambda x: abs(x - f_val))
            idx = freq2idx[nearest_freq]
        indices.append(idx)
    class_tensor = torch.tensor(indices, dtype=torch.long, device=freq_tensor.device).view(freq_tensor.shape)
    return class_tensor

def class_to_freq(class_tensor):
    flat_class = class_tensor.view(-1).clamp(min=0, max=len(idx2freq) - 1)
    freqs = [idx2freq[i.item()] for i in flat_class]
    freq_tensor = torch.tensor(freqs, dtype=torch.float32, device=class_tensor.device).view(class_tensor.shape)
    return freq_tensor

freq_values = [10.0] + list(np.arange(24.0, 40.5, 0.5))  # 共34类
freq2idx = {v: i for i, v in enumerate(freq_values)}
idx2freq = {i: v for i, v in enumerate(freq_values)}
num_classes = len(freq_values)

def train_freq_model(
    model,
    batch_size=64,
    num_epochs=50,
    lr=1e-3,
    weight_smooth=0.01,
    val_ratio=0.1,
    save_path="freq_model.pth",
    device="cuda" if torch.cuda.is_available() else "cpu"
):

    pretrained_model_path = save_path
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        print('Pretrained model loaded')
    else:
        print('Pretrained model not found')

    model = model.to(device)
    dataset = EMInverseDataset(root_path='dataset')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    label_min = dataset.label_min
    label_max = dataset.label_max
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_total = 0.0
        loop = tqdm(train_loader, desc=f"[Epoch {epoch}/{num_epochs}] Train")

        for x_img, recv, x_param, line_freq in loop:
            optimizer.zero_grad()
            x_param = x_param.to(device)      
            line_freq = line_freq.to(device)   
            target_class = freq_to_class(line_freq*40)  
            logits = model(x_param)
            loss = criterion(logits.view(-1, num_classes), target_class.view(-1))
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * x_param.size(0)
            loop.set_postfix(loss=loss.item())
        avg_train_loss = train_loss_total / train_size

        # === Validation ===
        model.eval()
        val_loss_total = 0.0
        mae_total = 0.0
        mean_drop_mag = 0.0
        mean_drop_rate = 0.0
        with torch.no_grad():
            for idx,(x_img, recv, x_param, line_freq) in enumerate(val_loader):
                x_param = x_param.to(device)
                line_freq = line_freq.to(device)
                target_class = freq_to_class(line_freq*40)
                logits = model(x_param)
                loss = criterion(logits.view(-1, num_classes), target_class.view(-1))
                val_loss_total += loss.item() * x_param.size(0)

                pred_class = logits.argmax(dim=-1)
                pred_freq = class_to_freq(pred_class)
                real_freq = class_to_freq(target_class)
                mae = torch.abs(pred_freq - real_freq).mean()
                mae_total += mae.item() * x_param.size(0)

                base_freq_norm = line_freq
                base_freq = base_freq_norm * 40
                mask = (pred_freq != 10.0) & (base_freq!=10.0)
                drop_mag = torch.abs((base_freq - pred_freq)[mask])
                mean_drop_mag += drop_mag.mean()
        
        avg_val_loss = val_loss_total / val_size
        avg_mae = mae_total / val_size
        mean_drop_mag = mean_drop_mag / (val_size//batch_size+1)
        mean_drop_rate = mean_drop_rate / (val_size//batch_size+1)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MAE: {avg_mae:.4f} | DropMag:{mean_drop_mag:.4f}")
        
model = FreqDriftModel()
# train_freq_model(model, num_epochs=200, lr=1e-6, batch_size=64)