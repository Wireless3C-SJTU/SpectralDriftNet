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
from utils import dms_to_decimal, EMInverseDataset
import seaborn as sns
from forward import ForwardModel

class SwinEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.backbone = timm.create_model(
            'swinv2_tiny_window8_256',
            pretrained=False,
            num_classes=0,
            global_pool='',
            features_only=True
        )
        self.out_proj = nn.Linear(self.backbone.feature_info[-1]['num_chs'], out_dim)

    def forward(self, x):  
        x = x.repeat(1, 3, 1, 1)
        feat = self.backbone(x)[-1]  
        B, H, W, C = feat.shape
        feat = feat.view(B, H * W, C)  
        return self.out_proj(feat)  



class ParamDecoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, seq_len=256, output_dim=4, coord_dim=2):
        super().__init__()
        self.seq_len = seq_len
        self.query_embed = nn.Parameter(torch.randn(seq_len, hidden_dim))  

        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tx_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=8,
            num_encoder_layers=3, num_decoder_layers=3,
            dim_feedforward=1024, dropout=0.1, batch_first=True
        )

        self.param_gru = nn.GRU(hidden_dim + 1, hidden_dim, batch_first=True)
        self.param_head = nn.Linear(hidden_dim, output_dim)  
        self.freq_rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                               batch_first=True, bidirectional=True)
        self.freq_head = nn.Linear(hidden_dim*2, num_classes)  

    def forward(self, src, recv_coord, tx_coord):
        B = src.size(0)
        src = self.input_proj(src) 
        coord_feat = self.coord_encoder(recv_coord)  
        coord_feat = coord_feat.unsqueeze(1).expand(-1, self.seq_len, -1)
        tgt = self.query_embed.unsqueeze(0).expand(B, -1, -1) + coord_feat
        
        if tx_coord is not None:
            tx_coord = tx_coord.unsqueeze(-1).float()      
            tx_feat = self.tx_encoder(tx_coord)           
            tgt = tgt + tx_feat                           
        out = self.transformer(src, tgt)  
        init_pos = torch.zeros(B, 1, 1, device=src.device) 
        pos_preds = []
        pos = init_pos
        h = torch.zeros(1, B, out.size(-1), device=src.device)

        for t in range(self.seq_len):
            inp = torch.cat([out[:,t,:], pos.squeeze(1)], dim=-1).unsqueeze(1)  
            memory, h = self.param_gru(inp, h)   
            delta_param = self.param_head(memory)     
            delta_pos = delta_param[:, :, :]             
            pos = pos+delta_pos
            pos_preds.append(pos)

        params = torch.cat(pos_preds, dim=1) 
        freq_feat, _ = self.freq_rnn(out)  
        freq = self.freq_head(freq_feat) 
        return freq, params

class SequenceUpsampler(nn.Module):
    def __init__(self, input_len=64, output_len=256, dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_len, output_len)
        self.dim = dim

    def forward(self, x):  
        x = x.permute(0, 2, 1)  
        x = self.fc1(x)         
        x = x.permute(0, 2, 1)  
        return x

class SwinTFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SwinEncoder(out_dim=768)
        self.upsampler = SequenceUpsampler(input_len=64, output_len=256, dim=768)
        self.decoder = ParamDecoder(input_dim=768, hidden_dim=256, seq_len=256, output_dim=1)

    def forward(self, x, recv_coord,x_lat):
        feat = self.encoder(x)           
        feat = self.upsampler(feat)      
        logits,lon = self.decoder(feat, recv_coord,x_lat)  
        return logits,lon

freq_values = [10.0] + list(np.arange(24.0, 40.5, 0.5))  
freq2idx = {v: i for i, v in enumerate(freq_values)}
idx2freq = {i: v for i, v in enumerate(freq_values)}
num_classes = len(freq_values)

def denormalize(pred, label_min, label_max):
    pred[:,:,:3] = pred[:,:,:3] * (label_max[:3] - label_min[:3]) + label_min[:3]
    pred[:,:,3] = pred[:,:,3]*40
    return pred

def compute_errors(pred, target):
    B, T, _ = pred.shape
    loc_errs = []
    depth_errs = []
    freq_errs = []

    for b in range(B):
        for t in range(T):
            p1 = (pred[b, t, 0], pred[b, t, 1])  
            p2 = (target[b, t, 0], target[b, t, 1])
            loc_err = geodesic(p1, p2).meters
            loc_errs.append(loc_err)
            depth_errs.append(abs(pred[b, t, 2] - target[b, t, 2]))
            freq_errs.append(abs(pred[b, t, 3] - target[b, t, 3]))

    return {
        'loc_error(m)': np.mean(loc_errs),
        'depth_error(m)': np.mean(depth_errs),
        'freq_error(Hz)': np.mean(freq_errs)
    }

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        loss = ((pred - target) ** 2) * self.weights
        pred_vel = pred[:, 1:, :2] - pred[:, :-1, :2]
        true_vel = target[:, 1:, :2] - target[:, :-1, :2]
        loss_vel = F.mse_loss(pred_vel, true_vel)
        return loss.mean()+loss_vel*100


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

def train_inverse_model(model,forward_model,epochs=100,batch_size=32,lr=1e-6,train_ratio=0.9,device='cuda' if torch.cuda.is_available() else 'cpu'):
    pretrained_model_path = 'inverse_model_withforward.pth'.format(train_ratio)
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        print('pretrained model loaded')
    else:
        print('pretrained model not found')
    model = model.to(device)
    forward_model = ForwardModel()
    
    full_dataset = EMInverseDataset(root_path='dataset')
    label_min = torch.tensor(full_dataset.label_min).to(device)
    label_max = torch.tensor(full_dataset.label_max).to(device)
    train_ratio = train_ratio
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    print(f" Dataset loaded: {len(train_dataset)} train / {len(val_dataset)} val")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = [0.0, 1.0, 0, 0.5] 
    criterion = WeightedMSELoss(weights).to(device)
    freq_criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for train_idx,(x_img, recv_coord, x_param, freq) in enumerate(loop):
            x_img = x_img.to(device)
            x_param = x_param.to(device)
            recv_coord = recv_coord.to(device)
            freq = freq.to(device)

            optimizer.zero_grad()
            logits,x_lon_pred = model(x_img, recv_coord, x_param[:,:,0]) 
            target_class = freq_to_class(x_param[:,:,3]*40)
            pred_class = logits.argmax(dim=-1)  
            pred_freq = class_to_freq(pred_class)
            
            x_param_new = x_param.clone()  
            x_param_new[:, :, 1] = x_lon_pred.squeeze(-1)  
            x_param_new[:, :, 3] = (pred_freq/40).squeeze(-1)  
            x_param_pred = x_param_new
            loss = criterion(x_param_pred, x_param)
            recon = forward_model(x_param_pred)
            recon_loss = mse_loss(recon,freq)
            freq_loss = freq_criterion(logits.view(-1, num_classes), target_class.view(-1))
            loss = loss+freq_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_img.size(0)
            loop.set_postfix(loss=loss.item())
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        all_pred = []
        all_true = []
        mean_drop_mag = 0.0
        mean_drop_rate = 0.0
        with torch.no_grad():
            for val_idx, (x_img, recv_coord, x_param,freq) in enumerate(val_loader):
                x_img = x_img.to(device)
                x_param = x_param.to(device)
                recv_coord = recv_coord.to(device)
                freq = freq.to(device)
                logits,x_lon_pred = model(x_img, recv_coord, x_param[:,:,0]) 
                target_class = freq_to_class(x_param[:,:,3]*40)
                pred_class = logits.argmax(dim=-1)  
                pred_freq = class_to_freq(pred_class)

                x_param_new = x_param.clone()  
                x_param_new[:, :, 1] = x_lon_pred.squeeze(-1)  
                x_param_new[:, :, 3] = (pred_freq/40).squeeze(-1)  
                x_param_pred = x_param_new
                loss_val = criterion(x_param_pred, x_param)
                freq_loss = freq_criterion(logits.view(-1, num_classes), target_class.view(-1))
                loss_val = loss_val + freq_loss
                val_loss += loss_val.item() * x_img.size(0)

                base_freq_norm = x_param[:, :, -1]
                base_freq = base_freq_norm * 40
                mask = (base_freq != 10.0)
                drop_mag = torch.abs((base_freq - pred_freq)[mask])
                mean_drop_mag += drop_mag.mean()

                pred_denorm = denormalize(x_param_pred, label_min, label_max).cpu().numpy()
                label_denorm = denormalize(x_param, label_min, label_max).cpu().numpy()
                all_pred.append(pred_denorm)
                all_true.append(label_denorm)

        val_loss /= len(val_dataset)
        pred_all = np.concatenate(all_pred, axis=0)
        label_all = np.concatenate(all_true, axis=0)
        metrics = compute_errors(pred_all, label_all)
        mean_drop_mag = mean_drop_mag / (val_size//batch_size+1)

        print(f"Epoch {epoch+1}: "
              f"Train Loss = {train_loss:.6f}, "
              f"Val Loss = {val_loss:.6f}, "
              f"Loc Error = {metrics['loc_error(m)']:.2f}m, "
              f"Depth Error = {metrics['depth_error(m)']:.2f}m, "
              f"DropMag = {mean_drop_mag:.4f}Hz, "
              f"Freq Error = {metrics['freq_error(Hz)']:.4f}Hz")

model = SwinTFModel()
forward_model = ForwardModel()
# train_inverse_model(model,forward_model,batch_size=32,lr=1e-6,train_ratio=0.8)


