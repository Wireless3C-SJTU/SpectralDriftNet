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
from utils import EMInverseDataset, nearestUpsmaple
import seaborn as sns
from SpectralNet import FreqDriftModel,freq_to_class,class_to_freq
from inverse import SwinTFModel

device="cuda" if torch.cuda.is_available() else "cpu"

freq_values = [10.0] + list(np.arange(24.0, 40.5, 0.5))
freq2idx = {v: i for i, v in enumerate(freq_values)}
idx2freq = {i: v for i, v in enumerate(freq_values)}
num_classes = len(freq_values)

def latlon_to_xy(coords, ref_point=None):
    lat = coords[:,0]
    lon = coords[:,1]
    if ref_point is None:
        lat0, lon0 = np.mean(lat), np.mean(lon)
    else:
        lat0, lon0 = ref_point

    R = 6371000  
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * R
    y = (lat_rad - lat0_rad) * R
    return np.stack([x, y], axis=-1)

def getSinglePredResult(idx=0):
    model = FreqDriftModel()
    pretrained_model_path = "freq_model.pth"
    model.load_state_dict(torch.load(pretrained_model_path))
    model = model.to(device)
    dataset = EMInverseDataset(root_path='dataset')
    criterion = nn.MSELoss()
    freq_min = torch.tensor(dataset.freq_min).to(device)
    freq_max = torch.tensor(dataset.freq_max).to(device)

    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    batch_size=64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    label_min = dataset.label_min
    label_max = dataset.label_max

    model.eval()
    with torch.no_grad():
        for x_img, recv, x_param, line_freq in val_loader:
            x_param = x_param.to(device)
            line_freq = line_freq.to(device)

            target_class = freq_to_class(line_freq*40)
            logits = model(x_param)  
            pred_class = logits.argmax(dim=-1)  
            
            base_freq_norm = x_param[idx, :, -1].detach().cpu().numpy() 
            base_freq = base_freq_norm * (label_max[-1] - label_min[-1] + 1e-6) + label_min[-1]
            
            pred_freq = class_to_freq(pred_class)
            real_freq = class_to_freq(target_class)
            pred_freq = pred_freq[idx].detach().cpu().numpy()
            real_freq = real_freq[idx].detach().cpu().numpy()

            plt.figure(figsize=(10, 5))
            sns.lineplot(x=list(range(len(pred_freq))), y=pred_freq, color='red', label='Predicted', linewidth=3, alpha=0.5)
            sns.lineplot(x=list(range(len(real_freq))), y=real_freq, color='blue', label='Real', linewidth=3, alpha=0.5)
            sns.lineplot(x=list(range(len(base_freq))), y=base_freq, color='green', label='Transmit', linewidth=3, alpha=0.5, linestyle='--')
            plt.xlabel("time(s)")
            plt.ylabel("freqency(hz)")
            plt.ylim(0, 40)  
            plt.title("Frequency of single window sample")
            plt.tight_layout()
            plt.savefig('plot/SinglePredResult_{}.png'.format(idx+1))
            break

def getSingleInverseResult(idx=0):
    model = SwinTFModel()
    pretrained_model_path = "inverse_model_withforward.pth"
    model.load_state_dict(torch.load(pretrained_model_path))
    model = model.to(device)
    dataset = EMInverseDataset(root_path='dataset')
    criterion = nn.MSELoss()
    freq_min = torch.tensor(dataset.freq_min).to(device)
    freq_max = torch.tensor(dataset.freq_max).to(device)

    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    batch_size=64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    label_min = dataset.label_min
    label_max = dataset.label_max

    model.eval()
    with torch.no_grad():
        for val_idx, (x_img, recv_coord, x_param,freq) in enumerate(val_loader):
            x_img = x_img.to(device)
            x_param = x_param.to(device)
            recv_coord = recv_coord.to(device)
            freq = freq.to(device)
            
            target_class = freq_to_class(freq*40)
            logits, x_lon_pred = model(x_img, recv_coord, x_param[:,:,0]) 

            pred_class = logits.argmax(dim=-1)  
            pred_freq = class_to_freq(pred_class)
            real_freq = class_to_freq(target_class)
            base_freq = x_param[:,:,-1] * 40

            x_param_new = x_param.clone()  
            x_param_new[:, :, 1] = x_lon_pred.squeeze(-1)  
            x_param_new[:, :, 3] = (pred_freq/40).squeeze(-1)  
            x_param_pred = x_param_new

            pred_freq = pred_freq[idx].detach().cpu().numpy()
            real_freq = real_freq[idx].detach().cpu().numpy()
            base_freq = base_freq[idx].detach().cpu().numpy()

            plt.figure(figsize=(10, 5))
            sns.lineplot(x=list(range(len(pred_freq))), y=pred_freq, color='red', label='Predicted', linewidth=3, alpha=0.5)
            sns.lineplot(x=list(range(len(real_freq))), y=real_freq, color='blue', label='Real', linewidth=3, alpha=0.5)
            sns.lineplot(x=list(range(len(base_freq))), y=base_freq, color='green', label='Transmit', linewidth=3, alpha=0.5, linestyle='--')
            plt.xlabel("time(s)")
            plt.ylabel("freqency(hz)")
            plt.ylim(0, 40)  
            plt.title("Frequency Result of Inverse Model")
            plt.tight_layout()
            plt.savefig('plot/FreqInverseResult_{}.png'.format(idx+1))
        
            pred_coord = x_param_pred[idx, :, :2].detach().cpu().numpy()  
            real_coord = x_param[idx, :, :2].detach().cpu().numpy()      
            pred_coord = pred_coord * (label_max[:2] - label_min[:2] + 1e-6) + label_min[:2]
            real_coord = real_coord * (label_max[:2] - label_min[:2] + 1e-6) + label_min[:2]
            pred_coord = latlon_to_xy(pred_coord)
            real_coord = latlon_to_xy(real_coord)
            
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=real_coord[:,0], y=real_coord[:,1],label="Real trajectory", color="blue", marker="o", s=40, alpha=0.6)
            sns.scatterplot(x=pred_coord[:,0], y=pred_coord[:,1],label="Predicted trajectory", color="red", marker="o", s=40, alpha=0.6)
            plt.plot(real_coord[:,0], real_coord[:,1], color="blue", linewidth=2, alpha=0.7, label="Real trajectory")
            plt.plot(pred_coord[:,0], pred_coord[:,1], color="red", linewidth=2, alpha=0.7, label="Predicted trajectory")

            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.legend()
            plt.title("Position Result of Inverse Model")
            
            plt.xlim(-200, 200)
            plt.ylim(-150, 150)
            plt.tight_layout()
            plt.savefig(f'plot/PositionInverseResult{idx+1}.png')
            plt.close()
            break
        
for i in range(10):
    getSingleInverseResult(idx=i)