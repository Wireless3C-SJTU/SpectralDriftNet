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
from geopy.distance import geodesic
import math
from utils import dms_to_decimal, EMInverseDataset
import seaborn as sns
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2
 
 
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.model(x)
 
 
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)
 
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x
 
 
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)
 
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)



class ConditionEncoder(nn.Module):
    def __init__(self, freq_dim=256, cond_dim=128):
        super().__init__()
        self.cond_dim = cond_dim 
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, 512),
            nn.ReLU(),
            nn.Linear(512, cond_dim)
        )

    def forward(self, freq, useGuide=True):
        if not useGuide:
            return torch.zeros((freq.shape[0], self.cond_dim), device=freq.device)
        freq_feat = self.freq_proj(freq)          
        return freq                          
   
class FiLM(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)  
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)    
        return gamma * x + beta

class CrossAttention(nn.Module):
    def __init__(self, query_dim, cond_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=query_dim, kdim=cond_dim, vdim=cond_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x, cond):
        """
        x: [B, C, H, W] → reshape to [B, H*W, C] as query
        cond: [B, cond_dim]
        """
        B, C, H, W = x.shape
        x_ = x.view(B, C, -1).permute(0, 2, 1)  
        cond_ = cond.unsqueeze(1) 

        attn_out, _ = self.attn(query=x_, key=cond_, value=cond_)  
        out = self.norm(x_ + attn_out)  
        return out.permute(0, 2, 1).view(B, C, H, W)  

class UNetWithCond(nn.Module):
    def __init__(self, in_channels=1, cond_dim=128, n_feat=64):
        super().__init__()
        self.n_feat = n_feat
        self.cond_dim = cond_dim

        self.init_conv = ResidualConvBlock(in_channels, n_feat)

        self.down1 = UnetDown(n_feat, n_feat)             
        self.down2 = UnetDown(n_feat, 2 * n_feat)          
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)      
        self.down4 = UnetDown(4 * n_feat, 8 * n_feat)      
        self.down5 = UnetDown(8 * n_feat, 16 * n_feat)     
        self.down6 = UnetDown(16 * n_feat, 16 * n_feat)   
        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        self.temb1 = EmbedFC(1, 16 * n_feat)
        self.temb2 = EmbedFC(1, 16 * n_feat)
        self.temb3 = EmbedFC(1, 8 * n_feat)
        self.temb4 = EmbedFC(1, 4 * n_feat)
        self.temb5 = EmbedFC(1, 2 * n_feat)
        self.temb6 = EmbedFC(1, 1 * n_feat)

        self.down_attn1 = CrossAttention(n_feat, cond_dim)
        self.down_attn2 = CrossAttention(n_feat, cond_dim)
        self.down_attn3 = CrossAttention(2 * n_feat, cond_dim)
        self.down_attn4 = CrossAttention(4 * n_feat, cond_dim)
        self.down_attn5 = CrossAttention(8 * n_feat, cond_dim)
        self.down_attn6 = CrossAttention(16 * n_feat, cond_dim)

        self.attn1 = CrossAttention(16 * n_feat, cond_dim)
        self.attn2 = CrossAttention(16 * n_feat, cond_dim)
        self.attn3 = CrossAttention(8 * n_feat, cond_dim)
        self.attn4 = CrossAttention(4 * n_feat, cond_dim)
        self.attn5 = CrossAttention(2 * n_feat, cond_dim)
        self.attn6 = CrossAttention(1 * n_feat, cond_dim)

        self.up0 = nn.Sequential(
            ResidualConvBlock(16 * n_feat, 16 * n_feat),
            nn.GroupNorm(8, 16 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(32 * n_feat, 16 * n_feat)
        self.up2 = UnetUp(32 * n_feat, 8 * n_feat)
        self.up3 = UnetUp(16 * n_feat, 4 * n_feat)
        self.up4 = UnetUp(8 * n_feat, 2 * n_feat)
        self.up5 = UnetUp(4 * n_feat, n_feat)
        self.up6 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, padding=1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, padding=1),
        )

    def forward(self, x, t, cond):
        if cond is None:
            cond = torch.zeros(x.shape[0], self.cond_dim).to(x.device)

        x0 = self.init_conv(x)
        d1 = self.down_attn1(x0, cond)
        d1 = self.down1(d1)
        d2 = self.down_attn2(d1, cond)
        d2 = self.down2(d2)
        d3 = self.down_attn3(d2, cond)
        d3 = self.down3(d3)
        d4 = self.down_attn4(d3, cond)
        d4 = self.down4(d4)
        d5 = self.down_attn5(d4, cond)
        d5 = self.down5(d5)
        d6 = self.down_attn6(d5, cond)
        d6 = self.down6(d6)

        temb1 = self.temb1(t).view(-1, 16 * self.n_feat, 1, 1)
        temb2 = self.temb2(t).view(-1, 16 * self.n_feat, 1, 1)
        temb3 = self.temb3(t).view(-1, 8 * self.n_feat, 1, 1)
        temb4 = self.temb4(t).view(-1, 4 * self.n_feat, 1, 1)
        temb5 = self.temb5(t).view(-1, 2 * self.n_feat, 1, 1)
        temb6 = self.temb6(t).view(-1, 1 * self.n_feat, 1, 1)

        up = self.up0(d6)
        up = self.attn1(up + temb1, cond)
        up = self.up1(up, d6)
        up = self.attn2(up + temb2, cond)
        up = self.up2(up, d5)
        up = self.attn3(up + temb3, cond)
        up = self.up3(up, d4)
        up = self.attn4(up + temb4, cond)
        up = self.up4(up, d3)
        up = self.attn5(up + temb5, cond)
        up = self.up5(up, d2)
        up = self.attn6(up + temb6, cond)
        up = self.up6(up, d1)
        out = self.out(torch.cat([up, x0], dim=1))
        return out

class DDPM(nn.Module):
    def __init__(self, model, cond_dim,betas, n_T, device):
        super(DDPM, self).__init__()
        self.model = model.to(device)
        self.input_size=256
        self.cond_encoder = ConditionEncoder(freq_dim=self.input_size,cond_dim=cond_dim)
        for k, v in self.ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
 
        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()
        
 
    def ddpm_schedules(self, beta1, beta2, T):
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1 
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp() 
 
        sqrtab = torch.sqrt(alphabar_t) 
        oneover_sqrta = 1 / torch.sqrt(alpha_t)
        sqrtmab = torch.sqrt(1 - alphabar_t) 
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
 
        return {
            "alpha_t": alpha_t,  
            "oneover_sqrta": oneover_sqrta,  
            "sqrt_beta_t": sqrt_beta_t,  
            "alphabar_t": alphabar_t, 
            "sqrtab": sqrtab, 
            "sqrtmab": sqrtmab, 
            "mab_over_sqrtmab": mab_over_sqrtmab_inv, 
        }
    
    def forward(self, x, freq, drop_cond_prob=0.1,struct_loss_weight=0.1):
        B, _, H, T = x.shape 
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        drop_mask = (torch.rand(B, device=self.device) < drop_cond_prob).float().unsqueeze(1)
        cond = self.cond_encoder(freq,useGuide=True)
        cond_null = self.cond_encoder(freq,useGuide=False)
        cond = cond * (1.0 - drop_mask) + cond_null * drop_mask
        noise_pred = self.model(x_t, _ts / self.n_T, cond)

        x_recon = (
            self.oneover_sqrta[_ts, None, None, None] *
            (x_t - self.mab_over_sqrtmab[_ts, None, None, None] * noise_pred)
        )
        loss_mse = self.loss_mse(noise_pred, noise)
        struct_regions = []
        with torch.no_grad():
            mask = freq != (10/40)     
            valid_indices = torch.nonzero(mask, as_tuple=False)  
            if valid_indices.shape[0] == 0:
                return loss_mse  

            batch_ids = valid_indices[:, 0]              
            time_ids = valid_indices[:, 1]               
            freq_values = freq[batch_ids, time_ids]      
            freq_bins = (self.input_size-1 - freq_values * (self.input_size-1)).round().clamp(0, self.input_size-1).long()

        window_size = 4
        half_w = window_size // 2
        loss_struct = 0.0
        count = 0
        for offset in range(-half_w, half_w + 1):  
            offset_bins = (freq_bins + offset).clamp(0, self.input_size-1)
            pred_vals = x_recon[batch_ids, 0, offset_bins, time_ids]
            true_vals = x[batch_ids, 0, offset_bins, time_ids]
            loss_struct += F.mse_loss(pred_vals, true_vals)
            count += 1

        loss_struct = loss_struct / count  
        return loss_mse + struct_loss_weight * loss_struct
    
    @torch.no_grad()
    def generate(self, freq, guidance_weight=1.5,enhance=False):
        device = self.device
        if freq.ndim!=2:
            freq = freq.unsqueeze(0).to(device)
        
        batch = freq.shape[0]
        shape = (batch, 1, self.input_size, self.input_size)
        self.eval()
        x = torch.randn(shape).to(device)
                        
        B = freq.shape[0]
        T = freq.shape[1]
        F = self.input_size 
        init_freq_idx = (F - 1 - freq * (F - 1)).round().clamp(0, F - 1).long()  
        mask = (freq != (10 / 40))
        cond_img = torch.zeros((B, F, T), device=freq.device)
        batch_idx = torch.arange(B, device=freq.device).unsqueeze(1).expand(B, T)  
        time_idx  = torch.arange(T, device=freq.device).unsqueeze(0).expand(B, T) 
        cond_img[batch_idx[mask], init_freq_idx[mask], time_idx[mask]] = 5.0  
        cond_img = cond_img.unsqueeze(1)  
        
        x = x + cond_img
        cond = self.cond_encoder(freq,useGuide=True)
        uncond = self.cond_encoder(freq,useGuide=False)

        for t in reversed(range(1, self.n_T + 1)):
            t_tensor = torch.full((1,), t, dtype=torch.float32, device=device) / self.n_T
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            eps_cond = self.model(x, t_tensor, cond)
            eps_uncond = self.model(x, t_tensor, uncond)
            eps_theta = (1 + guidance_weight) * eps_cond - guidance_weight * eps_uncond
            x = (
                self.oneover_sqrta[t] * (x - self.mab_over_sqrtmab[t] * eps_theta)
            ) + self.sqrt_beta_t[t] * z
        x_new = x.clone()  
        B, _, H, W = x_new.shape
        
        if enhance:
            time_kernel_size = 15
            time_sigma = 1.5
            t_offsets = torch.arange(-(time_kernel_size // 2), time_kernel_size // 2 + 1)
            time_weights = torch.exp(-0.5 * (t_offsets / time_sigma) ** 2)
            time_weights /= time_weights.max() 

            freq_kernel_size = 3
            freq_sigma = 0.8
            f_offsets = torch.arange(-(freq_kernel_size // 2), freq_kernel_size // 2 + 1)
            freq_weights = torch.exp(-0.5 * (f_offsets / freq_sigma) ** 2)
            freq_weights = freq_weights / freq_weights[freq_kernel_size // 2].clone()
            for b in range(B):
                freq_b = freq[b].cpu().numpy()
                above = freq_b > (20/40)
                segments = []
                start = None
                for t in range(W):
                    if above[t]:
                        if start is None:
                            start = t
                    else:
                        if start is not None and (t - start) >= 5:
                            segments.append((start, t - 1))
                        start = None
                if start is not None and (W - start) >= 5:
                    segments.append((start, W - 1))

                for seg_start, seg_end in segments:
                    for dt in t_offsets:
                        t = torch.arange(seg_start, seg_end + 1) + dt.item()
                        t = t[(t >= 0) & (t < W)]  
                        if len(t) == 0:
                            continue
                        weight_t = time_weights[dt + time_kernel_size // 2].item()
                        for tt in t:
                            f_val = freq[b, tt].item()
                            if f_val <= 0 or f_val > 1.0:
                                continue
                            freq_idx = int((1 - f_val) * (H - 1))

                            for df in f_offsets:
                                i = freq_idx + df.item()
                                if 0 <= i < H:
                                    weight_f = freq_weights[df + freq_kernel_size // 2].item()
                                    boost = weight_t * weight_f * 2  
                                    x_new[b, 0, i, tt] += boost
                                    x_new[b, 0, i, tt] = min(x_new[b, 0, i, tt], 1.0)
        return x_new
    
    @torch.no_grad()
    def cal_snr(self,freq,x,x_mean,x_std):
        device = self.device
        x = x.squeeze(1).to(device)
        batch = freq.shape[0]
        self.eval()
        T = freq.shape[0]
        F = self.input_size  
        init_freq_idx = (F - 1 - freq * (F - 1)).round().clamp(0, F - 1).long().to(device)  # [T]
        mask = (freq != (10 / 40))
        time_idx  = torch.arange(T, device=freq.device)
        signal_mag_norm = x[0, init_freq_idx[mask], time_idx[mask]].mean()
    
        vals=[]
        for f, t in zip(init_freq_idx[mask], time_idx[mask]):
            f_start = max(f - 5, 0)
            f_end = f  
            local_mean = x[0, f_start:f_end, t].mean()
            vals.append(local_mean) 
        
        if len(vals) == 0:
            return 0

        noise_mag_norm = torch.stack(vals).mean()
        signal_mag = signal_mag_norm * (x_std+1e-6) + x_mean
        noise_mag = noise_mag_norm * (x_std+1e-6) + x_mean
        snr = 10**(abs(signal_mag-noise_mag))
        return snr
        
        
def drawAFT(a_list_1,a_list_2,epoch,struct_regions=None,save_fig_dir='forward_AFT',max_amp=-7,min_amp=-10,mode='val',draw_batch_id=None):
    os.makedirs(save_fig_dir, exist_ok=True)
    input_size = a_list_1.shape[1]
    fig, axes = plt.subplots(1,2,figsize=(10, 4))
    freq_bins = a_list_1.shape[0]
    time_steps = a_list_1.shape[1]
    extent = [0, time_steps, 0, 40]
    
    a_list_1 = np.flipud(a_list_1)
    a_list_2 = np.flipud(a_list_2)
    axes[0].imshow(a_list_1, cmap='jet', aspect='auto',vmin=min_amp,vmax=max_amp,interpolation='nearest',extent=extent, origin='lower')
    axes[0].set_title("Ground Truth")
    axes[1].imshow(a_list_2, cmap='jet', aspect='auto',vmin=min_amp,vmax=max_amp,interpolation='nearest',extent=extent, origin='lower')
    axes[1].set_title("Prediction")
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    if struct_regions is not None:
        time_list = []
        freq_list = []
        for (batch_id,freq_value,time_id) in struct_regions:
            if isinstance(batch_id, torch.Tensor):
                batch_id = batch_id.item()
            if batch_id == draw_batch_id:
                if isinstance(freq_value, torch.Tensor):
                    freq_value = freq_value.item()
                if isinstance(time_id, torch.Tensor):
                    time_id = time_id.item()

                time_list.append(time_id)
                freq_hz = freq_value * 40
                freq_list.append(freq_hz)
                
        target_len = 256
        time_list_restore = torch.round(torch.tensor(time_list, dtype=torch.float32) / (target_len - 1) * (time_steps - 1)).cpu().numpy()
        axes[0].scatter(time_list_restore, freq_list, c='blue', s=10, marker='o', alpha=0.8)
    plt.tight_layout()  
    plt.savefig(os.path.join(save_fig_dir, "epoch{}_{}.png".format(epoch+1,mode)))

def train_diffusion_model_with_val(model, batch_size=32, epochs=50, lr=2e-6, val_ratio=0.1, device='cuda'):
    pretrained_model_path = 'forward_model.pth'
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        print('pretrained model loaded')
    else:
        print('pretrained model not found')
    model = model.to(device)
    full_dataset = EMInverseDataset(root_path='dataset')
    train_ratio = 0.9
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print(f" Dataset loaded: {len(train_dataset)} train / {len(val_dataset)} val")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    draw_batch_id=0 

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Train")
        for train_idx,(x_img, recv_coord, x_param, freq) in enumerate(train_loop):
            B = x_img.size(0)
            x_img = x_img.to(device)
            x_param = x_param.to(device)
            recv_coord = recv_coord.to(device)
            freq = freq.to(device)

            loss = model(x_img,freq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * B
            train_loop.set_postfix(train_loss=loss.item())
            
            if train_idx == 0 and epoch%4==0:
                mean = full_dataset.input_mean
                std = full_dataset.input_std
                x_pred = model.generate(freq[draw_batch_id])  
                x_true = F.interpolate(x_img[0].unsqueeze(0), size=(641, 300), mode='bilinear', align_corners=False) 
                x_pred = F.interpolate(x_pred, size=(641, 300), mode='bilinear', align_corners=False)
                x_true = x_true.cpu().squeeze().numpy() * (std+1e-6) + mean     
                x_pred = x_pred[0].cpu().squeeze().numpy() * (std+1e-6) + mean  
                drawAFT(x_true,x_pred,epoch, mode='train')
        avg_train_loss = total_train_loss / train_size

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_idx, (x_img, recv_coord, x_param,freq) in enumerate(val_loader):
                B = x_img.size(0)

                x_img = x_img.to(device)
                x_param = x_param.to(device)
                recv_coord = recv_coord.to(device)
                freq = freq.to(device)
                loss = model(x_img,freq)
                total_val_loss += loss.item() * B

                if val_idx == 0 and epoch%4==0:
                    mean = full_dataset.input_mean
                    std = full_dataset.input_std
                    x_pred = model.generate(freq[0]) 
                    x_true = F.interpolate(x_img[0].unsqueeze(0), size=(641, 300), mode='bilinear', align_corners=False)
                    x_pred = F.interpolate(x_pred, size=(641, 300), mode='bilinear', align_corners=False)
                    snr = model.cal_snr(freq[0],x_pred,mean,std)
                    print('snr={}'.format(snr))
                    x_true = x_true.cpu().squeeze().numpy() * (std+1e-6) + mean     
                    x_pred = x_pred[0].cpu().squeeze().numpy() * (std+1e-6) + mean  
                    drawAFT(x_true,x_pred,epoch, mode='val')
        avg_val_loss = total_val_loss / val_size

        
def generate_result(model, batch_size=32,val_ratio=0.1, device='cuda'):
    pretrained_model_path = 'forward_model.pth'
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        print('pretrained model loaded')
    else:
        print('pretrained model not found')
                
    model = model.to(device)
        
    full_dataset = EMInverseDataset(root_path='dataset')

    train_ratio = 0.9
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    with torch.no_grad():
        for val_idx, (x_img, recv_coord, x_param,freq) in enumerate(val_loader):
            B = x_img.size(0)

            x_img = x_img.to(device)
            x_param = x_param.to(device)
            recv_coord = recv_coord.to(device)
            freq = freq.to(device)
            
            mean = full_dataset.input_mean
            std = full_dataset.input_std
            x_pred = model.generate(freq,enhance=False) 
            x_true = F.interpolate(x_img, size=(641, 300), mode='bilinear', align_corners=False)
            x_pred = F.interpolate(x_pred, size=(641, 300), mode='bilinear', align_corners=False)
            x_true = x_true.cpu().squeeze().numpy() * (std+1e-6) + mean     
            x_pred = x_pred.cpu().squeeze().numpy() * (std+1e-6) + mean  

            for i in range(B):
                drawAFT(x_true[i],x_pred[i],epoch=i,save_fig_dir='forward_result')
                real_freq = freq[i].detach().cpu().numpy()
                real_freq = real_freq*40
                plt.figure(figsize=(10, 5))
                sns.lineplot(x=list(range(len(real_freq))), y=real_freq, color='blue', label='Real', linewidth=3, alpha=0.5)
                plt.xlabel("time(s)")
                plt.ylabel("freqency(hz)")
                plt.ylim(0, 40)  
                plt.title("Frequency of single window sample")
                plt.tight_layout()
                plt.savefig('forward_result/real_freq_{}.png'.format(i+1))
                print('{} done'.format(i))
            break
  
model = DDPM(model=UNetWithCond(in_channels=1, cond_dim=256, n_feat=128), cond_dim=256,betas=(1e-4, 0.02), n_T=200, device='cuda')
# train_diffusion_model_with_val(model,batch_size=8,epochs=100, lr=1e-6)
# generate_result(model,batch_size=8)

