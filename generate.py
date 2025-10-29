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
from cbam import CBAM
from collections import Counter
from geopy.distance import geodesic
import math
from utils import dms_to_decimal, EMInverseDataset
import seaborn as sns
from SpectralNet import FreqDriftModel,freq_to_class,class_to_freq
from condition_diffusion import DDPM,UNetWithCond

device="cuda" if torch.cuda.is_available() else "cpu"

freq_values = [10.0] + list(np.arange(24.0, 40.5, 0.5))  # 共34类
freq2idx = {v: i for i, v in enumerate(freq_values)}
idx2freq = {i: v for i, v in enumerate(freq_values)}
num_classes = len(freq_values)

def generateAFT():
    model = FreqDriftModel()
    pretrained_model_path = "freq_model.pth"
    model.load_state_dict(torch.load(pretrained_model_path))
    model = model.to(device)

    forward_model = DDPM(model=UNetWithCond(in_channels=1, cond_dim=256, n_feat=128), cond_dim=256,betas=(1e-4, 0.02), n_T=200, device='cuda')
    pretrained_model_path = "forward_model.pth"
    forward_model.load_state_dict(torch.load(pretrained_model_path))
    forward_model = forward_model.to(device)

    dataset = EMInverseDataset(root_path='dataset')
    criterion = nn.MSELoss()

    batch_size=8
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    label_min = dataset.label_min
    label_max = dataset.label_max
    all_preds = []  
    model.eval()
    with torch.no_grad():
        for idx, (x_img, recv, x_param, line_freq) in enumerate(train_loader):
            x_param = x_param.to(device)
            line_freq = line_freq.to(device)

            target_class = freq_to_class(line_freq*40)
            logits = model(x_param)  
            pred_class = logits.argmax(dim=-1)  
            
            pred_freq = class_to_freq(pred_class)
            pred_freq = pred_freq/40

            x_img_pred = forward_model.generate(pred_freq,enhance=False)  
            mean = dataset.input_mean
            std = dataset.input_std

            x_img_pred = F.interpolate(x_img_pred, size=(641, 300), mode='bilinear', align_corners=False)
            x_img_pred = x_img_pred.cpu().squeeze().numpy() * (std+1e-6) + mean  

            all_preds.append(x_img_pred)
            print('batch {} generated'.format(idx+1))
    all_preds = np.concatenate(all_preds, axis=0)

    np.save("dataset\pred_results.npy", all_preds)
    print("保存完成，shape:", all_preds.shape)
generateAFT()