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

def dms_to_decimal(dms):
    parts = dms.replace('″', '').split('°')
    degrees = float(parts[0])
    minutes_seconds = parts[1].split('′')
    
    minutes = float(minutes_seconds[0])
    seconds = float(minutes_seconds[1])
    
    return degrees + minutes/60 + seconds/3600

class EMInverseDataset(Dataset):
    def __init__(self, root_path='dataset'):

        node1_files = glob.glob(os.path.join(root_path, '*node1*.npy'))
        node1_data = [np.load(f) for f in node1_files]
        node1_data = np.concatenate(node1_data, axis=0)  
        node2_files = glob.glob(os.path.join(root_path, '*node2*.npy'))
        node2_data = [np.load(f) for f in node2_files]
        node2_data = np.concatenate(node2_data, axis=0)  
        self.train_data = np.concatenate([node1_data, node2_data], axis=0)  

        label_data = np.load(os.path.join(root_path, 'labels.npy'))  
        self.label_data = np.concatenate([label_data, label_data], axis=0)  

        freq_data_1 = np.load(os.path.join(root_path, 'relabel_1.npy'))  
        freq_data_2 = np.load(os.path.join(root_path, 'relabel_2.npy'))  
        self.freq_label = np.concatenate([freq_data_1, freq_data_2], axis=0)  

        recv1 = ("18°24′28.87445″", "110°03′13.0532″")
        recv2 = ("18°24′29.18965″", "110°03′13.21095″")
        recv3 = ("18°24′28.6564″", "110°03′12.0625″")

        first_receiver_1 = [dms_to_decimal(recv3[0]), dms_to_decimal(recv3[1])]
        second_receiver_1 = [dms_to_decimal(recv2[0]), dms_to_decimal(recv2[1])]
        first_receiver_2 = [dms_to_decimal(recv1[0]), dms_to_decimal(recv1[1])]
        second_receiver_2 = [dms_to_decimal(recv2[0]), dms_to_decimal(recv2[1])]
        first_receiver_3 = [dms_to_decimal(recv1[0]), dms_to_decimal(recv1[1])]
        second_receiver_3 = [dms_to_decimal(recv2[0]), dms_to_decimal(recv2[1])]

        receiver_position_node1 = (
            [first_receiver_1]*209 + [first_receiver_2]*101 + [first_receiver_3]*203
        )
        receiver_position_node2 = (
            [second_receiver_1]*209 + [second_receiver_2]*101 + [second_receiver_3]*203
        )
        self.receiver_position = np.array(receiver_position_node1 + receiver_position_node2)  

        assert self.train_data.shape[0] == self.label_data.shape[0] == self.receiver_position.shape[0]

        self.input_mean = float(np.mean(self.train_data))
        self.input_std = float(np.std(self.train_data))

        self.label_min = np.min(self.label_data, axis=(0, 1))  
        self.label_max = np.max(self.label_data, axis=(0, 1)) 

        self.freq_min = np.min(self.freq_label)  
        self.freq_max = np.max(self.freq_label)  

        self.recv_min = np.min(self.receiver_position, axis=0)  
        self.recv_max = np.max(self.receiver_position, axis=0)  
        self.norm_stats = {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'label_min': self.label_min,
            'label_max': self.label_max,
            'freq_min': self.freq_min,
            'freq_max': self.freq_max,
            'recv_min': self.recv_min,
            'recv_max': self.recv_max,
        }

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        target_len=256
        x = self.train_data[idx]  
        x = (x - self.input_mean) / (self.input_std + 1e-6)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  
        x = F.interpolate(x.unsqueeze(0), size=(target_len,target_len), mode='bilinear', align_corners=False).squeeze(0)  

        label = self.label_data[idx].astype(np.float32)  
        label[:,:3] = (label[:,:3] - self.label_min[:3]) / (self.label_max[:3] - self.label_min[:3] + 1e-6)
        label[:,3] = label[:,3]/40
        label = torch.tensor(label, dtype=torch.float32)

        orig_len = label.shape[0]
        target_len = 256
        orig_idx = torch.arange(target_len, dtype=torch.float32) / (target_len - 1) * (orig_len - 1)
        nearest_idx = torch.round(orig_idx).long().clamp(0, orig_len - 1)
        label = label[nearest_idx]

        recv = self.receiver_position[idx]  
        recv = (recv - self.recv_min) / (self.recv_max - self.recv_min + 1e-6)
        recv = torch.tensor(recv, dtype=torch.float32)

        freq = self.freq_label[idx]
        freq = freq/40.0
        freq = torch.tensor(freq, dtype=torch.float32)
        freq = freq[nearest_idx]
        return x, recv, label, freq 


def nearestUpsmaple(freq_batch, target_len=300):
    B, src_len = freq_batch.shape
    tgt_idx = torch.arange(target_len, dtype=torch.float32, device=freq_batch.device)
    tgt_idx = tgt_idx / (target_len - 1) * (src_len - 1)
    nearest_idx = torch.round(tgt_idx).long().clamp(0, src_len - 1)  
    freq_upsampled = freq_batch[:, nearest_idx]  
    return freq_upsampled