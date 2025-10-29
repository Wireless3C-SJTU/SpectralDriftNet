import numpy as np
import os
from pymongo import MongoClient
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import time
import pickle
from scipy import signal
import cv2
import pymongo

mongo_host = 'localhost'  
mongo_port = 27017  
database_name = 'inversion_dataset'  

client = MongoClient(mongo_host, mongo_port)
db = client[database_name]

def low_pass(data, critical_frequencies = 30):
    b, a = signal.butter(5, 2 * critical_frequencies / 1024, "low")
    data = signal.lfilter(b, a, data)
    return data

def FFT(data, critical_frequencies = 30):
    l = len(data)
    window = np.hanning(l)
    res = data * window 
    frequency_list = 1024 / 2 * np.linspace(0, 1, len(data) // 2 + 1) 
    cut_f = np.where(frequency_list > critical_frequencies)[0][0]
    frequency = frequency_list[:cut_f]

    res = low_pass(data, critical_frequencies) 
    res = np.fft.fft(res)[:cut_f] / l * 2 
    amplitude = np.abs(res) 
    return frequency, amplitude

def get_AFT_offline(start_time = "2023-6-21 15:42:21", end_time = "2023-6-21 16:10:44", col_name = "first", channel = "channel1", critical_frequencies = 30, windowsize = 16, seconds = 1, interval = 1):
    col = db[col_name]
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    time_d = int(((end_time - start_time).total_seconds() - windowsize) // seconds + 1)  
    if time_d <= 0:
        print("时间长度不够")
        return {"error": "时间长度不够{}s".format(windowsize)}
    end_time = start_time + timedelta(seconds = time_d * seconds + windowsize)  
    print("find -- start_time: {}, end_time: {}".format(start_time, end_time))
    
    ptr = col.find({
        "time": {
            "$gte": start_time,
            "$lt": end_time,
        }
    })

    channel_list = []  
    time_list = [] 
    cnt = 0
    for data in ptr:
        if cnt == 0:
            print("result -- start time: {}".format(data["time"]))
        channel_list.append(data[channel])
        cnt += 1
        if cnt == (time_d * seconds + windowsize) * 1024: 
            print("end time: {}".format(data["time"]))
            break
    res = None

    if len(channel_list) == 0:
        return {
            "error": "No data in DB"
        }

    for i in range(time_d + 1):
        time_list.append(
            datetime.strftime(start_time + timedelta(seconds=i * seconds), "%Y-%m-%d %H:%M:%S"))
        f, a = FFT(channel_list[1024 * (i * seconds): 1024 * (windowsize + i * seconds)], critical_frequencies) 
        a = np.log10(a / interval / 500) 
        
        if res is None:
            res = a
        else:
            res = np.c_[res, a]

    data_list = []
    for i in range(time_d + 1):
        tmp=[]
        for j in range(len(f)):
            tmp.append(res[j, i])
        data_list.append(tmp)
    data_list = np.array(data_list,dtype=np.float)
    
    res_json = {
        "xList": time_list,
        "yList": f.tolist(),
        "dataList": data_list
    }
    return time_list,f.tolist(),data_list

def drawAFT(t_list, f_list, a_lists_1,a_lists_2, windowsize,max_amp,min_amp,start_time):
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    start_time = start_time.strftime('%Y%m%d_%H%M%S') 
    fig, ax = plt.subplots(figsize=(16, 10))
    x_ticks = np.arange(len(t_list))
    x_labels = t_list
    ax.set_xticks(x_ticks)

    if len(x_ticks) > 960:
        x_ticks_visible = x_ticks[::64]
    elif len(x_ticks) > 480:
        x_ticks_visible = x_ticks[::32]
    elif len(x_ticks) > 240:
        x_ticks_visible = x_ticks[::16]
    elif len(x_ticks) > 120:
        x_ticks_visible = x_ticks[::8]
    elif len(x_ticks) > 60:
        x_ticks_visible = x_ticks[::4]
    elif len(x_ticks) > 30:
        x_ticks_visible = x_ticks[::2]
    else:
        x_ticks_visible = x_ticks

    ax.set_xticks(x_ticks_visible)
    ax.set_xticklabels([x_labels[i] for i in x_ticks_visible], rotation=90)

    f_list = f_list*2
    y_ticks = np.flip(np.arange(len(f_list)))
    y_ticks_visible = y_ticks[::int(5*len(f_list)/f_list[-1]/2)]
    y_labels_visible = f_list[::int(5*len(f_list)/f_list[-1]/2)]
    ax.set_yticks(y_ticks_visible)
    ax.set_yticklabels(y_labels_visible)

    for y in y_ticks_visible:
        ax.axhline(y=y, linestyle='dashed', color='gray', linewidth=0.5)
    for x in x_ticks_visible:
        ax.axvline(x=x, linestyle='dashed', color='gray', linewidth=0.5)
    a_lists_1 = np.transpose(a_lists_1)
    new_order =  np.flip(np.arange(a_lists_1.shape[0]))   
    a_lists_1 = a_lists_1[new_order]    
    a_lists_2 = np.transpose(a_lists_2)
    new_order =  np.flip(np.arange(a_lists_2.shape[0]))   
    a_lists_2 = a_lists_2[new_order]  
    
    a_list = np.vstack((a_lists_1, a_lists_2)) 
    im = ax.imshow(a_list, cmap='jet', aspect='auto',vmin=min_amp,vmax=max_amp,interpolation='nearest')
    cbar = plt.colorbar(im, label='Amplitude')

    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.tight_layout() 
    plt.savefig('AFTresult\{}.png'.format(start_time))
    return
    
if __name__ == '__main__':
    start_time = "2025-4-28 15:14:00"
    end_time = "2025-4-28 15:51:20"
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')  
    end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    time_interval = 1
    time_delta = timedelta(hours=time_interval) 

    windowsize=16
    stop_freq=40
    interval1=8.82
    interval2=154.26
    channel='channel1'
    seconds=1
    max_amp=-6.5
    min_amp=-9.5
    
    start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
    t_list, f_list, a_lists_1 = get_AFT_offline(start_time = start_time, end_time = end_time, col_name="first", windowsize=windowsize,critical_frequencies=stop_freq,interval=interval1,channel=channel,seconds=seconds)
    t_list, f_list, a_lists_2 = get_AFT_offline(start_time = start_time, end_time = end_time, col_name="second", windowsize=windowsize,critical_frequencies=stop_freq,interval=interval2,channel=channel,seconds=seconds)
    print(a_lists_1.max())
    print(a_lists_1.min())
    drawAFT(t_list, f_list, a_lists_1,a_lists_2, windowsize,max_amp,min_amp,start_time)

