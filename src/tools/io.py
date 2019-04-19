import hdf5storage
import json
import tables
import numpy as np
import os
import scipy.io as sio

def load_summe_mat(dirname):
    mat_list = os.listdir(dirname)

    data_list = []
    for mat in mat_list:
        data = sio.loadmat(os.path.join(dirname, mat))
        
        item_dict = {
        'video': mat[:-4],
        'length': data['video_duration'],
        'nframes': data['nFrames'],
        'user_anno': data['user_score'],
        'gt_score': data['gt_score']
        }
        
        data_list.append((item_dict))
    
    return data_list

def load_tvsum_mat(filename):
    data = hdf5storage.loadmat(filename, variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()
    
    data_list = []
    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item
        
        item_dict = {
        'video': video[0, 0],
        'category': category[0, 0],
        'title': title[0, 0],
        'length': length[0, 0],
        'nframes': nframes[0, 0],
        'user_anno': user_anno,
        'gt_score': gt_score
        }
        
        data_list.append((item_dict))
    
    return data_list