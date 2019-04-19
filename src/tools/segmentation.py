import numpy as np
from tools.io import load_tvsum_mat
import scipy.io as sio

def get_segment_summe(N, video_id, method='one-peak'):
    if method == 'one-peak':
        return poisson_segment(N)
    elif method == 'two-peak':
        return two_peak_segment(N, lam_1=30, lam_2=90)
    elif method == 'KTS':
        return summe_kts_segment(video_id)
    elif method == 'randomized-KTS':
        return summe_random_kts_segment(video_id)
    elif method == 'uniform':
        return np.arange(0, N, 60).astype('i')
    else:
        raise RuntimeError
        
def get_segment_tvsum(N, video_id, method='one-peak'):
    if method == 'one-peak':
        return poisson_segment(N)
    elif method == 'two-peak':
        return two_peak_segment(N, lam_1=30, lam_2=90)
    elif method == 'KTS':
        return tvsum_kts_segment(video_id)
    elif method == 'randomized-KTS':
        return tvsum_random_kts_segment(video_id)
    elif method == 'uniform':
        return np.arange(0, N, 60).astype('i')
    else:
        raise RuntimeError
        
def get_summe_video2idx():
    videos = ['Air_Force_One',
    'Base jumping',
    'Bearpark_climbing',
    'Bike Polo',
    'Bus_in_Rock_Tunnel',
    'Car_railcrossing',
    'Cockpit_Landing',
    'Cooking',
    'Eiffel Tower',
    'Excavators river crossing',
    'Fire Domino',
    'Jumps',
    'Kids_playing_in_leaves',
    'Notre_Dame',
    'Paintball',
    'Playing_on_water_slide',
    'Saving dolphines',
    'Scuba',
    'St Maarten Landing',
    'Statue of Liberty',
    'Uncut_Evening_Flight',
    'Valparaiso_Downhill',
    'car_over_camera',
    'paluma_jump',
    'playing_ball']
    return {videos[i]:i for i in range(len(videos))}

def get_tvsum_video2idx():
    tvsum_data = load_tvsum_mat('./data/raw/tvsum/ydata-tvsum50.mat')
    return {tvsum_data[x]['video']: x for x in range(len(tvsum_data))}

def get_random_segmentation(N, step):
    shot_boundaries = np.arange(0, N, step)
    shot_boundaries += np.random.randint(-15, 15, shot_boundaries.size)
    shot_boundaries[0] = 0
    shot_boundaries[-1] = N
    return shot_boundaries

def two_peak_segment(n_fr, lam_1=40, lam_2=80):
    segment = []
    cur_pos = 0
    while(True):
        cur_pos += np.random.choice([np.random.poisson(lam_1), np.random.poisson(lam_2)])
        if cur_pos > n_fr-1:
            break
        segment.append(cur_pos)
    return segment
    
def poisson_segment(n_fr, lam=60):
    segment = []
    cur_pos = 0
    while(True):
        cur_pos += np.random.poisson(lam)
        if cur_pos > n_fr-1:
            break
        segment.append(cur_pos)
    return segment

def tvsum_kts_segment(v_id):
    return _tvsum_shot_boundaries[_tvsum_video2idx[v_id]][0].ravel()

def tvsum_random_kts_segment(v_id):
    boundaries = _tvsum_shot_boundaries[_tvsum_video2idx[v_id]][0].ravel()
    seg_l = [boundaries[i+1] - boundaries[i] for i in range(boundaries.size - 1)]
    seg_l = np.random.permutation(seg_l)
    boundaries = np.add.accumulate(seg_l)
    return np.hstack(([0], boundaries))

def summe_kts_segment(v_id):
    return _summe_shot_boundaries[_summe_video2idx[v_id]][0].ravel()

def summe_random_kts_segment(v_id):
    boundaries = _summe_shot_boundaries[_summe_video2idx[v_id]][0].ravel()
    seg_l = [boundaries[i+1] - boundaries[i] for i in range(boundaries.size - 1)]
    seg_l = np.random.permutation(seg_l)
    boundaries = np.add.accumulate(seg_l)
    return np.hstack(([0], boundaries))

_tvsum_video2idx = get_tvsum_video2idx()
_tvsum_shot_boundaries = sio.loadmat('data/raw/tvsum/shot_TVSum.mat')['shot_boundaries']

_summe_video2idx = get_summe_video2idx()
_summe_shot_boundaries = sio.loadmat('data/raw/summe/shot_SumMe.mat')['shot_boundaries']