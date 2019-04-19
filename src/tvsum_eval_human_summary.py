import os
import scipy.io as sio
from tools.summarizer import summarize
from tools.io import load_tvsum_mat
from tools.segmentation import get_segment_tvsum
from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

def get_tvsum_gssummary(seg_type='uniform', sum_len=.15, use_sum=False):
    tvsum_data = load_tvsum_mat('./data/raw/tvsum/ydata-tvsum50.mat')
    
    gold_standard = []
    
    for item in tvsum_data:
        user_anno = item['user_anno']
        user_anno = user_anno.T
        
        n_fr = user_anno.shape[1]
        
        segment = get_segment_tvsum(n_fr, item['video'], method=seg_type)
            
        # generate human summary
        cache_dir = 'data/interim/gs/%s%s/'%('sum'*use_sum, seg_type)
            
        if seg_type in ['uniform', 'KTS']:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            if os.path.exists(cache_dir + '%s.npy'% item['video']):
                gs_summary = np.load(cache_dir + '%s.npy'% item['video'])
            else:
                gs_summary = [summarize(x, segment, int(n_fr * sum_len), use_sum=use_sum) for x in user_anno]
                gs_summary = np.vstack(gs_summary)
                np.save(cache_dir + '%s.npy'% item['video'], gs_summary)
        else:
            gs_summary = [summarize(x, segment, int(n_fr * sum_len), use_sum=use_sum) for x in user_anno]
            gs_summary = np.vstack(gs_summary)
            
        gold_standard.append(
            {
                'gs_summary': gs_summary,
                'video': item['video'],
                'category': item['category']
            }
        )
        
    return gold_standard

def eval_human_summary(seg_type='uniform', sum_len=.15, use_sum=False, verbose=False):
    gs_summary = get_tvsum_gssummary(seg_type, sum_len, use_sum=use_sum)
    num_videos = len(gs_summary)
    acc_mean = 0
    acc_min = 0
    acc_max = 0
    
    for item in gs_summary:
        gs_sum = item['gs_summary']
        
        N = len(gs_sum)

        gs_results = np.zeros((N,))
        
        gs_results_mean = np.zeros((N,))
        gs_results_max = np.zeros((N,))
        gs_results_min = np.zeros((N,))
        
        for i in range(N):
            res = [f1_score(gs_sum[x], gs_sum[i]) for x in range(N) if x != i]
            
            gs_results_mean[i] = sum(res) / len(res)
            gs_results_max[i] = max(res)
            gs_results_min[i] = min(res)
            
        st_min = gs_results_min.mean()
        st_mean = gs_results_mean.mean()
        st_max = gs_results_max.mean()

        acc_mean += st_mean
        acc_min += st_min
        acc_max += st_max
            
        if verbose:
            print('%20s | %.2f | %.2f | %.2f |' % (item['video'][:20], st_min * 100, st_mean * 100, st_max * 100))
            
    if verbose:       
        print('%20s | %.2f | %.2f | %.2f |' % ('Avr.', acc_min / num_videos * 100, acc_mean / num_videos * 100, acc_max / num_videos * 100))
        
    return np.asarray([acc_min / num_videos, acc_mean / num_videos, acc_max / num_videos])
    

def run():
    score_list = []
    seg_type_list = []
    summary_type_list = []

    for seg_type in ['one-peak', 'two-peak', 'KTS', 'randomized-KTS', 'uniform']:
        scores = []

        if seg_type in ['uniform', 'KTS']:
            scores = eval_human_summary(seg_type=seg_type, verbose=False)
            scores = scores[None, :]
            seg_type_list.append(seg_type)
            summary_type_list.append('human')
            
        else:
            N = 100
            res = Parallel(n_jobs=-1)( [delayed(eval_human_summary)(seg_type=seg_type, verbose=False) for _ in range(N)] )
            seg_type_list += [seg_type] * N
            summary_type_list += ['human'] * N
            scores = np.vstack(res)

        mean_score = scores.mean(axis=0)
        
        print('%15s | %.2f | %.2f | %.2f |' % (seg_type, mean_score[0] * 100, mean_score[1] * 100, mean_score[2] * 100))

        score_list.append(scores)

    score_arr = np.vstack(score_list)

    df = pd.DataFrame({'summary_type': summary_type_list,
                      'seg_type': seg_type_list,
                      'min': score_arr[:, 0],
                      'mean': score_arr[:, 1],
                      'max': score_arr[:, 2]})

    df.to_csv('data/processed/tvsum_human_eval.csv')
    
    
if __name__=='__main__':
    run()