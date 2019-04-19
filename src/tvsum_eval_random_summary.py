import scipy.io as sio
from tools.summarizer import summarize
from tools.io import load_tvsum_mat
from tools.segmentation import get_segment_tvsum
from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import os

def eval_random_summary(seg_type='uniform', score_type='random', sum_len=.15, use_sum=False, verbose=True):
    tvsum_data = load_tvsum_mat('./data/raw/tvsum/ydata-tvsum50.mat')
    result = []
    seg_l_mode = 60
    for item in tvsum_data:
        user_anno = item['user_anno'].T
        n_fr = user_anno.shape[1]
        
        segment = get_segment_tvsum(n_fr, item['video'], method=seg_type)
        
        # generate human summary
        cache_dir = 'data/interim/%s%s/'%('sum'*use_sum, seg_type)
            
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
        
        # generate random summary
        if score_type == 'random':
            rand_score = np.random.random((n_fr,))
        else:
            rand_score = np.ones((n_fr,)) * .5
        
        rand_summary = summarize(rand_score, segment, int(n_fr * sum_len), use_sum=use_sum)
        
        score = [f1_score(x, rand_summary) for x in gs_summary]
        
        f1_min = min(score)
        f1_mean = sum(score) / len(score)
        f1_max = max(score)
        
        result.append((f1_min, f1_mean, f1_max))
        
        if verbose:
            print('%s | %.2f | %.2f | %.2f |' % (item['video'], f1_min * 100, f1_mean * 100, f1_max * 100))
    
    result = np.array(result)
    result_summary = result.mean(axis=0)
    
    if verbose:
        print('%s | %.2f | %.2f | %.2f |' % ('Avg.', result_summary[0] * 100, result_summary[1] * 100, result_summary[2] * 100))
    
    return result_summary

def run(use_sum=False):
    score_list = []
    seg_type_list = []
    summary_type_list = []
    score_type = 'random'
    
    debug = True

    for seg_type in ['one-peak', 'two-peak', 'KTS', 'randomized-KTS', 'uniform']:
        N = 100
        res = Parallel(n_jobs=-1)( [delayed(eval_random_summary)(seg_type=seg_type, score_type=score_type, use_sum=use_sum, verbose=False) for _ in range(N)] )
        res = np.vstack(res)
        seg_type_list += [seg_type] * N
        summary_type_list += ['random'] * N

        mean_score = res.mean(axis=0)
        print('%15s | %.2f | %.2f | %.2f |' % (seg_type, mean_score[0] * 100, mean_score[1] * 100, mean_score[2] * 100))

        score_list.append(res)

    score_arr = np.vstack(score_list)

    df = pd.DataFrame({'summary_type': summary_type_list,
                      'seg_type': seg_type_list,
                      'min': score_arr[:, 0],
                      'mean': score_arr[:, 1],
                      'max': score_arr[:, 2]})

    df.to_csv('data/processed/tvsum_%s_%seval.csv' % (score_type, 'sum_'*use_sum))
    
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sum', action="store_true")
    args = parser.parse_args()
    run(use_sum=args.use_sum)