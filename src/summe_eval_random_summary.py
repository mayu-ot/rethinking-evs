from tools.summarizer import summarize
from tools.io import load_summe_mat
from tools.segmentation import get_segment_summe
from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

def get_summe_gssummary():
    summe_data = load_summe_mat('data/raw/summe/GT/')
    
    gold_standard = []
    for item in summe_data:
        user_anno = item['user_anno']
        user_anno = user_anno.T
        user_anno = user_anno.astype(np.bool)
        
        gold_standard.append(
            {
                'gs_summary': user_anno,
                'video': item['video']
            }
        )
        
    return gold_standard

def eval_random_summary(seg_type='uniform', score_type='random', sum_len=.15, verbose=True):
    gt_summary = get_summe_gssummary()
    result = []
    seg_l_mode = 60
    for item in gt_summary:
        gs_summary = item['gs_summary']
        n_fr = gs_summary.shape[1]
        
        segment = get_segment_summe(n_fr, item['video'], method=seg_type)
        
        # generate random summary
        if score_type == 'constant':
            rand_score = np.ones((n_fr,)) * .5
        elif score_type == 'random':
            rand_score = np.random.random((n_fr,))

        rand_summary = summarize(rand_score, segment, int(n_fr * sum_len))
        
        score = [f1_score(x, rand_summary) for x in gs_summary]
        
        f1_min = min(score)
        f1_mean = sum(score) / len(score)
        f1_max = max(score)
        
        result.append((f1_min, f1_mean, f1_max))
        
        if verbose:
            print('%20s | %-.2f | %-.2f | %-.2f |' % (item['video'], f1_min * 100, f1_mean * 100, f1_max * 100))

    result = np.array(result)
    result_summary = result.mean(axis=0)
    
    if verbose:
        print('%s | %.2f | %.2f | %.2f |' % ('Avg.', result_summary[0] * 100, result_summary[1] * 100, result_summary[2] * 100))
    
    return result_summary

def main():
    score_list = []
    seg_type_list = []
    summary_type_list = []
    score_type = 'random'
    
    for seg_type  in ['one-peak', 'two-peak', 'KTS', 'randomized-KTS', 'uniform']:
        N = 100
        res = Parallel(n_jobs=-1)( [delayed(eval_random_summary)(seg_type=seg_type, score_type=score_type, verbose=False) for _ in range(N)] )
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

    df.to_csv('data/processed/summe_%s_eval.csv' % score_type)
    
    
if __name__=='__main__':
    main()