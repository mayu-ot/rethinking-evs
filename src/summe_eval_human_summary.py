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

def summe_human_score(metric='mean'):
    gs_summary = get_summe_gssummary()
    num_videos = len(gs_summary)
    acc_mean = 0
    acc_min = 0
    acc_max = 0
    
    ds = []
    for item in gs_summary:
        gs_sum = item['gs_summary']
        
        N = len(gs_sum)

        gs_results = np.zeros((N,))
        
        for i in range(N):
            res = [f1_score(gs_sum[x], gs_sum[i]) for x in range(N) if x != i]
            
            if metric=='mean':
                gs_results[i] = np.mean(res)
            elif metric=='max':
                gs_results[i] = np.max(res)
            else:
                raise RuntimeError
        
        worst_human = gs_results.min()
        avr_human = gs_results.mean()
        best_human = gs_results.max()
        
        acc_mean += avr_human
        acc_min += worst_human
        acc_max += best_human
        
        ds.append({'video': item['video'],
                   'metric': metric,
                   'worst_human': worst_human,
                   'avg_human': avr_human,
                   'best_human': best_human})
        
    return pd.DataFrame(ds)

def main():
    df_mean = summe_human_score('mean')
    
    print("""
    Scores by taking the 'average' over all reference summaries
    (Reproduction of human scores in the original SumMe paper)
    """)
    print(df_mean[['worst_human', 'avg_human', 'best_human']].mean(axis=0))
    
    df_max = summe_human_score('max')
    
    print("""
    Scores by taking the 'maximum' over all reference summaries
    """)
    print(df_max[['worst_human', 'avg_human', 'best_human']].mean(axis=0))
    
    df = pd.concat([df_mean, df_max])
    df.to_csv('data/processed/summe_human_eval.csv')
    
if __name__=='__main__':
    main()