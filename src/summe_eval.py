from tools.summarizer import summarize
from tools.io import load_summe_mat
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import json
from joblib import Parallel, delayed
import os

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

def get_random_summary(N, segment, budget):
    rand_score = np.random.random((N,))
    rand_summary = summarize(rand_score, segment, int(N * budget))
    return rand_summary

def evaluate_baseline(in_file, verbose=True):
    results = json.load(open(in_file))
    gt_summary = get_summe_gssummary()
    
    b_score = []
    
    for item in gt_summary:
        gs_summary = item['gs_summary']
        N = gs_summary.shape[1]
        segment = results[item['video']]['segment']
        
        rand_summary = get_random_summary(N, segment, budget=0.15)
        
        f1_scores = [f1_score(x, rand_summary) for x in gs_summary]
        f1_min = min(f1_scores)
        f1_mean = sum(f1_scores) / len(f1_scores)
        f1_max = max(f1_scores)
        
        b_score.append((f1_min, f1_mean, f1_max))
        
        if verbose:
            print('%25s | %6.2f | %6.2f | %6.2f |' % (item['video'], f1_min * 100, f1_mean * 100, f1_max * 100))
        
    b_score = np.array(b_score)
    score_summary = b_score.mean(axis=0)
    
    if verbose:
        print('%25s | %6.2f | %6.2f | %6.2f |' % ('Avg.', score_summary[0] * 100, score_summary[1] * 100, score_summary[2] * 100))
    
    return {'method': 'Random',
            'min': score_summary[0],
            'avg': score_summary[1],
            'max': score_summary[2]}
        
def evaluate(in_file, name=None, verbose=True):
    results = json.load(open(in_file))
    gt_summary = get_summe_gssummary()
    
    score = []
    baseline_score = []
    
    for item in gt_summary:
        gs_summary = item['gs_summary']
        N = gs_summary.shape[1]
        
        summary = results[item['video']]['summary']
        f1_scores = [f1_score(x, summary) for x in gs_summary]
        
        f1_min = min(f1_scores)
        f1_mean = sum(f1_scores) / len(f1_scores)
        f1_max = max(f1_scores)
        
        score.append((f1_min, f1_mean, f1_max))
        
        if verbose:
            print('%25s | %6.2f | %6.2f | %6.2f |' % (item['video'], f1_min * 100, f1_mean * 100, f1_max * 100))

    score = np.array(score)
    score_summary = score.mean(axis=0)
    
    if verbose:
        print('%25s | %6.2f | %6.2f | %6.2f |' % ('Avg.', score_summary[0] * 100, score_summary[1] * 100, score_summary[2] * 100))
    
    if name is None:
        name = in_file 
        
    return {'method': name,
            'min': score_summary[0],
            'avg': score_summary[1],
            'max': score_summary[2]}

def run(in_file):
    score_summary = evaluate(in_file, verbose=True)
    
    print('evaluating baseline scores')
    N = 100
    res = Parallel(n_jobs=-1)( [delayed(evaluate_baseline)(in_file, verbose=False) for _ in range(N)] )
    res.append(score_summary)
    df = pd.DataFrame(res)
    print(df[df.method=='Random'][['min', 'avg', 'max']].describe())
    
    out_file = 'data/processed/'+os.path.basename(in_file)+'.eval.csv'
    print(f'writing the results in {out_file}')
    df.to_csv(out_file,
             index=False)
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    args = parser.parse_args()
    run(in_file=args.in_file)