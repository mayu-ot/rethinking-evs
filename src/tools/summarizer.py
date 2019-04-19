import numpy as np
from . import knapsack_solver as knapsack_solver

def summarize(score, segment, capacity, use_sum=False):
        # generate summary
        score = np.asarray(score).ravel()
        f_idx = np.zeros_like(score)
        
        score = np.split(score, segment)
        score = list(filter(lambda x: x.size, score)) # remove empty elements
        
        f_idx = np.split(f_idx, segment)
        f_idx = list(filter(lambda x: x.size, f_idx)) # remove empty elements
        
        weights = [x.size for x in score]
        
        if use_sum:
            values = [x.sum() for x in score]
        else:
            values = [x.mean() for x in score]
        
        _, selected_cut = knapsack_solver.knapsack([(v, w) for v, w in zip(values, weights)], capacity)
        for si in selected_cut:
            f_idx[si][:] = 1
        
        return np.hstack(f_idx)