import numpy as np

def threshold_for_fpr(scores_val, target_fpr=0.02):
    return np.quantile(scores_val, 1.0 - target_fpr)