
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_metrics(y_true, y_prob):
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "aupr": float(average_precision_score(y_true, y_prob)),
    }
