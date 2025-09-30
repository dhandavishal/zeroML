from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

def prf(y_true, y_pred): 
    return precision_recall_fscore_support(y_true, y_pred, average="binary")

def rocauc(y_true, scores): 
    return roc_auc_score(y_true, scores)

def cm(y_true, y_pred): 
    return confusion_matrix(y_true, y_pred)