from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib, numpy as np

class OCSVMModel:
    def __init__(self, **kw): 
        self.m = make_pipeline(StandardScaler(with_mean=False), OneClassSVM(**kw))
    
    def fit(self, X): 
        self.m.fit(X)
        return self
    
    def score(self, X): 
        return (-self.m.decision_function(X))
    
    def save(self, path): 
        joblib.dump(self.m, path)
    
    @classmethod
    def load(cls, path): 
        obj = cls()
        obj.m = joblib.load(path)
        return obj