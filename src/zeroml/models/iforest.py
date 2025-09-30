from sklearn.ensemble import IsolationForest
import joblib

class IForestModel:
    def __init__(self, **kw): 
        self.m = IsolationForest(**kw)
    
    def fit(self, X): 
        self.m.fit(X)
        return self
    
    def score(self, X): 
        return (-self.m.score_samples(X))
    
    def save(self, path): 
        joblib.dump(self.m, path)
    
    @classmethod
    def load(cls, path): 
        obj = cls()
        obj.m = joblib.load(path)
        return obj