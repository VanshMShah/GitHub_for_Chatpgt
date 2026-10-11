from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def walk_forward_direction(X: np.ndarray, y: np.ndarray, min_train: int = 126, step: int = 1) -> dict:
    X=np.asarray(X,float); y=np.asarray(y,float)
    if len(X)!=len(y) or len(X)<=min_train: raise ValueError('insufficient aligned observations')
    predictions=[]; actual=[]; probs=[]
    for i in range(min_train,len(X),step):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=400))
        model.fit(X[:i],(y[:i]>0).astype(int)); p=float(model.predict_proba(X[i:i+1])[0,1]); predictions.append(int(p>=.5)); probs.append(p); actual.append(int(y[i]>0))
    if not actual: return {'n':0}
    a=np.asarray(actual); p=np.asarray(probs); eps=1e-12
    return {'n':len(a),'accuracy':float((a==np.asarray(predictions)).mean()),'brier':float(np.mean((p-a)**2)),'log_loss':float(-np.mean(a*np.log(np.clip(p,eps,1))+(1-a)*np.log(np.clip(1-p,eps,1))))}
