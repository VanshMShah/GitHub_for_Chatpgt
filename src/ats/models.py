from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.ats.arima import fit_arima_mean_model
from src.ats.gjr_garch import fit_gjr_garch, one_step_variance_forecast

@dataclass
class Forecast:
    mean: float
    p_up: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    volatility: float
    metadata: dict[str, Any]

class PersistenceExpert:
    name = "persistence"
    def fit(self, y, X=None):
        self.scale = float(np.std(y[-60:], ddof=1) or 1e-6); return self
    def predict(self, X=None, last_return=0.0):
        s = self.scale
        return Forecast(0.0,0.5,-1.645*s,-0.674*s,0,0.674*s,1.645*s,s,{"expert":self.name})

class LogisticExpert:
    name = "logistic"
    def fit(self, X, y):
        yy=(y>0).astype(int); self.model=make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=500)); self.model.fit(X,yy); self.scale=float(np.std(y[-60:],ddof=1) or 1e-6); return self
    def predict(self,X):
        p=float(self.model.predict_proba(X[-1:])[0,1]); s=self.scale
        return Forecast(0.0,p,-1.645*s,-0.674*s,0,0.674*s,1.645*s,s,{"expert":self.name})

class QuantileGBDT:
    name="lightgbm_compatible"
    def fit(self,X,y):
        self.models={q:HistGradientBoostingRegressor(loss="quantile",quantile=q,max_depth=3,max_iter=120,learning_rate=0.05,l2_regularization=0.2,random_state=7).fit(X,y) for q in (0.05,0.25,0.5,0.75,0.95)}
        self.direction=make_pipeline(StandardScaler(),LogisticRegression(C=0.4,max_iter=500)).fit(X,(y>0).astype(int))
        return self
    def predict(self,X):
        q={k:float(m.predict(X[-1:])[0]) for k,m in self.models.items()}; p=float(self.direction.predict_proba(X[-1:])[0,1]); v=max((q[0.95]-q[0.05])/3.29,1e-5)
        return Forecast(q[0.5],p,q[0.05],q[0.25],q[0.5],q[0.75],q[0.95],v,{"expert":self.name,"point_estimate":"median"})

class ARIMAGARCHExpert:
    name="arima_gjr_garch"
    def fit(self,y):
        ar=fit_arima_mean_model(np.asarray(y,dtype=float),max_p=2,max_q=2,criterion="aic")
        ga=fit_gjr_garch(ar["residuals"])
        self.mean=float(ar["mean_forecast"]); self.g=ga; self.res=np.asarray(ar["residuals"]); return self
    def predict(self):
        p=self.g["params"]; s=float(np.sqrt(one_step_variance_forecast(self.res,self.g))); qs=[]
        from src.ats.gjr_garch import hansen_skewed_t_ppf
        for q in (0.05,0.25,0.5,0.75,0.95): qs.append(self.mean+s*hansen_skewed_t_ppf(q,p["nu"],p["lambda"]))
        from src.ats.gjr_garch import hansen_skewed_t_cdf
        probability = 1.0 - hansen_skewed_t_cdf(-self.mean/s, p["nu"], p["lambda"])
        return Forecast(self.mean, float(np.clip(probability, 0.001, 0.999)), *qs, s, {"expert":self.name,"garch_persistence":self.g["persistence"]})

def enforce_quantiles(f: Forecast) -> Forecast:
    numeric = [f.mean,f.p_up,f.q05,f.q25,f.q50,f.q75,f.q95,f.volatility]
    if not np.isfinite(numeric).all() or not 0 <= f.p_up <= 1 or f.volatility <= 0:
        raise ValueError("Forecast contains invalid values")
    vals=np.maximum.accumulate([f.q05,f.q25,f.q50,f.q75,f.q95]); return Forecast(f.mean,f.p_up,*map(float,vals),f.volatility,f.metadata)

def blend(forecasts: dict[str,Forecast], weights: dict[str,float]) -> Forecast:
    keys=[k for k in forecasts if weights.get(k,0)>0]; w=np.array([weights[k] for k in keys],dtype=float); w/=w.sum()
    arr=lambda attr: float(sum(wi*getattr(forecasts[k],attr) for wi,k in zip(w,keys)))
    meta={"experts":keys,"point_estimate":"blended point estimate","interval_coverage":0.90}
    return enforce_quantiles(Forecast(arr("mean"),np.clip(arr("p_up"),0.001,0.999),arr("q05"),arr("q25"),arr("q50"),arr("q75"),arr("q95"),max(arr("volatility"),1e-5),meta))
