"""
Flexible Model Ensemble Module

- Supports averaging and stacking ensemble strategies
- Pluggable interface for any model with a compatible predict() or callable
- Optional meta-model (e.g. linear regression or LGBM) for stacking

Example usage:
    from models.ensemble import ModelEnsemble
    ens = ModelEnsemble()
    ens.add_model('lgb', lgbm_model.predict)
    ens.add_model('nn', nn_model.predict)
    preds = ens.predict(X, method='mean')  # simple average
    # Train stacker
    ens.fit_stack(X_val, y_val, model_preds=dict(lgb=lpreds, nn=npreds))
    preds = ens.predict(X_test, method='stack')
"""

import numpy as np
from sklearn.linear_model import RidgeCV

class ModelEnsemble:
    def __init__(self, weights=None):
        self.models = {}
        self.weights = weights  # dict of model_name->weight
        self.stacker = None
        self.model_names = []
    def add_model(self, name, predict_fn):
        self.models[name] = predict_fn
        self.model_names.append(name)
    def predict(self, X, model_preds=None, method='mean'):
        # model_preds: dict of {name: preds}, for custom dataset/scenario
        preds = []
        if model_preds is not None:
            for name in self.model_names:
                preds.append(np.array(model_preds[name]))
        else:
            for name, fn in self.models.items():
                p = fn(X)
                preds.append(np.array(p))
        stack = np.stack(preds, axis=0)  # shape: [n_models, n_samples]
        if method == 'mean':
            return np.mean(stack, axis=0)
        elif method == 'weighted':
            if self.weights is None:
                raise ValueError("Ensemble weights not set")
            wvec = np.array([self.weights[name] for name in self.model_names])[:,None]
            return np.sum(stack * wvec, axis=0) / np.sum(wvec)
        elif method == 'stack':
            if self.stacker is None:
                raise ValueError("Stacker not fit yet")
            # stack.T: [n_samples, n_models]
            return self.stacker.predict(stack.T)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    def fit_stack(self, X_val, y_val, model_preds=None):
        # learns meta-model using validation set preds
        preds = []
        if model_preds is not None:
            for name in self.model_names:
                preds.append(np.array(model_preds[name]))
        else:
            for name, fn in self.models.items():
                p = fn(X_val)
                preds.append(np.array(p))
        stackX = np.stack(preds, axis=1)  # [n_samples, n_models]
        self.stacker = RidgeCV(alphas=[0.1,1.0,10.0]).fit(stackX, y_val)
        return self
