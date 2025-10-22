# ===自訂義設定黨===
import numpy as np, joblib
from catboost import CatBoostClassifier

class WeightedVoter:
    def __init__(self, cat_path, rf_path, weights=(1, 2)):
        self.cat_path = cat_path
        self.rf_path = rf_path
        self.w = np.array(weights, dtype=float)
        self.w = self.w / self.w.sum()
        self._load_models()

    def _load_models(self):
        self.cat = CatBoostClassifier()
        self.cat.load_model(self.cat_path)
        self.rf = joblib.load(self.rf_path)
        # （可選）保存類別順序以便檢查對齊
        self.classes_ = getattr(self.rf, "classes_", None)

    def predict_proba(self, X):
        p_cat = self.cat.predict_proba(X)
        p_rf  = self.rf.predict_proba(X)
        return self.w[0]*p_cat + self.w[1]*p_rf

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
# ===自訂義設定黨===