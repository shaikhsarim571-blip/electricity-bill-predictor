import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
# from prophet import Prophet  # Commented out for Streamlit Cloud compatibility
import joblib
import warnings

warnings.filterwarnings('ignore')


class PredictiveModels:
    def __init__(self, config):
        self.config = config
        self.rf_model = None
        self.xgb_model = None
        self.prophet_model = None
        
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest Regressor"""
        self.rf_model = RandomForestRegressor(**self.config.RANDOM_FOREST_PARAMS)
        self.rf_model.fit(X_train, y_train)
        return self.rf_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost Regressor"""
        self.xgb_model = xgb.XGBRegressor(**self.config.XGBOOST_PARAMS)
        self.xgb_model.fit(X_train, y_train, verbose=False)
        return self.xgb_model
    
    def train_prophet(self, df, target_col='unit_consumption_kwh', date_col='date'):
        """Placeholder for Prophet - disabled for Streamlit Cloud"""
        # Prophet disabled for Streamlit Cloud compatibility
        return None
    
    def predict(self, model_type, X_test):
        """Generate predictions"""
        if model_type == 'rf':
            return self.rf_model.predict(X_test)
        elif model_type == 'xgb':
            return self.xgb_model.predict(X_test)
        else:
            raise ValueError(f"Unknown model: {model_type}")
    
    def forecast_prophet(self, periods=3):
        """Forecast future values - Prophet disabled for Streamlit Cloud"""
        return None  # Prophet disabled for Streamlit Cloud compatibility
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def get_feature_importance(self, model_type, feature_names):
        """Extract feature importance"""
        if model_type == 'rf' and self.rf_model:
            importances = self.rf_model.feature_importances_
        elif model_type == 'xgb' and self.xgb_model:
            importances = self.xgb_model.feature_importances_
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self, path):
        """Save trained models"""
        if self.rf_model:
            joblib.dump(self.rf_model, f"{path}/rf_model.pkl")
        if self.xgb_model:
            joblib.dump(self.xgb_model, f"{path}/xgb_model.pkl")
    
    def load_models(self, path):
        """Load pre-trained models"""
        try:
            self.rf_model = joblib.load(f"{path}/rf_model.pkl")
        except:
            pass
        try:
            self.xgb_model = joblib.load(f"{path}/xgb_model.pkl")
        except:
            pass


class ModelComparison:
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def compare_models(self, X_train, X_test, y_train, y_test):
        """Train and compare multiple models"""
        models = PredictiveModels(self.config)
        
        models.train_random_forest(X_train, y_train)
        rf_pred = models.predict('rf', X_test)
        rf_metrics = models.evaluate(y_test, rf_pred)
        self.results['Random Forest'] = rf_metrics
        
        models.train_xgboost(X_train, y_train)
        xgb_pred = models.predict('xgb', X_test)
        xgb_metrics = models.evaluate(y_test, xgb_pred)
        self.results['XGBoost'] = xgb_metrics
        
        return self.results
    
    def get_best_model(self):
        """Return model with highest R2"""
        if not self.results:
            return None
        
        best_model = max(self.results.items(), key=lambda x: x[1]['r2'])
        return best_model[0]
    
    def to_dataframe(self):
        """Convert results to DataFrame"""
        return pd.DataFrame(self.results).T.round(4)
