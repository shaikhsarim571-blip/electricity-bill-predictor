import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, df, config):
        self.df = df.copy()
        self.config = config
        self.scaler = MinMaxScaler() if config.NORMALIZATION_METHOD == "minmax" else StandardScaler()
        
    def handle_missing_data(self):
        """Fill missing values using seasonal median"""
        df = self.df.copy()
        
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
        elif 'month' in df.columns:
            pass
        else:
            return df
            
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().sum() > 0:
                seasonal_medians = df.groupby('month')[col].transform('median')
                df[col].fillna(seasonal_medians, inplace=True)
                df[col].fillna(df[col].median(), inplace=True)
                
        return df
    
    def detect_outliers(self, column):
        """IQR-based outlier detection"""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - self.config.OUTLIER_IQR_MULTIPLIER * IQR
        upper = Q3 + self.config.OUTLIER_IQR_MULTIPLIER * IQR
        
        return (self.df[column] < lower) | (self.df[column] > upper)
    
    def remove_outliers(self, columns):
        """Remove rows with outliers"""
        outlier_mask = pd.Series([False] * len(self.df))
        
        for col in columns:
            if col in self.df.columns:
                outlier_mask |= self.detect_outliers(col)
        
        initial_len = len(self.df)
        self.df = self.df[~outlier_mask].reset_index(drop=True)
        removed = initial_len - len(self.df)
        
        return self.df, removed
    
    def normalize_features(self, columns):
        """Normalize specified columns"""
        df = self.df.copy()
        
        for col in columns:
            if col in df.columns:
                df[col] = self.scaler.fit_transform(df[[col]])
        
        return df
    
    def extract_temporal_features(self):
        """Generate temporal features from date"""
        df = self.df.copy()
        
        if 'date' not in df.columns:
            return df
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                      9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df['season'] = df['month'].map(season_map)
        
        return df
    
    def get_correlation_matrix(self, numeric_cols=None):
        """Calculate correlation matrix for numeric features"""
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        return self.df[numeric_cols].corr()
    
    def decompose_timeseries(self, column, frequency=12):
        """Seasonal decomposition"""
        if len(self.df) < 2 * frequency:
            return None
        
        df = self.df.set_index('date') if 'date' in self.df.columns else self.df
        df = df.sort_index()
        
        if column not in df.columns:
            return None
        
        try:
            decomposition = seasonal_decompose(df[column], model='additive', period=frequency)
            return decomposition
        except Exception as e:
            print(f"Decomposition failed: {str(e)}")
            return None
    
    def prepare_ml_data(self, target_col, feature_cols):
        """Prepare data for ML models"""
        df = self.handle_missing_data()
        df = self.extract_temporal_features()
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_col].fillna(df[target_col].mean())
        
        return X, y, df


def create_sample_dataset(n_months=36):
    """Generate synthetic Pakistan electricity data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2021-01-01', periods=n_months, freq='ME')
    
    base_consumption = np.linspace(100, 150, n_months)
    seasonal_pattern = 50 * np.sin(np.arange(n_months) * 2 * np.pi / 12)
    noise = np.random.normal(0, 10, n_months)
    consumption = base_consumption + seasonal_pattern + noise
    consumption = np.clip(consumption, 30, 500)
    
    temp = 20 + 15 * np.sin(np.arange(n_months) * 2 * np.pi / 12) + np.random.normal(0, 2, n_months)
    humidity = 60 + 20 * np.sin(np.arange(n_months) * 2 * np.pi / 12 + np.pi/4) + np.random.normal(0, 5, n_months)
    humidity = np.clip(humidity, 20, 100)
    
    df = pd.DataFrame({
        'date': dates,
        'unit_consumption_kwh': consumption,
        'temperature': temp,
        'humidity': humidity,
        'rainfall': np.random.exponential(2, n_months),
        'wind_speed': np.random.gamma(2, 2, n_months)
    })
    
    df['bill_amount'] = consumption * 22 + np.random.normal(0, 50, n_months)
    df['bill_amount'] = np.clip(df['bill_amount'], 500, 15000)
    
    return df
