import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class TimeSeriesPredictor:
    def __init__(self, n_features=3, test_size=0.2, random_state=42):
        self.n_features = n_features
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

    def load_and_prepare_data(self, file_path, sheet_name=0):
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        if self.df.shape[1] <= self.n_features:
            raise ValueError(f"Insufficient columns: {self.df.shape[1]} found, {self.n_features + 1} required")
        return self.df

    def create_features_target(self):
        self.X = self.df.iloc[:, :self.n_features].values
        self.y = self.df.iloc[:, self.n_features].values
        self.feature_names = self.df.columns[:self.n_features].tolist()
        self.target_name = self.df.columns[self.n_features]
        return self.X, self.y

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess_data(self):
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return self.X_train_scaled, self.X_test_scaled

    def train_model(self, model=None):
        if model is None:
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            self.model = model
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_fitted = True
        self.y_train_pred = self.model.predict(self.X_train_scaled)
        return self.model
    def predict(self, X=None):
        if not self.is_fitted:
            raise ValueError("Model not trained")

        if X is None:
            self.y_pred = self.model.predict(self.X_test_scaled)
            return self.y_pred
        else:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
def main():
    excel_path = 'data.xlsx'
    predictor = TimeSeriesPredictor(n_features=3, test_size=0.4)
    predictor.load_and_prepare_data(excel_path)
    X, y = predictor.create_features_target()
    X_train, X_test, y_train, y_test = predictor.split_data()
    X_train_scaled, X_test_scaled = predictor.preprocess_data()
    predictor.train_model()
    y_pred = predictor.predict()

if __name__ == "__main__":
    main()
