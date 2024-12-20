# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedHousePricePredictor:
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.model = self._get_model()
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_regression, k='all')
        self.selected_features = None
        
    def _get_model(self):
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        else:
            raise ValueError("Unsupported model type")
    
    def prepare_data(self, data):
        """
        Prepare the data by handling missing values and encoding categorical variables
        """
        df = data.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_columns)
        
        return df
    
    def train(self, X, y, feature_selection=True, k=10):
        """
        Train the model with the given data
        """
        if feature_selection:
            X = self.select_features(X, y, k=k)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if self.selected_features:
            X = X[self.selected_features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance with modified cross-validation
        """
        predictions = self.predict(X)
        metrics = {
            'MSE': mean_squared_error(y, predictions),
            'RMSE': np.sqrt(mean_squared_error(y, predictions)),
            'R2': r2_score(y, predictions)
        }
        
        # Use KFold for cross-validation
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            X_scaled = self.scaler.transform(X)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train and evaluate on fold
                self.model.fit(X_train, y_train)
                val_pred = self.model.predict(X_val)
                cv_scores.append(r2_score(y_val, val_pred))
            
            metrics['CV_Mean_R2'] = np.mean(cv_scores)
            metrics['CV_Std_R2'] = np.std(cv_scores)
        except Exception as e:
            print(f"Warning: Cross-validation failed: {str(e)}")
            metrics['CV_Mean_R2'] = None
            metrics['CV_Std_R2'] = None
        
        return metrics
    
    def select_features(self, X, y, k=10):
        """
        Select top k features based on F-regression scores
        """
        self.feature_selector.set_params(k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        return X_selected
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler
        """
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)

    def plot_feature_importance(self):
        """
        Plot feature importance or coefficients depending on model type
        """
        plt.figure(figsize=(10, 6))
        
        if self.model_type == 'linear':
            # For linear regression, use coefficients
            if hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
                feature_names = self.selected_features if self.selected_features else range(len(importance))
                importance_dict = dict(zip(feature_names, importance))
                
                # Create DataFrame and sort
                importance_df = pd.DataFrame.from_dict(
                    importance_dict, orient='index', columns=['Coefficient']
                )
                importance_df.sort_values(by='Coefficient', ascending=True, inplace=True)
                
                # Plot
                importance_df.plot(kind='barh')
                plt.title('Feature Coefficients (Absolute Values)')
                plt.xlabel('Coefficient Magnitude')
            else:
                plt.text(0.5, 0.5, 'Model not trained yet', 
                        horizontalalignment='center', verticalalignment='center')
        
        elif self.model_type in ['random_forest', 'xgboost']:
            # For tree-based models, use feature_importances_
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = self.selected_features if self.selected_features else range(len(importance))
                importance_dict = dict(zip(feature_names, importance))
                
                # Create DataFrame and sort
                importance_df = pd.DataFrame.from_dict(
                    importance_dict, orient='index', columns=['Importance']
                )
                importance_df.sort_values(by='Importance', ascending=True, inplace=True)
                
                # Plot
                importance_df.plot(kind='barh')
                plt.title('Feature Importance')
                plt.xlabel('Importance Score')
            else:
                plt.text(0.5, 0.5, 'Model not trained yet', 
                        horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(0.5, 0.5, 'Feature importance not available for this model type', 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        return plt.gcf()

    def plot_predictions(self, X, y):
        """
        Plot actual vs predicted values
        """
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predictions')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_residuals(self, X, y):
        """
        Plot residuals analysis
        """
        predictions = self.predict(X)
        residuals = y - predictions
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals vs Predicted
        ax1.scatter(predictions, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        
        # Residuals distribution
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residuals')
        
        plt.tight_layout()
        return fig
        
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model
        """
        instance = cls()
        saved_objects = joblib.load(filepath)
        instance.model = saved_objects['model']
        instance.scaler = saved_objects['scaler']
        return instance