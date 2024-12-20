import pandas as pd
from sklearn.model_selection import train_test_split
from model import *
from app import *


def main():
    # Load your dataset
    data = pd.read_csv('house_data.csv')
    
    # Separate features and target
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Try different models
    models = ['linear', 'random_forest', 'xgboost']
    best_model = None
    best_score = float('-inf')
    
    for model_type in models:
        predictor = AdvancedHousePricePredictor(model_type=model_type)
        
        # Prepare the data
        X_train_prepared = predictor.prepare_data(X_train)
        X_test_prepared = predictor.prepare_data(X_test)
        
        # Train and evaluate
        predictor.train(X_train_prepared, y_train, feature_selection=True, k=10)
        metrics = predictor.evaluate(X_test_prepared, y_test)
        
        print(f"\nModel: {model_type}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
        
        # Save the best model
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model = predictor           
        # Generate and save plots
        predictor.plot_feature_importance()
        plt.savefig(f'feature_importance_{model_type}.png')
        plt.close()
        
        predictor.plot_residuals(X_test_prepared, y_test)
        plt.savefig(f'residuals_{model_type}.png')
        plt.close()
    
    # Save the best model
    best_model.save_model('best_house_price_model.joblib')
    
    # Start the Flask application
    # Start the Flask application
    predictor = best_model
    app.run(debug=True)

    # Inside your main() function, replace or add after the other plots:
    fig = predictor.plot_residuals(X_test_prepared, y_test)
    plt.savefig(f'residuals_{model_type}.png')
    plt.close()

if __name__ == "__main__":
    main()