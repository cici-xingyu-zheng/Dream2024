
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from utils import *
import xgboost as xgb

# n_iter = 100
# cv = 10

def para_search(seed, X, y_true):
    # Define the search space 
    rf_param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True]
    }

    xgb_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0]
    }

    # Create models
    rf = RandomForestRegressor(random_state=seed)
    xgb_model = xgb.XGBRegressor(random_state=seed)

    # Perform Random Search with cross-validation for Random Forest
    rf_random = RandomizedSearchCV(estimator=rf, 
                                param_distributions=rf_param_dist, 
                                n_iter=50, 
                                cv=10, 
                                random_state=seed, 
                                n_jobs=-1)
    rf_random.fit(X, y_true)

    best_rf = rf_random.best_estimator_

    # Perform Random Search with cross-validation for XGBoost
    xgb_random = RandomizedSearchCV(estimator=xgb_model, 
                                    param_distributions=xgb_param_dist, 
                                    n_iter=100, 
                                    cv=10, 
                                    random_state=seed, 
                                    n_jobs=-1)
    xgb_random.fit(X, y_true)

    best_xgb = xgb_random.best_estimator_

    # Evaluate 
    rf_pred = best_rf.predict(X)
    rf_corr = np.corrcoef(rf_pred, y_true)[0, 1]
    rf_rmse = np.sqrt(mean_squared_error(y_true, rf_pred))

    xgb_pred = best_xgb.predict(X)
    xgb_corr = np.corrcoef(xgb_pred, y_true)[0, 1]
    xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_pred))

    print("Best Random Forest model:")
    print("Hyperparameters:", rf_random.best_params_)
    print("Correlation:", rf_corr)
    print("RMSE:", rf_rmse)
    print()
    print("Best XGBoost model:")
    print("Hyperparameters:", xgb_random.best_params_)
    print("Correlation:", xgb_corr)
    print("RMSE:", xgb_rmse)

    return  rf_random.best_params_, xgb_random.best_params_
