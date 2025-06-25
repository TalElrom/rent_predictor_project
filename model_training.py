import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from assets_data_prep import prepare_data

df = pd.read_csv('train.csv')

df_prepared = prepare_data(df, 'train')

X = df_prepared.drop('price', axis=1)  # Feature matrix
y = df_prepared['price']               # Target variable

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
    
# Step 2: Define base ElasticNet model (no CV here!)
elastic_net = ElasticNet(max_iter=10000)

# Step 3: Define hyperparameter grid to search over
param_grid = {
    'alpha': np.logspace(-3, 3, 20),           # Regularization strength
    'l1_ratio': np.linspace(0, 1, 11)         # Mix between Lasso (1.0) and Ridge (0.0)
}

# Step 4: Inner CV - for Grid Search hyperparameter tuning (5-fold)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)

# Step 5: Wrap model in GridSearchCV to tune hyperparameters using inner CV
"""
    We selected Grid Search for hyperparameter optimization because the hyperparameter space 
    for ElasticNet (alpha and l1_ratio) is relatively small and continuous. Grid Search allows 
    for a systematic evaluation of all combinations, ensuring we find the most suitable pair 
    without missing potential optima. RandomizedSearch is better for large spaces, 
    but here a full search is feasible and effective.
"""
grid_search = GridSearchCV(
    estimator=elastic_net,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='neg_root_mean_squared_error',  # RMSE, negated (because scikit-learn maximizes by default)
    n_jobs=-1                               # Use all CPU cores
)

# Step 6: Outer CV - for evaluating model performance with 10-fold cross-validation
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Step 7: Perform nested cross-validation
nested_scores = cross_val_score(
    grid_search,    # includes inner CV inside
    X, y,           # full dataset
    cv=outer_cv,    # outer loop CV
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

# Step 8: Print mean and std of outer CV scores (RMSE)
print("📊 Nested CV RMSE mean:", -np.mean(nested_scores))
print("📉 Nested CV RMSE std:", np.std(nested_scores))


# Step 1: Fit the final model on the entire dataset
grid_search.fit(X, y)
en_model = grid_search.best_estimator_

# Step 2: Save the best model to a file
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(en_model, f)

print("✅ Final model trained and saved successfully.")
print("Best alpha:", en_model.alpha)
print("Best l1_ratio:", en_model.l1_ratio)
# Step 1: Get coefficients from the model
coefs = en_model.coef_

# Step 2: Create a Series mapping feature names to their coefficients
coef_series = pd.Series(coefs, index=X.columns)

# Step 3: Sort by absolute value of coefficients (magnitude)
top_5_features = coef_series.reindex(coef_series.abs().sort_values(ascending=False).head(5).index)

# Step 4: Print results
print("🔥 Top 5 most influential features (by absolute coefficient):")
print(top_5_features)

# Predict on training set
y_pred = en_model.predict(X)

# Evaluate
print("📈 R²:", r2_score(y, y_pred))
print("📉 MAE:", mean_absolute_error(y, y_pred))
print("📉 RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

zero_coef_cols = X.columns[en_model.coef_ == 0]
print("🎯 Features with zero weight in ElasticNet:")
print(zero_coef_cols.tolist())