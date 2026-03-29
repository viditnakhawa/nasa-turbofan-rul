import json

with open(r'c:\Users\Sunil\.vscode\NASA CMaps\notebooks\03_Classical_ML.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Simplify RF Grid (cell 5)
rf_code = """rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_param_grid = {'n_estimators': [100], 'max_depth': [10, None]}

rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=1)
rf_grid.fit(X_train, y_train)

print(f"Random Forest Best Params: {rf_grid.best_params_}")
rf_model = rf_grid.best_estimator_

# Cross Validation Scores on Training Data
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1)
rf_cv_rmse = -rf_cv_scores
print(f"Random Forest CV RMSE: Mean = {rf_cv_rmse.mean():.2f}, Std = {rf_cv_rmse.std():.2f}")

# Predict on Test Set
rf_preds = rf_model.predict(X_test_last)

# Metrics
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_nasa = nasa_score(y_test.values, rf_preds)

print(f"Random Forest Test RMSE: {rf_rmse:.2f}")
print(f"Random Forest Test MAE: {rf_mae:.2f}")
print(f"Random Forest Test NASA Score: {rf_nasa:.2f}")"""
nb['cells'][5]['source'] = [line + '\n' for line in rf_code.split('\n')]
nb['cells'][5]['source'][-1] = nb['cells'][5]['source'][-1].rstrip('\n')


# Simplify XGB Grid (cell 7)
xgb_code = """xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
xgb_param_grid = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3, 5]}

xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=1)
xgb_grid.fit(X_train, y_train)

print(f"XGBoost Best Params: {xgb_grid.best_params_}")
xgb_model = xgb_grid.best_estimator_

# Cross Validation Scores on Training Data
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1)
xgb_cv_rmse = -xgb_cv_scores
print(f"XGBoost CV RMSE: Mean = {xgb_cv_rmse.mean():.2f}, Std = {xgb_cv_rmse.std():.2f}")

# Predict on Test Set
xgb_preds = xgb_model.predict(X_test_last)

# Metrics
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_nasa = nasa_score(y_test.values, xgb_preds)

print(f"XGBoost Test RMSE: {xgb_rmse:.2f}")
print(f"XGBoost Test MAE: {xgb_mae:.2f}")
print(f"XGBoost Test NASA Score: {xgb_nasa:.2f}")"""
nb['cells'][7]['source'] = [line + '\n' for line in xgb_code.split('\n')]
nb['cells'][7]['source'][-1] = nb['cells'][7]['source'][-1].rstrip('\n')

with open(r'c:\Users\Sunil\.vscode\NASA CMaps\notebooks\03_Classical_ML.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Simplified Grid Search.")
