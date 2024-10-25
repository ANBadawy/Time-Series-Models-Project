import os
import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from sklearn.ensemble import RandomForestRegressor


@csrf_exempt
def predict_next_value(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)

    try:
        request_body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON provided.'}, status=400)

    dataset_id = request_body.get('dataset_id')
    values = request_body.get('values')

    if not dataset_id or not values:
        return JsonResponse({'error': 'dataset_id and values are required.'}, status=400)


    rf_model_path = os.path.join(settings.BASE_DIR, r'E:\Assignments\Time Series\Time_Series_Project\data\New_Model',
                                     f"rf_model_{dataset_id}.joblib")
    xgb_model_path = os.path.join(settings.BASE_DIR, r'E:\Assignments\Time Series\Time_Series_Project\data\New_Model',
                                      f"xgb_model_{dataset_id}.joblib")

    try:
        rf_model = joblib.load(rf_model_path)
        xgb_model = joblib.load(xgb_model_path)
    except FileNotFoundError:
        return JsonResponse({'error': f'Model file not found for dataset_id: {dataset_id}'}, status=404)

    def create_features(df):
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['dayofweek'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        df['trend'] = np.arange(len(df))

        for lag in range(1, 6):
            df[f'lag{lag}'] = df['value'].shift(lag)

        df['rolling_mean_3'] = df['value'].shift(1).rolling(window=3).mean()
        df['rolling_std_3'] = df['value'].shift(1).rolling(window=3).std()
        df['rolling_max_7'] = df['value'].shift(1).rolling(window=7).max()
        df['rolling_min_7'] = df['value'].shift(1).rolling(window=7).min()
        df['rolling_median_7'] = df['value'].shift(1).rolling(window=7).median()

        df['S_day'] = np.sqrt(
            np.sin(2 * np.pi * df['hour'] / 24) ** 2 +
            np.cos(2 * np.pi * df['hour'] / 24) ** 2
        )
        df['S_week'] = np.sqrt(
            np.sin(2 * np.pi * df['dayofweek'] / 7) ** 2 +
            np.cos(2 * np.pi * df['dayofweek'] / 7) ** 2
        )

        return df.drop(columns=['time'])

    def get_feature_importance(X, y, model=None, top_n=6):
        if model is None:
            model = RandomForestRegressor(n_estimators=50, random_state=42)

        model.fit(X, y)
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        top_features = importance_df.head(top_n)

        return top_features['Feature'].tolist()

    df = pd.DataFrame(values)
    df['time'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={'value': 'value'}).sort_values('time')
    df = create_features(df).dropna()

    if df.empty:
        return JsonResponse({'error': 'Not enough data to make a prediction.'}, status=400)

    feature_cols = [
        'hour', 'minute', 'dayofweek', 'month', 'quarter', 'trend',
        'lag1', 'lag2', 'lag3', 'lag4', 'lag5',
        'rolling_mean_3', 'rolling_std_3', 'rolling_max_7', 'rolling_min_7', 'rolling_median_7'
        , 'S_day', 'S_week'
    ]

    X = df[feature_cols]
    y = df['value']

    top_features = get_feature_importance(X, y, top_n=6)
    X_top = X[top_features]

    latest_row = X_top.iloc[-1].values.reshape(1, -1)

    rf_features = min(rf_model.n_features_in_, X_top.shape[1])
    xgb_features = min(xgb_model.n_features_in_, X_top.shape[1])

    try:
        rf_prediction = rf_model.predict(latest_row[:, :rf_features])[0]
        xgb_prediction = xgb_model.predict(latest_row[:, :xgb_features])[0]
        ensemble_prediction = (rf_prediction + xgb_prediction) / 2
    except Exception as e:
        return JsonResponse({'error': f'Prediction error: {str(e)}'}, status=500)

    return JsonResponse({'Prediction': ensemble_prediction})
