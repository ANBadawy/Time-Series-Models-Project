import os
import joblib
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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


    rf_model_path = os.path.join(settings.BASE_DIR, r'E:\Assignments\Time Series\Time_Series_Project\data\Models', f"rf_model_{dataset_id}.joblib")
    xgb_model_path = os.path.join(settings.BASE_DIR, r'E:\Assignments\Time Series\Time_Series_Project\data\Models', f"xgb_model_{dataset_id}.joblib")

    rf_model = joblib.load(rf_model_path)
    xgb_model = joblib.load(xgb_model_path)


    def create_features(df):
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['dayofweek'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter

        # Lag features
        df['lag1'] = df['value'].shift(1)
        df['lag2'] = df['value'].shift(2)
        df['lag3'] = df['value'].shift(3)
        df['lag4'] = df['value'].shift(4)
        df['lag5'] = df['value'].shift(5)

        # Rolling mean and std features
        df['rolling_mean_3'] = df['value'].shift(1).rolling(window=3).mean()
        df['rolling_std_3'] = df['value'].shift(1).rolling(window=3).std()

        return df

    feature_cols = ['hour', 'minute', 'dayofweek', 'month', 'quarter',
                    'lag1', 'lag2', 'lag3', 'lag4', 'lag5',
                    'rolling_mean_3', 'rolling_std_3']

    # Process input data and make predictions
    df = pd.DataFrame(values)
    df['time'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={'value': 'value'}).sort_values('time')
    df = create_features(df).dropna()

    if df.empty:
        return JsonResponse({'error': 'Not enough data to make a prediction.'}, status=400)

    latest_row = df.iloc[-1]
    X_new = latest_row[feature_cols].values.reshape(1, -1)

    # Make predictions
    rf_prediction = rf_model.predict(X_new)[0]
    xgb_prediction = xgb_model.predict(X_new)[0]
    ensemble_prediction = (rf_prediction + xgb_prediction) / 2

    return JsonResponse({'Prediction': ensemble_prediction})


