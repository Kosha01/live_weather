import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_weather_data():
    """Create comprehensive weather dataset with realistic patterns"""
    print("Creating comprehensive weather dataset...")
    
    np.random.seed(42)
    n_samples = 10000  # Increased sample size
    
    # Generate realistic weather data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
    
    # Temperature with realistic seasonal and daily patterns
    day_of_year = dates.dayofyear
    hour_of_day = dates.hour
    
    # Seasonal temperature variation
    seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    # Daily temperature variation
    daily_temp_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    # Random variation
    temp_noise = np.random.normal(0, 3, n_samples)
    
    temp = seasonal_temp + daily_temp_variation + temp_noise
    temp = np.clip(temp, -10, 45)  # Realistic temperature range
    
    # Humidity (inversely related to temperature with some randomness)
    base_humidity = 70 - 0.8 * (temp - 20)
    humidity_noise = np.random.normal(0, 15, n_samples)
    humidity = base_humidity + humidity_noise
    humidity = np.clip(humidity, 20, 95)
    
    # Pressure (realistic atmospheric pressure)
    pressure_base = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365)
    pressure_noise = np.random.normal(0, 8, n_samples)
    pressure = pressure_base + pressure_noise
    pressure = np.clip(pressure, 980, 1040)
    
    # Wind speed (seasonal variation + randomness)
    wind_base = 8 + 5 * np.sin(2 * np.pi * (day_of_year + 120) / 365)
    wind_noise = np.random.exponential(4, n_samples)
    wind_speed = wind_base + wind_noise
    wind_speed = np.clip(wind_speed, 0, 60)
    
    # Create rolling averages
    temp_series = pd.Series(temp)
    humidity_series = pd.Series(humidity)
    
    temp_rolling_3 = temp_series.rolling(3, min_periods=1).mean().values
    humidity_rolling_3 = humidity_series.rolling(3, min_periods=1).mean().values
    
    # Create realistic rain target based on multiple factors
    # Rain more likely when: high humidity, low pressure, moderate temperature
    rain_probability = (
        0.3 * (humidity > 75).astype(float) +
        0.3 * (pressure < 1005).astype(float) +
        0.2 * ((temp > 10) & (temp < 30)).astype(float) +
        0.2 * (wind_speed > 15).astype(float)
    )
    
    # Add some randomness to rain
    rain_random = np.random.random(n_samples)
    rain_target = (rain_probability > rain_random).astype(int)
    
    # Temperature forecast (next day with some variation)
    temp_trend = np.random.normal(0, 2, n_samples)  # Random temperature change
    temp_next_day = temp + temp_trend
    temp_next_day = np.clip(temp_next_day, -10, 45)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temp': temp,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'temp_rolling_3': temp_rolling_3,
        'humidity_rolling_3': humidity_rolling_3,
        'rain_target': rain_target,
        'temp_next_day': temp_next_day,
        'month': dates.month,
        'day_of_year': day_of_year,
        'season': ((dates.month % 12) // 3 + 1),
        'hour': dates.hour
    })
    
    print(f"Created dataset with {len(df)} samples")
    print(f"Rain target distribution: {df['rain_target'].value_counts().to_dict()}")
    print(f"Rain percentage: {df['rain_target'].mean():.2%}")
    
    return df

def train_models():
    """Train both classification and regression models"""
    print("=" * 60)
    print("WEATHER PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create or load data
    data_path = 'data/historical_weather.csv'
    
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        df = pd.read_csv(data_path)
        if len(df) < 1000:  # If data is too small, recreate
            print("Existing data too small, creating new dataset...")
            df = create_comprehensive_weather_data()
            df.to_csv(data_path, index=False)
    else:
        print("Creating new weather dataset...")
        df = create_comprehensive_weather_data()
        df.to_csv(data_path, index=False)
    
    # Prepare features
    feature_columns = [
        'temp', 'humidity', 'pressure', 'wind_speed',
        'temp_rolling_3', 'humidity_rolling_3',
        'month', 'day_of_year', 'season'
    ]
    
    # Add hour if available
    if 'hour' in df.columns:
        feature_columns.append('hour')
    
    print(f"Using features: {feature_columns}")
    
    X = df[feature_columns].fillna(df[feature_columns].mean())
    
    # Train Classification Model (Rain Prediction)
    print("\n" + "=" * 40)
    print("TRAINING CLASSIFICATION MODEL")
    print("=" * 40)
    
    y_class = df['rain_target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Scale features
    class_scaler = StandardScaler()
    X_train_scaled = class_scaler.fit_transform(X_train)
    X_test_scaled = class_scaler.transform(X_test)
    
    # Train classification model
    class_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    print("Training classification model...")
    class_model.fit(X_train_scaled, y_train)
    
    # Evaluate classification model
    y_pred_class = class_model.predict(X_test_scaled)
    class_accuracy = accuracy_score(y_test, y_pred_class)
    
    print(f"Classification Accuracy: {class_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_class))
    
    # Save classification model
    joblib.dump(class_model, 'models/weather_classification_model.pkl')
    joblib.dump(class_scaler, 'models/weather_classification_scaler.pkl')
    print("✓ Classification model saved")
    
    # Train Regression Model (Temperature Prediction)
    print("\n" + "=" * 40)
    print("TRAINING REGRESSION MODEL")
    print("=" * 40)
    
    y_reg = df['temp_next_day']
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    # Scale features for regression
    reg_scaler = StandardScaler()
    X_train_reg_scaled = reg_scaler.fit_transform(X_train)
    X_test_reg_scaled = reg_scaler.transform(X_test)
    
    # Train regression model
    reg_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    print("Training regression model...")
    reg_model.fit(X_train_reg_scaled, y_train_reg)
    
    # Evaluate regression model
    y_pred_reg = reg_model.predict(X_test_reg_scaled)
    reg_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    reg_r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"Regression RMSE: {reg_rmse:.4f}")
    print(f"Regression R² Score: {reg_r2:.4f}")
    
    # Save regression model
    joblib.dump(reg_model, 'models/weather_regression_model.pkl')
    joblib.dump(reg_scaler, 'models/weather_regression_scaler.pkl')
    print("✓ Regression model saved")
    
    # Feature importance
    print("\n" + "=" * 40)
    print("FEATURE IMPORTANCE")
    print("=" * 40)
    
    feature_importance_class = pd.DataFrame({
        'feature': feature_columns,
        'importance': class_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Classification Model Feature Importance:")
    print(feature_importance_class)
    
    feature_importance_reg = pd.DataFrame({
        'feature': feature_columns,
        'importance': reg_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nRegression Model Feature Importance:")
    print(feature_importance_reg)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Models created:")
    print("- models/weather_classification_model.pkl")
    print("- models/weather_classification_scaler.pkl")
    print("- models/weather_regression_model.pkl")
    print("- models/weather_regression_scaler.pkl")
    print("\nYou can now run your Flask app!")

if __name__ == "__main__":
    train_models()