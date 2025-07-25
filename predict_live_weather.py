import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import requests
import json

class WeatherPredictor:
    def __init__(self, model_dir='models'):
        """Initialize the weather predictor with trained models"""
        self.model_dir = model_dir
        
        # Load classification model and scaler
        self.class_model_path = os.path.join(model_dir, 'weather_classification_model.pkl')
        self.class_scaler_path = os.path.join(model_dir, 'weather_classification_scaler.pkl')
        
        # Load regression model and scaler
        self.reg_model_path = os.path.join(model_dir, 'weather_regression_model.pkl')
        self.reg_scaler_path = os.path.join(model_dir, 'weather_regression_scaler.pkl')
        
        self.load_models()
    
    def load_models(self):
        """Load trained models and scalers"""
        try:
            print("Loading trained models...")
            
            # Load classification model
            if os.path.exists(self.class_model_path) and os.path.exists(self.class_scaler_path):
                self.class_model = joblib.load(self.class_model_path)
                self.class_scaler = joblib.load(self.class_scaler_path)
                print("✓ Classification model loaded successfully")
            else:
                print("✗ Classification model not found")
                self.class_model = None
                self.class_scaler = None
            
            # Load regression model
            if os.path.exists(self.reg_model_path) and os.path.exists(self.reg_scaler_path):
                self.reg_model = joblib.load(self.reg_model_path)
                self.reg_scaler = joblib.load(self.reg_scaler_path)
                print("✓ Regression model loaded successfully")
            else:
                print("✗ Regression model not found")
                self.reg_model = None
                self.reg_scaler = None
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def prepare_features(self, temp, humidity, pressure, wind_speed, date=None):
        """Prepare features for prediction"""
        if date is None:
            date = datetime.now()
        
        # Create rolling averages (simulate with current values for single prediction)
        temp_rolling_3 = temp
        humidity_rolling_3 = humidity
        
        # Extract date features
        month = date.month
        day_of_year = date.timetuple().tm_yday
        season = ((date.month % 12) // 3) + 1
        
        # Create feature vector
        features = np.array([[
            temp, humidity, pressure, wind_speed,
            temp_rolling_3, humidity_rolling_3,
            month, day_of_year, season
        ]])
        
        feature_names = [
            'temp', 'humidity', 'pressure', 'wind_speed',
            'temp_rolling_3', 'humidity_rolling_3',
            'month', 'day_of_year', 'season'
        ]
        
        return features, feature_names
    
    def predict_rain(self, temp, humidity, pressure, wind_speed, date=None):
        """Predict probability of rain"""
        if self.class_model is None:
            return None, "Classification model not available"
        
        try:
            features, feature_names = self.prepare_features(temp, humidity, pressure, wind_speed, date)
            features_scaled = self.class_scaler.transform(features)
            
            # Get prediction and probability
            prediction = self.class_model.predict(features_scaled)[0]
            probability = self.class_model.predict_proba(features_scaled)[0]
            
            rain_probability = probability[1] * 100  # Probability of rain
            
            result = {
                'will_rain': bool(prediction),
                'rain_probability': round(rain_probability, 2),
                'confidence': 'High' if max(probability) > 0.8 else 'Medium' if max(probability) > 0.6 else 'Low'
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Error in rain prediction: {e}"
    
    def predict_temperature(self, temp, humidity, pressure, wind_speed, date=None):
        """Predict next day temperature"""
        if self.reg_model is None:
            return None, "Regression model not available"
        
        try:
            features, feature_names = self.prepare_features(temp, humidity, pressure, wind_speed, date)
            features_scaled = self.reg_scaler.transform(features)
            
            # Get temperature prediction
            temp_prediction = self.reg_model.predict(features_scaled)[0]
            
            result = {
                'predicted_temp': round(temp_prediction, 1),
                'temp_change': round(temp_prediction - temp, 1),
                'trend': 'Rising' if temp_prediction > temp else 'Falling' if temp_prediction < temp else 'Stable'
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Error in temperature prediction: {e}"
    
    def get_weather_forecast(self, temp, humidity, pressure, wind_speed, date=None):
        """Get complete weather forecast"""
        if date is None:
            date = datetime.now()
        
        print(f"\n🌤️  WEATHER FORECAST for {date.strftime('%Y-%m-%d')}")
        print("=" * 50)
        
        # Current conditions
        print(f"📊 Current Conditions:")
        print(f"   Temperature: {temp}°C")
        print(f"   Humidity: {humidity}%")
        print(f"   Pressure: {pressure} hPa")
        print(f"   Wind Speed: {wind_speed} km/h")
        print()
        
        # Rain prediction
        rain_result, rain_error = self.predict_rain(temp, humidity, pressure, wind_speed, date)
        if rain_result:
            rain_status = "🌧️  YES" if rain_result['will_rain'] else "☀️  NO"
            print(f"🌧️  Rain Prediction: {rain_status}")
            print(f"   Rain Probability: {rain_result['rain_probability']}%")
            print(f"   Confidence: {rain_result['confidence']}")
        else:
            print(f"❌ Rain Prediction Error: {rain_error}")
        print()
        
        # Temperature prediction
        temp_result, temp_error = self.predict_temperature(temp, humidity, pressure, wind_speed, date)
        if temp_result:
            trend_emoji = "📈" if temp_result['trend'] == 'Rising' else "📉" if temp_result['trend'] == 'Falling' else "➡️"
            print(f"🌡️  Tomorrow's Temperature: {temp_result['predicted_temp']}°C")
            print(f"   Change: {temp_result['temp_change']:+.1f}°C ({temp_result['trend']}) {trend_emoji}")
        else:
            print(f"❌ Temperature Prediction Error: {temp_error}")
        
        print("=" * 50)
        
        return {
            'rain_forecast': rain_result,
            'temperature_forecast': temp_result,
            'current_conditions': {
                'temp': temp, 'humidity': humidity, 
                'pressure': pressure, 'wind_speed': wind_speed
            }
        }

# ==========================
# WEATHERAPI INTEGRATION
# ==========================
def get_live_weather_data(api_key, city):
    """Get live weather data from WeatherAPI"""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            weather_data = {
                'temp': data['current']['temp_c'],
                'humidity': data['current']['humidity'],
                'pressure': data['current']['pressure_mb'],
                'wind_speed': data['current']['wind_kph'],
                'city': data['location']['name'],
                'description': data['current']['condition']['text']
            }
            return weather_data, None
        else:
            return None, f"API Error: {data.get('error', {}).get('message', 'Unknown error')}"
            
    except Exception as e:
        return None, f"Error fetching weather data: {e}"

# ==========================
# INTERACTIVE INTERFACE
# ==========================
def interactive_weather_prediction():
    """Interactive weather prediction interface"""
    print("🌤️  WEATHER PREDICTION SYSTEM (WeatherAPI)")
    print("=" * 50)
    
    # Initialize predictor
    predictor = WeatherPredictor()
    
    while True:
        print("\nChoose an option:")
        print("1. Manual weather input")
        print("2. Use live weather data (WeatherAPI)")
        print("3. Test with sample data")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Manual input
            try:
                print("\n📝 Enter current weather conditions:")
                temp = float(input("Temperature (°C): "))
                humidity = float(input("Humidity (%): "))
                pressure = float(input("Pressure (hPa): "))
                wind_speed = float(input("Wind Speed (km/h): "))
                
                predictor.get_weather_forecast(temp, humidity, pressure, wind_speed)
                
            except ValueError:
                print("❌ Please enter valid numbers")
        
        elif choice == '2':
            # Live weather data
            api_key = input("Enter your WeatherAPI key: ").strip()
            city = input("Enter city name: ").strip()
            
            if api_key and city:
                weather_data, error = get_live_weather_data(api_key, city)
                if weather_data:
                    print(f"\n🌍 Live weather data for {weather_data['city']}:")
                    print(f"Description: {weather_data['description']}")
                    
                    predictor.get_weather_forecast(
                        weather_data['temp'],
                        weather_data['humidity'],
                        weather_data['pressure'],
                        weather_data['wind_speed']
                    )
                else:
                    print(f"❌ {error}")
            else:
                print("❌ API key and city name are required")
        
        elif choice == '3':
            # Test with sample data
            sample_data = [
                {'temp': 25, 'humidity': 80, 'pressure': 1010, 'wind_speed': 15, 'desc': 'Warm and humid'},
                {'temp': 15, 'humidity': 60, 'pressure': 1020, 'wind_speed': 10, 'desc': 'Cool and dry'},
                {'temp': 30, 'humidity': 90, 'pressure': 1000, 'wind_speed': 20, 'desc': 'Hot and very humid'},
                {'temp': 10, 'humidity': 70, 'pressure': 1015, 'wind_speed': 5, 'desc': 'Cold and moderate humidity'}
            ]
            
            print("\n🧪 Testing with sample weather conditions:")
            for i, data in enumerate(sample_data, 1):
                print(f"\n--- Sample {i}: {data['desc']} ---")
                predictor.get_weather_forecast(
                    data['temp'], data['humidity'], 
                    data['pressure'], data['wind_speed']
                )
                input("Press Enter to continue...")
        
        elif choice == '4':
            print("👋 Thank you for using Weather Prediction System!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    interactive_weather_prediction()
