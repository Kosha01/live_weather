import time
from datetime import datetime
from weather_prediction.predict_live_weather import WeatherPredictor, get_live_weather_data

# User input for API key and city
api_key = input('Enter your OpenWeatherMap API key: ').strip()
city = input('Enter city name for real-time prediction: ').strip()
interval = input('Enter interval in minutes (default 10): ').strip()
interval = int(interval) if interval.isdigit() and int(interval) > 0 else 10

predictor = WeatherPredictor()

print(f'\nStarting real-time weather prediction for {city} (every {interval} minutes)...')
print('='*60)

try:
    while True:
        now = datetime.now()
        weather_data, error = get_live_weather_data(api_key, city)
        if weather_data:
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Live weather in {city}: {weather_data['description'].title()}")
            forecast = predictor.get_weather_forecast(
                weather_data['temp'],
                weather_data['humidity'],
                weather_data['pressure'],
                weather_data['wind_speed'],
                now
            )
        else:
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Error fetching weather: {error}")
        print('-'*60)
        time.sleep(interval * 60)
except KeyboardInterrupt:
    print('\nReal-time weather prediction stopped.') 