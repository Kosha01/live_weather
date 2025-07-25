from flask import Flask, render_template, request
from predict_live_weather import WeatherPredictor, get_live_weather_data
import os

app = Flask(__name__)

# Load the predictor once
predictor = WeatherPredictor(model_dir=os.path.join(os.getcwd(), 'models'))

# Your WeatherAPI key
WEATHER_API_KEY = "00d4c13e7309498498a183536252407 "

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    option = request.form.get('option')

    if option == "api":
        city = request.form.get('city')
        weather_data, error = get_live_weather_data(WEATHER_API_KEY, city)
        
        if error:
            return render_template('result.html', error=error)
        
        forecast = predictor.get_weather_forecast(
            weather_data['temp'],
            weather_data['humidity'],
            weather_data['pressure'],
            weather_data['wind_speed']
        )
        return render_template('result.html', city=city, forecast=forecast)

    elif option == "manual":
        try:
            temp = float(request.form.get('temp'))
            humidity = float(request.form.get('humidity'))
            pressure = float(request.form.get('pressure'))
            wind_speed = float(request.form.get('wind_speed'))

            forecast = predictor.get_weather_forecast(temp, humidity, pressure, wind_speed)
            return render_template('result.html', forecast=forecast)
        except ValueError:
            return render_template('result.html', error="Invalid manual input values.")

    return render_template('result.html', error="Invalid option selected.")

if __name__ == '__main__':
    app.run(debug=True)
