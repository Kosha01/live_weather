Weather Prediction System (Model + API)
This project provides a Flask-based web application that predicts weather conditions using:

Machine Learning models (trained on historical weather data).

Live weather data from WeatherAPI for real-time predictions.

The user can choose between manual input or fetching live weather data for predictions.

Project Structure
graphql
Copy
Edit
weather_prediction/
│
│   app.py                      # Flask web app (main entry point)
│   fetch_live_weather.py       # Fetches weather data (API-based)
│   predict_live_weather.py     # Contains WeatherPredictor class (ML-based)
│   real_time_weather_prediction.py
│   train_model.py              # Training script for ML models
│   requirements.txt            # Python dependencies
│   README.md                   # Documentation
│   __init__.py
│
├───data
│       historical_weather.csv  # Dataset used for training
│
├───models                      # Pre-trained ML models
│       weather_classification_model.pkl
│       weather_classification_scaler.pkl
│       weather_regression_model.pkl
│       weather_regression_scaler.pkl
│
├───templates                   # Flask HTML templates
│       index.html              # Main input page
│       result.html             # Prediction results page
│
└───static (optional)           # Static CSS/JS files
        style.css
Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/<your-username>/weather_prediction.git
cd weather_prediction
2. Create a Virtual Environment
It is recommended to use a virtual environment:

Windows (PowerShell):

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
Linux/Mac:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Install all required packages using:

bash
Copy
Edit
pip install -r requirements.txt
4. Add Your WeatherAPI Key
Edit the app.py file and set your API key:

python
Copy
Edit
WEATHER_API_KEY = "YOUR_WEATHER_API_KEY"
Tip: You can get a free WeatherAPI key from https://www.weatherapi.com/.

Running the Flask App
1. Start the Flask Server
bash
Copy
Edit
python app.py
or explicitly:

bash
Copy
Edit
flask run
2. Open in Browser
Visit:

cpp
Copy
Edit
http://127.0.0.1:5000/
Features
Live Weather Data: Fetches current weather from WeatherAPI.

Model-Based Forecast: Predicts rain probability and tomorrow’s temperature trend using trained ML models.

Manual Input: Allows users to enter custom temperature, humidity, pressure, and wind speed.

Interactive UI: Built using Flask templates (index.html and result.html).

Execution Commands
To train models (if needed):

bash
Copy
Edit
python train_model.py
This will generate .pkl files inside the models/ folder.

To test live weather fetching:

bash
Copy
Edit
python fetch_live_weather.py
To run model-based predictions from CLI:

bash
Copy
Edit
python predict_live_weather.py
To start the Flask web app:

bash
Copy
Edit
python app.py
Example Usage
From Web Interface
Open http://127.0.0.1:5000/.

Select Live Weather or Manual Input.

Enter the city or weather values.

Click Predict Weather to see the forecast.

From Command Line
bash
Copy
Edit
python predict_live_weather.py
Follow the interactive prompts.

Requirements
Python 3.8+

Flask

pandas

numpy

scikit-learn

joblib

requests

(Installed automatically via requirements.txt.)

Future Improvements
Add weekly forecast.

Integrate advanced weather models.

Enhance UI with charts & graphs.

License
This project is open-source under the MIT License.