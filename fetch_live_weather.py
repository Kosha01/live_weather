import requests
import json

# Replace with your WeatherAPI key
API_KEY = "00d4c13e7309498498a183536252407"
BASE_URL = "http://api.weatherapi.com/v1/current.json"

def fetch_live_weather(city):
    """
    Fetch live weather data using WeatherAPI
    Returns weather data dictionary or None if failed
    """
    params = {
        'key': API_KEY,
        'q': city,
        'aqi': 'no'  # Exclude air quality for simplicity
    }
    
    try:
        print(f'Fetching live weather data for {city}...')
        response = requests.get(BASE_URL, params=params, timeout=10)
        print(f'API Response Status: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()

            weather_features = {
                'temp': data['current']['temp_c'],
                'humidity': data['current']['humidity'],
                'pressure': data['current']['pressure_mb'],
                'wind_speed': data['current']['wind_kph'],
                'city': data['location']['name'],
                'country': data['location']['country'],
                'description': data['current']['condition']['text'],
                'icon': data['current']['condition']['icon'],
                'condition': data['current']['condition']['text']
            }
            
            print(f'✓ Successfully fetched weather for {city}')
            print(f'  Temperature: {weather_features["temp"]}°C')
            print(f'  Condition: {weather_features["description"]}')
            
            return weather_features

        elif response.status_code == 400:
            print(f'❌ Bad request or City not found: {city}')
            return None

        elif response.status_code == 401:
            print('❌ API Key Error: Invalid or expired API key')
            return None

        else:
            print(f'❌ API Error {response.status_code}: {response.text}')
            return None

    except requests.exceptions.Timeout:
        print('❌ Request timeout - API took too long to respond')
        return None
    except requests.exceptions.ConnectionError:
        print('❌ Connection error - Check your internet connection')
        return None
    except requests.exceptions.RequestException as e:
        print(f'❌ Request error: {e}')
        return None
    except json.JSONDecodeError:
        print('❌ Invalid JSON response from API')
        return None
    except KeyError as e:
        print(f'❌ Missing data in API response: {e}')
        return None
    except Exception as e:
        print(f'❌ Unexpected error: {e}')
        return None


def test_api_connection():
    """Test the API connection with a sample city"""
    print("Testing WeatherAPI connection...")
    test_cities = ['London', 'Chennai', 'New York', 'Tokyo']
    
    for city in test_cities:
        print(f"\n--- Testing {city} ---")
        weather_data = fetch_live_weather(city)
        
        if weather_data:
            print(f"✅ Success for {city}")
            break
        else:
            print(f"❌ Failed for {city}")
    else:
        print("\n❌ All test cities failed. Please check:")
        print("1. Your API key is correct")
        print("2. Your internet connection")
        print("3. WeatherAPI service status")

if __name__ == "__main__":
    test_api_connection()
