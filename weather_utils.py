import requests

def get_weather(city, api_key):

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    res = requests.get(url).json()

    temp = res["main"]["temp"]
    humidity = res["main"]["humidity"]

    return temp, humidity