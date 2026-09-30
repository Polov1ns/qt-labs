"""Модуль інтеграції із зовнішнім API погоди.

Це єдине місце застосунку, яке знає про HTTP: адреси сервісів, параметри
запиту, коди відповіді й формат JSON. Веб-рівень (`app/main.py`) отримує
звідси готовий результат або зрозумілу помилку і нічого не знає про
`requests`.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherError(Exception):
    """Помилка отримання погоди, зрозуміла веб-рівню."""
    pass


def find_city(name: str):
    """Знайти координати міста за його назвою.

    Повертає словник із ключами `latitude`, `longitude` та `name`.
    Викидає WeatherError, якщо місто не знайдено, сталася помилка мережі або статус-код != 200.
    """
    if not name or not name.strip():
        raise WeatherError("Назва міста не може бути порожньою.")

    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": name.strip(), "count": 1},
            timeout=5.0
        )
    except requests.exceptions.Timeout:
        raise WeatherError("Перевищено час очікування відповіді від сервісу геокодування.")
    except requests.exceptions.ConnectionError:
        raise WeatherError("Помилка підключення до мережі інтернет.")
    except requests.exceptions.RequestException as e:
        raise WeatherError(f"Мережева помилка: {e}")


    if response.status_code != 200:
        if 400 <= response.status_code < 500:
            raise WeatherError(f"Помилка клієнта при запиті геокодування (статус {response.status_code}).")
        elif response.status_code >= 500:
            raise WeatherError(f"Збій на сервері геокодування (статус {response.status_code}).")
        else:
            raise WeatherError(f"Неочікуваний статус відповіді: {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        raise WeatherError("Отримано невалідний формат JSON від сервісу геокодування.")


    results = data.get("results")
    if not results or not isinstance(results, list) or len(results) == 0:
        raise WeatherError(f"Місто '{name}' не знайдено.")

    city_info = results[0]
    return {
        "name": city_info.get("name"),
        "latitude": city_info.get("latitude"),
        "longitude": city_info.get("longitude")
    }


def get_current_weather(city: str):
    """Повернути поточну погоду в місті: температуру й швидкість вітру.

    Викликає спочатку `find_city`, а потім робить запит до погодного API.
    """

    location = find_city(city)
    lat = location["latitude"]
    lon = location["longitude"]
    city_name = location["name"]


    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m"
            },
            timeout=5.0
        )
    except requests.exceptions.Timeout:
        raise WeatherError("Перевищено час очікування відповіді від погодного сервісу.")
    except requests.exceptions.ConnectionError:
        raise WeatherError("Помилка підключення до погодного сервісу.")
    except requests.exceptions.RequestException as e:
        raise WeatherError(f"Мережева помилка погодного API: {e}")

 
    if response.status_code != 200:
        if 400 <= response.status_code < 500:
            raise WeatherError(f"Помилка клієнта погодного API (статус {response.status_code}).")
        elif response.status_code >= 500:
            raise WeatherError(f"Збій на сервері погодного API (статус {response.status_code}).")
        else:
            raise WeatherError(f"Неочікуваний статус погодного API: {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        raise WeatherError("Отримано невалідний формат JSON від погодного сервісу.")


    current = data.get("current")
    if not isinstance(current, dict):
        raise WeatherError("У відповіді погодного сервісу відсутні дані поточних показників ('current').")

    temperature = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")

    if temperature is None or wind_speed is None:
        raise WeatherError("У відповіді погодного сервісу немає значень температури або вітру.")

    return {
        "city": city_name,
        "temperature": temperature,
        "wind_speed": wind_speed
    }