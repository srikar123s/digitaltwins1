import requests
import numpy as np
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_live_rainfall(lat, lon):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "forecast_days": 3
    }

    for attempt in range(3):   # retry 3 times

        try:
            response = requests.get(
                url,
                params=params,
                timeout=10,
                verify=False
            )

            response.raise_for_status()

            data = response.json()

            rainfall = data.get("hourly", {}).get("precipitation", [])

            rainfall = np.array(rainfall, dtype=np.float32)
            rainfall = np.nan_to_num(rainfall)

            return rainfall[:48]

        except Exception as e:
            print(f"Attempt {attempt+1} failed:", e)
            time.sleep(2)

    # fallback
    return np.zeros(48, dtype=np.float32)