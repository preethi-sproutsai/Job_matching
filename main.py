import time
import requests
from add_jobs_qdrant import process_sprouts_response
import json
from pathlib import Path
# main.py
from config import EXCHANGE_RATE_FILE, EXCHANGE_API_KEY, LAST_UPDATE_FILE, SPR_OUTS_API_URL
from add_jobs_qdrant import process_sprouts_response

def load_last_updated_at():
    if LAST_UPDATE_FILE.exists():
        return json.loads(LAST_UPDATE_FILE.read_text()).get("last_updated_at_time")
    return None

def save_last_updated_at(timestamp):
    LAST_UPDATE_FILE.write_text(json.dumps({"last_updated_at_time": timestamp}))

def fetch_rate(from_currency: str = "INR", to_currency: str = "USD", force_refresh: bool = False) -> float:
    """
    Fetch INR→USD exchange rate. Stores locally and reuses until next refresh.
    """
    # If cached rate exists and not forcing refresh, use it
    if EXCHANGE_RATE_FILE.exists() and not force_refresh:
        data = json.loads(EXCHANGE_RATE_FILE.read_text())
        rate = data.get("rate")
        timestamp = data.get("timestamp", 0)
        # Optionally refresh if older than 1 day
        if rate and (time.time() - timestamp) < 24*3600:
            return rate

    # Otherwise, fetch from API
    url = f"https://api.freecurrencyapi.com/v1/latest?apikey={EXCHANGE_API_KEY}&base_currency={from_currency}&currencies={to_currency}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        rate = data.get("data", {}).get(to_currency)
        if rate:
            rate = float(rate)
            # save locally
            EXCHANGE_RATE_FILE.write_text(json.dumps({"rate": rate, "timestamp": time.time()}))
            return rate
        else:
            print("Error: to_currency not found in API response", data)
    except Exception as e:
        print("Error fetching rate:", e)

    # fallback
    if EXCHANGE_RATE_FILE.exists():
        cached = json.loads(EXCHANGE_RATE_FILE.read_text())
        return cached.get("rate", 1.0)
    return 1.0

def fetch_and_process():
    last_updated_at_time = load_last_updated_at()
    params = {}
    if last_updated_at_time:
        params['last_updated_at_time'] = last_updated_at_time

    # Get INR→USD rate
    inr_usd_rate = fetch_rate()
    print(f"Current INR→USD rate: {inr_usd_rate}")

    try:
        response = requests.get(SPR_OUTS_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        next_req = process_sprouts_response(data, inr_usd_rate=inr_usd_rate) 
        if next_req and "last_updated_at_time" in next_req:
            save_last_updated_at(next_req["last_updated_at_time"])
        print("Sprouts jobs synced successfully.")
    except Exception as e:
        print(f"Error syncing Sprouts jobs: {e}")

if __name__ == "__main__":
    POLL_INTERVAL = 86400  # 24 hours
    while True:
        fetch_and_process()
        time.sleep(POLL_INTERVAL)
