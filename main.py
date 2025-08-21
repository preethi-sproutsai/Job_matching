import time
import requests
from add_jobs_qdrant import process_sprouts_response

SPR_OUTS_API_URL = "http://sprouts-service/api/jobs"

last_updated_at_time = None  # store timestamp from previous run


def fetch_and_process():
    global last_updated_at_time
    params = {}
    if last_updated_at_time:
        # Send last_updated_at_time to Sprouts API (assumed as a query param)
        params['last_updated_at_time'] = last_updated_at_time

    response = requests.get(SPR_OUTS_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        next_req = process_sprouts_response(data)
        print("Next Request payload:", next_req)
        if next_req and "last_updated_at_time" in next_req:
            last_updated_at_time = next_req["last_updated_at_time"]  # update for next run
    else:
        print("Sprouts API error:", response.text)


if __name__ == "__main__":
    POLL_INTERVAL = 300  # 5 minutes
    while True:
        fetch_and_process()
        time.sleep(POLL_INTERVAL)
