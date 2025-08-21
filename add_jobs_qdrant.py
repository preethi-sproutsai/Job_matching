import qdrant_client
from qdrant_client.models import PointStruct, VectorParams
from datetime import datetime
from schema import SproutsResponse, SproutsRequest  # Your pydantic schemas
from sentence_transformers import SentenceTransformer
import requests

import uuid
from bson import ObjectId  
# Initialize Qdrant client and embedding model
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "jobs"
#qdrant = qdrant_client.QdrantClient(":memory:")  # For demo; replace with actual Qdrant host/url
qdrant = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create collection if not exists (no recreation)
if QDRANT_COLLECTION_NAME not in [col.name for col in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance="Cosine")
    )


def objectid_to_uuid(oid_str: str) -> str:
    # Convert ObjectId hex string (24 hex chars) to UUID by padding or filling bytes
    oid_bytes = bytes.fromhex(oid_str)
    # Pad or trim to 16 bytes for UUID (ObjectId is 12 bytes, pad with 4 zeros)
    padded_bytes = oid_bytes + b'\x00\x00\x00\x00'
    return str(uuid.UUID(bytes=padded_bytes))
import requests

EXCHANGE_API_KEY = "fca_live_Mz2e64CuQ8CpmQueVp0SC12rPTAMOgTHhSeEurCv"

def fetch_rate(from_currency: str, to_currency: str = "USD") -> float:
    if from_currency == to_currency:
        return 1.0
    else:
        url = f"https://api.freecurrencyapi.com/v1/latest?apikey={EXCHANGE_API_KEY}&base_currency={from_currency}&currencies={to_currency}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            # The API returns data in the 'data' field
            rates = data.get("data", {})
            rate = rates.get(to_currency)
            if rate:
                return float(rate)
            else:
                print("Error: to_currency not found in response data", data)
        except Exception as e:
            print("Error fetching rate:", e)
        return 1.0  # fallback

def parse_notice_period_fixed(notice_data: str) -> dict:
    """
    Convert fixed options of notice period to min/max weeks as floats:

    Options:
    1) "Immediate" -> 0 to 0 weeks
    2) "3 to 7 days" -> 3/7 to 7/7 weeks (approx 0.43 to 1.0)
    3) "1 to 2 weeks" -> 1 to 2 weeks
    4) "2 to 4 weeks" -> 2 to 4 weeks
    5) "More than 4 weeks" -> 4 to None (open ended max)
    """
    mapping = {
        "immediate": {"min_weeks": 0.0, "max_weeks": 0.0},
        "3 to 7 days": {"min_weeks": 3 / 7, "max_weeks": 1.0},
        "1 to 2 weeks": {"min_weeks": 1.0, "max_weeks": 2.0},
        "2 to 4 weeks": {"min_weeks": 2.0, "max_weeks": 4.0},
        "more than 4 weeks": {"min_weeks": 4.0, "max_weeks": None},
    }

    nd_lower = notice_data.lower().strip()
    # Return mapped value or empty dict if unknown
    return mapping.get(nd_lower, {})



# Usage
rate_inr_usd = fetch_rate("INR", "USD")
print("INR to USD:", rate_inr_usd)

def convert_salary_to_per_month(salary: dict, job_type_list: list) -> dict:
    """
    Convert salary to per month salary in same structure:
    - convert min, max according to duration (day/week/year/hour)
    - handle currency conversion if needed (₹ to $)
    - For hourly, convert only if full-time or part-time
    """
    if not salary:
        return salary

    # Clone original to keep structure
    converted_salary = salary.copy()

    try:
        min_sal = float(salary.get("min", 0) or 0)
        max_sal = float(salary.get("max", 0) or 0)
        duration = salary.get("duration", "").lower()
        currency = salary.get("currency", "$") or "$"
    except Exception:
        return salary  # fallback to original if parsing fails

    # conversion rates
    rate = fetch_rate("INR", "USD") if currency =="₹" else 1

    # Check job types for hourly conversion
    job_types = [jt["type"].lower() for jt in job_type_list if isinstance(jt, dict) and "type" in jt]
    is_full_time = "full-time" in job_types
    is_part_time = "part-time" in job_types

    multiplier = 1.0
    if "per day" in duration:
        multiplier = 30
    elif "per week" in duration:
        multiplier = 4
    elif "per year" in duration:
        multiplier = 1 / 12
    elif "per hour" in duration:
        if is_full_time:
            multiplier = 40 * 4  # 160 hours/month
        elif is_part_time:
            multiplier = 20 * 4  # 80 hours/month
        else:
            multiplier = 40 * 4 # same as full-time

    # Convert min/max applying multiplier and exchange rate
    converted_salary["min"] = f"{min_sal * multiplier * rate:.2f}"
    converted_salary["max"] = f"{max_sal * multiplier * rate:.2f}"
    converted_salary["duration"] = "Per month"
    converted_salary["currency"] = "$"  # After conversion to USD

    return converted_salary
def process_sprouts_response(data: dict):
    sprouts_response = SproutsResponse(**data)

    latest_jobs = sprouts_response.jobs_updated_since
    if not latest_jobs:
        return None

    for job in latest_jobs:
        job_id = objectid_to_uuid(str(job.id))

        # Convert salary
        monthly_salary = None
        if job.salary:
            monthly_salary = convert_salary_to_per_month(
                job.salary.model_dump() if hasattr(job.salary, "dict") else job.salary,
                job.job_type or []
            )

        # Prepare payload
        payload = job.model_dump(by_alias=True)

        # Overwrite salary with converted monthly salary
        if monthly_salary:
            payload["salary"] = monthly_salary

        # Transform job_type: only include types where status == "true"
        if job.job_type:
            active_job_types = [jt.type for jt in job.job_type if jt.status.lower() == "true"]
            payload["job_type"] = active_job_types

        # Transform location: assuming location is a list of dicts with "name" and "status"
        # If your schema differs, adjust accordingly.
        if isinstance(job.location, list):
            active_locations = [loc.name for loc in job.location if (loc.status or "").lower() == "true"]
            payload["location"] = active_locations

        # Parse notice_period and add processed field
        if job.notice_period and job.notice_period.data:
            notice_weeks = parse_notice_period_fixed(job.notice_period.data)
            payload["notice_period"] = notice_weeks

        existing_points = qdrant.retrieve(
            collection_name=QDRANT_COLLECTION_NAME,
            ids=[job_id]
        )

        if existing_points and existing_points[0].vector:
            old_description = existing_points.payload.get("job_description")
            if old_description != job.job_description:
                vector = model.encode(job.job_description or "").tolist()
            else:
                vector = existing_points.vector
        else:
            vector = model.encode(job.job_description or "").tolist()

        qdrant.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=job_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    last_updated_at = max(job.updatedAt for job in latest_jobs)
    next_request = SproutsRequest(last_updated_at_time=last_updated_at)
    return next_request.model_dump()



if __name__ == "__main__":
    sprouts_data = {
        "jobs_updated_since": [
            {
                "_id": "68a6f05fc15d170007b87257",
                "status": "active",
                 "job_type": [
                    {
                    "type": "full-time",
                    "status": "false"
                    },
                    {
                    "type": "regular/permanent",
                    "status": "false"
                    },
                    {
                    "type": "part-time",
                    "status": "true"
                    },
                    {
                    "type": "internship",
                    "status": "false"
                    },
                    {
                    "type": "contract/temporary",
                    "status": "false"
                    },
                    {
                    "type": "volunteer",
                    "status": "false"
                    },
                    {
                    "type": "other",
                    "status": "false"
                    }
                ],
                "location": [
                    {
                    "name": "Austin, Texas, USA",
                    "status": "true"
                    },
                    {
                    "name": "Sunnyvale, California, USA",
                    "status": "true"
                    }
                ],
                "salary": {
                    "min": "100",
                    "max": "150",
                    "duration": "Per day",
                    "salvisibility": "Display",
                    "currency": "₹"
                },
                "name": "Computer Vision Engineer",
                "notice_period": {
                    "data": "2 to 4 weeks"
                },
                "job_description": "Work on NLP models.",
                "workplace": "Remote",
                "createdAt": "2025-08-20T10:00:00Z",
                "updatedAt": "2025-08-21T15:00:00Z"
            }
        ]
    }

    next_req = process_sprouts_response(sprouts_data)
    print("Next Request:", next_req)
