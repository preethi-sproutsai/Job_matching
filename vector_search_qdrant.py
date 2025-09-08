from qdrant_client.http.models import Filter, FieldCondition, Range
import qdrant_client
from qdrant_client.models import VectorParams
from sentence_transformers import SentenceTransformer
from add_jobs_qdrant import fetch_bounding_boxes
from qdrant_client.http.models import NestedCondition
from shapely.geometry import Point, box
from schema import CandidateRequest
import inspect
from qdrant_client.http.models import MatchExcept, FieldCondition, GeoBoundingBox, Nested, GeoPoint
print(inspect.getsource(MatchExcept))
# Initialize Qdrant client and embedding model
QDRANT_HOST = "localhost"
QDRANT_PORT = 6336
QDRANT_COLLECTION_NAME = "jobs"
qdrant = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("all-MiniLM-L6-v2")

async def filter_jobs(candidate_request, threshold=0.75):
    work_pref = candidate_request.work_preference
    salary_req = work_pref.monthlySalaryAmount if work_pref else None
    notice_req = work_pref.noticePeriodWeeks if work_pref else None
    locations_to_avoid = work_pref.locationsToAvoid if work_pref else None
    preferred_locations = work_pref.preferredLocations if work_pref else None
    page = candidate_request.page if candidate_request.page and candidate_request.page > 0 else 1
    page_size = candidate_request.page_size if candidate_request.page_size and candidate_request.page_size > 0 else 10
    #all_locations_to_avoid = get_possible_locations_from_api(locations_to_avoid)
    must_conditions = []
    should_conditions = []
    must_not_conditions = []
    # status == active
    must_conditions.append(
        FieldCondition(key="status", match={"value": "active"})
    )
    # Job-type preference mapping (Sprouts → Metantz)
    job_type_pref = work_pref.workAvailability if work_pref else None
    if job_type_pref:
        if job_type_pref.lower() == "full-time":
            must_conditions.append(
                FieldCondition(
                    key="job_type",
                    match={"any": ["Full-time", "regular/permanent", "internship", "contract/temporary"]}
                )
            )
        elif job_type_pref.lower() == "part-time":
            must_conditions.append(
                FieldCondition(
                    key="job_type",
                    match={"value": "Part-time"}
                )
            )
        elif job_type_pref.lower() == "flexible":
            must_conditions.append(
                FieldCondition(
                    key="job_type",
                    match={"any": ["Volunteer", "other"]}
                )
            )
    # Workplace preference mapping (Sprouts → Metantz)
    workplace_pref = work_pref.idealWorkSetup if work_pref else None
    if workplace_pref:
        if workplace_pref.lower() == "in-office":
            must_conditions.append(
                FieldCondition(
                    key="workplace",
                    match={"value": "On-site"}
                )
            )
        elif workplace_pref.lower() == "remote":
            must_conditions.append(
                FieldCondition(
                    key="workplace",
                    match={"value": "Remote"}
                )
            )
        elif workplace_pref.lower() == "hybrid":
            must_conditions.append(
                FieldCondition(
                    key="workplace",
                    match={"value": "Hybrid"}
                )
            )

    # Salary filtering
    if salary_req:
        must_conditions.append(
            FieldCondition(
                key="salary.min",
                range=Range(lte=salary_req)  # candidate salary >= job.min
            )
        )

    # Notice period filtering
    if notice_req:
        must_conditions.append(
            FieldCondition(
                key="notice_period_weeks.max_weeks",
                range=Range(gte=notice_req)  # job.max_weeks >= candidate notice
            )
        )

    # locations_filtering
    if locations_to_avoid:
        locations_to_avoid_bboxes = await fetch_bounding_boxes(locations_to_avoid)
        one_point_cond = [
            FieldCondition(
                key="point",
                geo_bounding_box=GeoBoundingBox(
                    top_left=GeoPoint(lat=n, lon=w),
                    bottom_right=GeoPoint(lat=s, lon=e)
                )
            )
            for (s, n, w, e) in locations_to_avoid_bboxes
        ]
       

        outside_nested = NestedCondition(
            nested={
                "key": "geo_points",
                "filter": Filter(
                    must_not=one_point_cond   # at least one point must NOT be in any forbidden bbox
                )
            }
        )

    #filter_cond = Filter(must=must_conditions + [outside_nested])  # other conditions AND at least one outside
    if locations_to_avoid:
        filter_cond = Filter(must=must_conditions + [outside_nested])
    else:
        filter_cond = Filter(must=must_conditions)

    # Embed resume text
    resume_text = candidate_request.resume or ""
    resume_vector = model.encode(resume_text).tolist()

    # Search in Qdrant with filter + vector search
    results = qdrant.search(collection_name=QDRANT_COLLECTION_NAME, query_vector=resume_vector, query_filter=filter_cond, limit=100)
   # Prioritize jobs in preferred locations
    preferred_jobs = []
    other_jobs = []

    # Filter by similarity threshold
    threshold_jobs = [res for res in results if res.score >= threshold]
    # Prioritize preferred locations- use geo-locations 
    if preferred_locations:
        # Fetch bounding boxes for all preferred locations
        preferred_bboxes = fetch_bounding_boxes(preferred_locations)
        preferred_polygons = [
            box(w, s, e, n)   # shapely box takes (minx, miny, maxx, maxy) = (west, south, east, north)
            for (s, n, w, e) in preferred_bboxes
        ]

        for job in threshold_jobs:
            geo_points_list = job.payload.get("geo_points", [])
            inside_any = False

            for pt in geo_points_list:
                geo_point = pt.get("point")
                if not geo_point:
                    continue
                lat = float(geo_point["lat"])
                lon = float(geo_point["lon"])
                p = Point(lon, lat)  # shapely Point takes (x=lon, y=lat)

                # Check if inside any bbox polygon
                if any(p.within(poly) for poly in preferred_polygons):
                    inside_any = True
                    break

            if inside_any:
                preferred_jobs.append(job)
            else:
                other_jobs.append(job)
    else:
        other_jobs = threshold_jobs

    # Merge results: preferred first
    final_jobs = preferred_jobs + other_jobs

    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_jobs = final_jobs[start_idx:end_idx]

    return paginated_jobs, len(final_jobs)

candidate_data1 = {
    "resume": "Experienced python developer with expertise in backend systems and cloud infrastructure.",
    "work_preference": {
        "monthlySalaryAmount": 0,
        "noticePeriodWeeks": 0,
        "locationsToAvoid": ["Hyderabad", "Austin", "Pakistan"]
    }
}
candidate_data2 = {
    "resume": "Experienced python developer with expertise in backend systems and cloud infrastructure.",
    "work_preference": {
        "monthlySalaryAmount": 0,
        "noticePeriodWeeks": 0,
        "locationsToAvoid": ["Hyderabad"]
    }
}
candidate_data3 = {
    "resume": "Experienced python developer with expertise in backend systems and cloud infrastructure.",
    "work_preference": {
        "monthlySalaryAmount": 0,
        "noticePeriodWeeks": 0,
        "locationsToAvoid": ["Mumbai"]
    }
}
candidate_data4 = {
    "resume": "Experienced python developer with expertise in backend systems and cloud infrastructure.",
    "work_preference": {
        "monthlySalaryAmount": 0,
        "noticePeriodWeeks": 0,
        "locationsToAvoid": ["India"]
    }
}
candidate_data5 = {
    "resume": "Experienced python developer with expertise in backend systems and cloud infrastructure.",
    "work_preference": {
        "monthlySalaryAmount": 0,
        "noticePeriodWeeks": 0,
        "locationsToAvoid": ["new york"]
    }
}
candidate_request1 = CandidateRequest(**candidate_data5)
import time
start_time = time.time()
import asyncio
matched_jobs, total = asyncio.run(filter_jobs(candidate_request1, threshold=0))
end_time = time.time()

#matched_jobs = filter_jobs(candidate_request1, threshold=0)
for job in matched_jobs:
   print(f"Job ID: {job.id} | Similarity Score: {job.score}")



print(f"\nTime taken to get results: {end_time - start_time:.4f} seconds")
