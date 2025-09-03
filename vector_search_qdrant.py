from qdrant_client.http.models import Filter, FieldCondition, Range
import qdrant_client
from qdrant_client.models import VectorParams
from sentence_transformers import SentenceTransformer
from add_jobs_qdrant import get_possible_locations_from_api, fetch_bounding_boxes
from qdrant_client.http.models import NestedCondition

from schema import CandidateRequest
import inspect
from qdrant_client.http.models import MatchExcept, FieldCondition, GeoBoundingBox, Nested, GeoPoint
print(inspect.getsource(MatchExcept))
# Initialize Qdrant client and embedding model
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "jobs"
qdrant = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("all-MiniLM-L6-v2")

def debug_nested_point_bbox_matching(job, must_not_conditions):
    geo_points_list = job.payload.get("geo_points", [])
    print(f"\nJob {job.id} has {len(geo_points_list)} geo_points entries")

    for i, cond in enumerate(must_not_conditions):
        nested_obj = getattr(cond, 'nested', None)
        if not nested_obj:
            print(f"MustNot Filter {i+1}: missing nested object")
            continue
        filter_obj = getattr(nested_obj, 'filter', None)
        if not filter_obj or not filter_obj.should:
            print(f"MustNot Filter {i+1}: missing or empty 'should' conditions")
            continue

        print(f"\nMustNot Filter {i+1} - Detailed Point vs BBox Matching:")

        bboxes = []
        # Extract all bbox coordinates
        for j, bbox_filter in enumerate(filter_obj.should):
            if bbox_filter.must and len(bbox_filter.must) > 0:
                fc = bbox_filter.must[0]
                bbox = getattr(fc, "geo_bounding_box", None)
                if bbox:
                    n = bbox.top_left.lat
                    w = bbox.top_left.lon
                    s = bbox.bottom_right.lat
                    e = bbox.bottom_right.lon
                    bboxes.append((j + 1, s, n, w, e))
                    print(f"  BBox {j + 1}: S={s}, N={n}, W={w}, E={e}")
                else:
                    print(f"  BBox {j + 1}: no geo_bounding_box found")
            else:
                print(f"  BBox {j + 1}: no must conditions found")

        # Now check every point against every bbox
        for idx, pt in enumerate(geo_points_list):
            geo_point = pt.get("point")
            if not geo_point:
                print(f"  Point {idx + 1}: missing 'point'")
                continue
            lat = float(geo_point["lat"])
            lon = float(geo_point["lon"])
            print(f"\n  Point {idx + 1}: lat={lat}, lon={lon}")

            match_any_bbox = False
            for bbox_idx, s, n, w, e in bboxes:
                inside = (s <= lat <= n) and (w <= lon <= e)
                print(f"    Inside BBox {bbox_idx}? {inside}")
                if inside:
                    match_any_bbox = True

            print(f"    → Matches any bbox? {match_any_bbox}")

def make_one_point_cond(field_name, bboxes):
    return Filter(
        should=[
            FieldCondition(
                key=field_name,
                geo_bounding_box=GeoBoundingBox(
                    top_left=GeoPoint(lat=n, lon=w),
                    bottom_right=GeoPoint(lat=s, lon=e)
                )
            )
            for (s, n, w, e) in bboxes
        ]
    )


def filter_jobs(candidate_request, threshold=0.75):
    work_pref = candidate_request.work_preference
    salary_req = work_pref.monthlySalaryAmount if work_pref else None
    notice_req = work_pref.noticePeriodWeeks if work_pref else None
    locations_to_avoid = work_pref.locationsToAvoid if work_pref else None
    all_locations_to_avoid = get_possible_locations_from_api(locations_to_avoid)
    must_conditions = []
    should_conditions = []
    must_not_conditions = []
    # status == active
    must_conditions.append(
        FieldCondition(key="status", match={"value": "active"})
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
        locations_to_avoid_bboxes = fetch_bounding_boxes(all_locations_to_avoid)
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

    filter_cond = Filter(
            must=must_conditions + [outside_nested]  # other conditions AND at least one outside
        )

       
    print(filter_cond)

    # Embed resume text
    resume_text = candidate_request.resume or ""
    resume_vector = model.encode(resume_text).tolist()

    # Search in Qdrant with filter + vector search
    results = qdrant.search(
    collection_name=QDRANT_COLLECTION_NAME,
    query_vector=resume_vector,    
    query_filter=filter_cond,
    limit=10
    )


    # Debug lat_longitudes vs must_not filters
   # print("\n--- DEBUG: Job points vs avoided bounding boxes ---")
    for job in results:
        #debug_nested_point_bbox_matching(job, must_not_conditions)
        print(f"Job ID: {job.id} | Similarity Score: {job.score}\n")

    # Filter by similarity threshold
    threshold_jobs = [res for res in results if res.score >= threshold]

    return threshold_jobs
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
matched_jobs = filter_jobs(candidate_request1, threshold=0)
for job in matched_jobs:
    print(f"Job ID: {job.id} | Similarity Score: {job.score}")
