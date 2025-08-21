from qdrant_client.http.models import Filter, FieldCondition, Range
import qdrant_client
from qdrant_client.models import VectorParams
from sentence_transformers import SentenceTransformer


# Initialize Qdrant client and embedding model
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "jobs"
qdrant = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("all-MiniLM-L6-v2")


def filter_jobs(candidate_request, threshold=0.75):
    work_pref = candidate_request.work_preference
    salary_req = work_pref.monthlySalaryAmount if work_pref else None
    notice_req = work_pref.noticePeriodWeeks if work_pref else None

    must_conditions = []

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

    filter_cond = Filter(must=must_conditions)

    # Embed resume text
    resume_text = candidate_request.resume or ""
    resume_vector = model.encode(resume_text).tolist()

    # Search in Qdrant with filter + vector search
    results = qdrant.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=resume_vector,
        filter=filter_cond,
        limit=10
    )

    threshold_jobs = []
    for res in results:
        similarity = res.score  # cosine similarity
        if similarity >= threshold:
            threshold_jobs.append(res)

    return threshold_jobs
