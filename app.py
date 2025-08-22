import time
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from schema import CandidateRequest, CandidateJobResponse, CandidateResponse
from vector_search_qdrant import filter_jobs
app = FastAPI()
PAGE_SIZE = 10
@app.post("/search_jobs", response_model=CandidateResponse)
def search_jobs(candidate_request: CandidateRequest):
    all_jobs = filter_jobs(candidate_request, threshold=0.75)  # Your existing function

    total_results = len(all_jobs)
    page = candidate_request.page if candidate_request.page and candidate_request.page > 0 else 1
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE

    jobs_page = all_jobs[start:end]

    jobs_serializable = []
    for job in jobs_page:
        jobs_serializable.append(
            CandidateJobResponse(
                job_id=str(job.id),
                job_title=job.payload.get("name"),
                location=", ".join(job.payload.get("location", [])) if isinstance(job.payload.get("location"), list) else job.payload.get("location"),
                workplace=job.payload.get("workplace"),
                job_type=job.payload.get("job_type")
            )
        )

    return CandidateResponse(
        jobs=jobs_serializable,
        page=page,
        page_size=PAGE_SIZE,
        total_results=total_results
    )