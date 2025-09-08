import time
import requests
import math
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from schema import CandidateRequest, CandidateJobResponse, CandidateResponse
from t1 import filter_jobs
app = FastAPI()
@app.post("/search_jobs", response_model=CandidateResponse)
async def search_jobs(candidate_request: CandidateRequest):
    page = candidate_request.page if candidate_request.page and candidate_request.page > 0 else 1
    page_size = candidate_request.page_size if candidate_request.page_size and candidate_request.page_size > 0 else 10
    paginated_jobs, total_results = await filter_jobs(candidate_request, threshold=0.75)  
    total_pages = math.ceil(total_results / page_size) if page_size > 0 else 1

    #total_results = len(paginated_jobs)
   
    jobs_serializable = []
    for job in paginated_jobs:
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
        page_size=page_size,
        total_pages=total_pages,
        total_results=total_results
    )
