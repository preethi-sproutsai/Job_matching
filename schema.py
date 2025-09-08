from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, root_validator


class Salary(BaseModel):
    min: Optional[str] = None
    max: Optional[str] = None
    duration: Optional[str] = None
    salvisibility: Optional[str] = None
    currency: Optional[str] = None

    @root_validator(pre=True)
    def set_default_currency(cls, values):
        if "currency" not in values or values["currency"] is None:
            values["currency"] = "$"
        return values


class JobType(BaseModel):
    type: str
    status: str

class NoticePeriod(BaseModel):
    data: Optional[str] = None

class LocationItem(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
class JobDoc(BaseModel):
    id: str = Field(..., alias="_id") 
    status: str  
    job_type: Optional[List[JobType]] = None
    location: Optional[List[LocationItem]] = None
    salary: Optional[Salary] = None
    name : Optional[str] = None
    notice_period: Optional[NoticePeriod] = None
    job_description: Optional[str] = None
    workplace: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime
    class Config:
        validate_by_name = True  # new key name in Pydantic v2 replaces allow_population_by_field_name

class SproutsResponse(BaseModel):
    jobs_updated_since: List[JobDoc] = Field(default_factory=list)


class SproutsRequest(BaseModel):
    last_updated_at_time: Optional[datetime] = None

class WorkPreference(BaseModel):
    userId: Optional[str] = None
    workAvailability: Optional[str] = None
    monthlySalaryAmount: Optional[float] = None
    monthlySalaryCurrency: Optional[str] = None
    noticePeriodWeeks: Optional[int] = None
    canStartImmediately: Optional[bool] = None
    termsAccepted: Optional[bool] = None
    currentlyBased: Optional[str] = None
    idealWorkSetup: Optional[str] = None
    preferredLocations: Optional[List[str]] = None
    locationsToAvoid: Optional[List[str]] = None
    preferredWorkShift: Optional[str] = None
    isCurrentlyEmployed: Optional[bool] = None
    preferredRoles: Optional[List[str]] = None
    preferredTechnologies: Optional[List[str]] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class CandidateRequest(BaseModel):
    work_preference: Optional[WorkPreference] = None
    resume: Optional[str] = None
    page: Optional[int] = 1  # Pagination, 1-based
    page_size: Optional[int] = 10
class CandidateJobResponse(BaseModel):
    job_id: str
    job_title: Optional[str] = None
    location: Optional[str] = None
    workplace: Optional[str] = None
    job_type: Optional[List[JobType]] = None

class CandidateResponse(BaseModel):
    jobs: list[CandidateJobResponse]
    page: int
    page_size: int
    total_pages: int
    total_results: int
