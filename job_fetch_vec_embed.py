from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio
from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import time

# ----------------- MongoDB Config -----------------
DATABASE_URL = "mongodb://admin:o1kN2aXSdyuWrMH@34.8.89.62:27017/?replicaSet=MongoReplicaSet"
client = AsyncIOMotorClient(DATABASE_URL)
db = client['job']
collection = db['job']

# ----------------- Fetch last n jobs -----------------
async def get_jobs(n):
    cursor = collection.find(
        {'status': 'active'},
        {'_id': 1, 'description': 1, 'createdAt': 1}
    ).sort('createdAt', -1).limit(n)
    jobs = await cursor.to_list(length=n)
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found")
    return jobs  # list of dicts containing '_id' and 'description'
async def get_jobs_without_desc():
    cursor = collection.find(
        {
            'status': 'active',
            '$or': [
                {'description': {'$exists': False}},
                {'description': None},
                {'description': ''}
            ]
        },
        {'_id': 1}
    )

    jobs = await cursor.to_list(length=None)
    return [str(job['_id']) for job in jobs]


# ----------------- Matching function -----------------
# ----------------- Matching function -----------------
def match_resume_top_jobs_rerank(jobs, candidate_resume, top_k_cosine=10, top_k_rerank=3):
    # Track both _id and description
    job_ids = [job["_id"] for job in jobs]
    job_texts = [job["description"] for job in jobs]

    # Step 1: Cosine similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_embeddings = model.encode(job_texts)
    candidate_embedding = model.encode([candidate_resume])

    similarity_scores = cosine_similarity(job_embeddings, candidate_embedding).flatten()
    top_cosine_indices = similarity_scores.argsort()[::-1][:top_k_cosine]

    # Prepare top cosine matches
    top_cosine_jobs = [job_texts[i] for i in top_cosine_indices]
    top_cosine_ids = [job_ids[i] for i in top_cosine_indices]
    top_cosine_scores = [similarity_scores[i] for i in top_cosine_indices]

    # Step 2: Rerank using Cross-Encoder
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[job, candidate_resume] for job in top_cosine_jobs]
    rerank_scores = cross_model.predict(pairs)

    # Get top_k_rerank indices
    top_rerank_indices = rerank_scores.argsort()[::-1][:top_k_rerank]

    # Prepare top matches with _id and both scores
    top_matches = []
    for idx in top_rerank_indices:
        top_matches.append({
            'job_id': str(top_cosine_ids[idx]),
            'job_description': top_cosine_jobs[idx],
            'cosine_similarity': round(float(top_cosine_scores[idx]), 3),
            'rerank_score': round(float(rerank_scores[idx]), 3)
        })

    # Return cross-attention scores AND cosine similarity for all top 10
    top10_scores = [{
        'job_id': str(top_cosine_ids[i]),
        'cosine_similarity': round(float(top_cosine_scores[i]), 3),
        'cross_score': round(float(rerank_scores[i]), 3)
    } for i in range(top_k_cosine)]

    return top_matches, top_cosine_ids, top10_scores
async def get_all_jobs_excluding_ids(exclude_ids):
    cursor = collection.find(
        {
            'status': 'active',
            '_id': {'$nin': [ObjectId(x) for x in exclude_ids]}
        },
        {'_id': 1, 'description': 1}
    )

    jobs = await cursor.to_list(length=None)

    return [
        {'_id': str(job['_id']), 'description': job.get('description', None)}
        for job in jobs
    ]


# ----------------- Run -----------------
async def main():
    exclude_ids = await get_jobs_without_desc()
    print(exclude_ids)
    jobs = await get_all_jobs_excluding_ids(exclude_ids)

    candidate_resume = "Vivek Chavan\nAurangabad, Maharashtra\n+91 7719968518 | vivek888chavan@gmail.com | linkedin.com/in/vivek-chavan | github.com/Vivek7038\n\nEducation:\n• CSMSS Chh. Shahu College Of Engineering 2020-present — B.Tech Computer Science, CGPA: 7.7\n• Deogiri College 2020 — Maharashtra State Board of Secondary and Higher Secondary Education, Percentage: 66\n\nExperience:\n• SproutsAI (July 2024 - Present) — Software Engineer (Frontend) Remote\n  – Led development of AI-powered Chrome extension using React.js, TypeScript, and Gemini API for automated task completion.\n  – Implemented Role-Based Access Control (RBAC) system for SaaS platform.\n  – Developed full-stack autosourcing feature using MERN stack and RabbitMQ.\n  – Built reusable custom components for consistent UI/UX.\n  – Contributed to analytics dashboard development using React.js, Redux Toolkit, and Rechart.js.\n  – Optimized application performance and fixed critical bugs while collaborating in Jira.\n\n• Klaimz (Feb 2024 - June 2024) — Software Developer Intern Remote\n  – Developed MVP for B2B startup using Next.js and Tailwind CSS.\n  – Implemented core frontend functionalities including API integrations, session management, dynamic data tables, React Flow integrations.\n  – Developed analytics dashboards using Ag-Grid and Highcharts.\n  – Built interactive flow diagrams with React Flow.\n  – Applied Agile methodologies with Trello and Git.\n  – Integrated forms with custom validation logic, reducing API calls by 50%.\n\n• Campus-Connect (Nov 2023 - Feb 2024) — Frontend Developer Remote\n  – Added advanced filter functionality for dynamic blog rendering.\n  – Built user-preview functionality using react-router-dom.\n  – Contributed innovative ideas to improve overall user experience.\n\nPersonal Projects:\n• Jotion (Dec 2023) — Productivity platform inspired by Notion.\n  – Tools: Next.js 13, React, Convex, Tailwind.\n  – Features: Real-time updates, authentication, file management, feature-rich text editor, expandable/collapsible sidebar, landing page.\n\nTechnical Skills & Interests:\nLanguages: JavaScript (ES6+), HTML5, CSS3, Tailwind CSS, Typescript\nDeveloper Tools: Visual Studio Code, Git, Chrome DevTools\nFrameworks/Libraries: ReactJs, NextJS, Redux Toolkit, Zustand\nCloud/Databases: Firebase\nSoft Skills: Effective communication and collaboration in a team environment"


    top_matches, top10_ids, top10_cross_scores = match_resume_top_jobs_rerank(
        jobs, candidate_resume, top_k_cosine=100, top_k_rerank=10
    )

    print("=== Top 10 Reranked Jobs ===")
    for i, match in enumerate(top_matches):
        print(f"{i+1}. Job ID: {match['job_id']}")
        print(f"   Cosine Sim: {match['cosine_similarity']}, Cross Score: {match['rerank_score']}")
        print(f"   Job Desc: {match['job_description'][:100]}...")
        print("-"*60)

    print("\n=== Top 100 Cosine Job IDs ===")
    for i, jid in enumerate(top10_ids):
        print(f"{i+1}. {jid}")

    print("\n=== Cross-Attention Scores for Top 100 ===")
    for score_info in top10_cross_scores:
        print(f"Job ID: {score_info['job_id']}, Cosine Sim: {score_info['cosine_similarity']}, Cross Score: {score_info['cross_score']}")


if __name__ == "__main__":
    asyncio.run(main())
