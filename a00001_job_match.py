from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample data
jobs = [
    "Python developer with machine learning experience",
    "Frontend developer React JavaScript CSS",
    "Data scientist with SQL and Python skills",
    "DevOps engineer AWS Docker Kubernetes"
]

candidates = [
    "Software engineer with Python and Django experience",
    "Full stack developer React Node.js JavaScript",
    "Data analyst with Python SQL and statistics background",
    "Cloud engineer with AWS and containerization skills"
]

def match_jobs_candidates(jobs, candidates):
    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert job descriptions to embeddings
    job_embeddings = model.encode(jobs)
    
    # Convert candidate descriptions to embeddings
    candidate_embeddings = model.encode(candidates)
    
    # Calculate cosine similarity between jobs and candidates
    similarity_matrix = cosine_similarity(job_embeddings, candidate_embeddings)

    print('similarity_matrix\n\n', similarity_matrix)
    
    # Find best matches
    matches = []
    for i, job in enumerate(jobs):
        best_candidate_idx = np.argmax(similarity_matrix[i])
        similarity_score = similarity_matrix[i][best_candidate_idx]
        
        matches.append({
            'job': job,
            'best_candidate': candidates[best_candidate_idx],
            'similarity_score': round(similarity_score, 3)
        })
    
    return matches

# Run the matching
results = match_jobs_candidates(jobs, candidates)

# Display results
print("Install required library: pip install sentence-transformers\n")

for i, match in enumerate(results):
    print(f"Job {i+1}: {match['job']}")
    print(f"Best Match: {match['best_candidate']}")
    print(f"Similarity: {match['similarity_score']}")
    print("-" * 50)