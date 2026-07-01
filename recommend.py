
"""
Recommendation Engine
AI-Powered Job Recommendation System
"""

import pickle
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from cluster_labels import CLUSTER_LABELS

from config import (
    MODEL_NAME,
    JOBS_PATH,
    EMBEDDINGS_PATH,
    TOP_K
)


# ==========================
# LOAD MODEL
# ==========================

def load_model():
    """
    Load Sentence Transformer model.
    """

    model = SentenceTransformer(MODEL_NAME)

    return model


# ==========================
# LOAD JOB DATA
# ==========================

def load_jobs():
    """
    Load processed jobs dataframe.
    """

    jobs = pd.read_pickle(JOBS_PATH)

    return jobs


# ==========================
# LOAD JOB EMBEDDINGS
# ==========================

def load_job_embeddings():
    """
    Load pre-computed job embeddings.
    """

    with open(EMBEDDINGS_PATH, "rb") as f:

        job_embeddings = pickle.load(f)

    return job_embeddings


# ==========================
# RESUME EMBEDDING
# ==========================

def create_resume_embedding(model, resume_text):
    """
    Convert resume text into embedding.
    """

    embedding = model.encode(
        resume_text,
        convert_to_numpy=True
    )

    return embedding


# ==========================
# RECOMMEND JOBS
# ==========================

def recommend_jobs(
    resume_embedding,
    jobs,
    job_embeddings,
    top_k=TOP_K
):
    """
    Recommend top matching jobs.
    """

    similarities = cosine_similarity(
        [resume_embedding],
        job_embeddings
    ).flatten()

    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]

    recommended_jobs = jobs.iloc[
        top_indices
    ].copy()

    recommended_jobs["match_score"] = (
        similarities[top_indices] * 100
    )

    recommended_jobs = recommended_jobs.reset_index(
        drop=True
    )

    return recommended_jobs


# ==========================
# SUMMARY METRICS
# ==========================

def get_summary_metrics(recommended_jobs):
    """
    Calculate dashboard metrics.
    """

    metrics = {

        "recommendations": len(
            recommended_jobs
        ),

        "highest_match": round(
            recommended_jobs["match_score"].max(),
            2
        ),

        "average_match": round(
            recommended_jobs["match_score"].mean(),
            2
        )

    }

    return metrics

def get_career_paths(recommended_jobs):
    """
    Return Top 3 Career Domains
    from clustered recommendations.
    """

    cluster_counts = (
        recommended_jobs["cluster"]
        .value_counts()
    )

    career_paths = []

    for cluster, count in cluster_counts.items():

        career_paths.append({

            "domain": CLUSTER_LABELS.get(
                cluster,
                "Other"
            ),

            "jobs": count

        })

    return career_paths[:3]
