"""
Configuration file for AI Job Recommendation System
"""

# ==========================
# MODEL CONFIGURATION
# ==========================

MODEL_NAME = "all-MiniLM-L6-v2"

# ==========================
# DATA PATHS
# ==========================

JOBS_PATH = "models/clustered_jobs.pkl"

EMBEDDINGS_PATH = "models/job_embeddings.pkl"

# ==========================
# RECOMMENDATION SETTINGS
# ==========================

TOP_K = 20

# ==========================
# UI SETTINGS
# ==========================

PAGE_TITLE = "AI-Powered Job Recommendation System"

PAGE_ICON = "💼"

LAYOUT = "wide"