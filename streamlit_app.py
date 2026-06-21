import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pdfplumber

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI-Powered Job Recommendation System")
st.write("Upload your resume and get personalized job recommendations.")

# ==========================
# LOAD MODEL & DATA
# ==========================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    return pd.read_pickle("jobs.pkl")

@st.cache_data
def load_embeddings():
    with open("job_embeddings.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
jobs = load_jobs()
job_embeddings = load_embeddings()

# ==========================
# RESUME UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# ==========================
# PROCESS RESUME
# ==========================


if uploaded_file:

    resume_text = ""

    with pdfplumber.open(uploaded_file) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                resume_text += text

    # Resume Preview

    st.subheader("Resume Preview")

    st.text_area(
        "Extracted Text",
        resume_text[:1500],
        height=250
    )

    # Resume Embedding

    resume_embedding = model.encode(
        resume_text,
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        [resume_embedding],
        job_embeddings
    ).flatten()

    top_indices = np.argsort(
        similarities
    )[-20:][::-1]

    recommended_jobs = jobs.iloc[
        top_indices
    ].copy()

    recommended_jobs["match_score"] = (
        similarities[top_indices] * 100
    )

    # Metrics

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jobs Matched",
            len(recommended_jobs)
        )

    with col2:
        st.metric(
            "Best Match",
            f"{recommended_jobs['match_score'].max():.1f}%"
        )

    with col3:
        st.metric(
            "Average Match",
            f"{recommended_jobs['match_score'].mean():.1f}%"
        )

    # Career Domains

    # st.subheader("🚀 Recommended Career Domains")

    # st.success("🥇 Data Engineering & Analytics")
    # st.info("🥈 Enterprise Systems & Software Engineering")
    # st.warning("🥉 Data Science & Business Intelligence")

    # Recommendations Table

    st.subheader("🎯 Top Job Recommendations")

    st.dataframe(
        recommended_jobs[
            [
                "title",
                "company_name",
                "location",
                "match_score"
            ]
        ],
        use_container_width=True
    )

    # Download Button

    csv = recommended_jobs.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Recommendations",
        data=csv,
        file_name="recommended_jobs.csv",
        mime="text/csv",
        key="download_csv"
    )

    st.markdown("---")

    st.markdown("""
    ### How It Works

    1. Resume is uploaded
    2. Text is extracted from PDF
    3. Sentence-BERT generates embeddings
    4. Semantic similarity search is performed
    5. Relevant jobs are recommended
    """)

