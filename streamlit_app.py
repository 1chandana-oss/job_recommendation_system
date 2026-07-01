
import streamlit as st
import plotly.express as px
import pandas as pd

from config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT
)

from utils import (
    extract_resume_text,
    get_resume_statistics,
    clean_text
)

from recommend import (
    load_model,
    load_jobs,
    load_job_embeddings,
    create_resume_embedding,
    recommend_jobs,
    get_summary_metrics,
    get_career_paths
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.title("💼 AI Job Recommender")

    st.markdown("### 🤖 Model")
    st.write("Sentence-BERT")
    st.caption("all-MiniLM-L6-v2")

    st.markdown("### 📊 Dataset")
    st.write("10,000 LinkedIn Job Postings")
    st.markdown("### 🚀 Features")

    st.write("✅ Resume Parsing")
    st.write("✅ Semantic Search")
    st.write("✅ Job Recommendation")
    st.write("✅ Download Results")

    st.markdown("---")

    st.caption("Developed by")
    st.write("**Chandana Das**")

# ==========================
# TITLE
# ==========================

st.title("💼 AI-Powered Job Recommendation System")

st.write(
    "Upload your resume and receive AI-powered personalized job recommendations."
)

# ==========================
# LOAD MODEL & DATA
# ==========================

@st.cache_resource
def load_resources():

    model = load_model()

    jobs = load_jobs()

    embeddings = load_job_embeddings()

    return model, jobs, embeddings


model, jobs, job_embeddings = load_resources()

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "📄 Upload Resume (PDF)",
    type=["pdf"]
)

# ==========================
# PROCESS RESUME
# ==========================

if uploaded_file:

    resume_text, total_pages = extract_resume_text(
        uploaded_file
    )

    resume_text = clean_text(
        resume_text
    )

    stats = get_resume_statistics(
        resume_text,
        total_pages
    )


    # ==========================
    # RECOMMENDATION
    # ==========================

    resume_embedding = create_resume_embedding(
        model,
        resume_text
    )

    recommended_jobs = recommend_jobs(
        resume_embedding,
        jobs,
        job_embeddings
    )

    metrics = get_summary_metrics(
        recommended_jobs
    )
    career_paths = get_career_paths(
       recommended_jobs
    )
    career_paths = [
       c for c in career_paths
       if c["jobs"] >= 2
    ]
    st.subheader("📄 Resume Insights")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Pages",
            stats["pages"]
        )

    with c2:
        st.metric(
            "Words",
            stats["words"]
        )

    with c3:
        st.metric(
            "Characters",
            stats["characters"]
        )
    # ==========================
    # DASHBOARD METRICS
    # ==========================

    # ==========================
# Recommendation Summary
# ==========================

# ==========================
# Career Paths
# ==========================

    st.subheader("🚀 Recommended Career Paths")

    total_jobs = metrics["recommendations"]

    for i, career in enumerate(career_paths[:3]):

       medal = ["🥇", "🥈", "🥉"][i]

       percentage = (
           career["jobs"] / total_jobs
        ) * 100

       st.markdown(f"""
       ### {medal} {career['domain']}

       **Matching Jobs:** {career['jobs']}

       **Coverage:** {percentage:.1f}%
        """)

       st.progress(percentage/100)

       st.markdown("---")


    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Recommendations",
            metrics["recommendations"]
        )

    with m2:
        st.metric(
            "Highest Match",
            f"{metrics['highest_match']:.2f}%"
        )

       
    with m3:
        st.metric(
            "Average Compatibility",
            f"{metrics['average_match']:.2f}%"
        )
        # ==========================
# SIDEBAR FILTERS
# ==========================

    st.sidebar.markdown("---")
    st.subheader("🔍 Filter Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        work_type = st.selectbox(
            "Work Type",
            ["All"] + sorted(
                recommended_jobs["formatted_work_type"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col2:
        experience = st.selectbox(
            "Experience Level",
            ["All"] + sorted(
                recommended_jobs["formatted_experience_level"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    filtered_jobs = recommended_jobs.copy()

    if work_type != "All":
        filtered_jobs = filtered_jobs[
            filtered_jobs["formatted_work_type"] == work_type
        ]

    if experience != "All":
        filtered_jobs = filtered_jobs[
            filtered_jobs["formatted_experience_level"] == experience
        ]

    search = st.text_input("🔍 Search Job Title")

    if search:
       filtered_jobs = filtered_jobs[
        filtered_jobs["title"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

    st.subheader("📊 Top Match Scores")
    chart = (
    filtered_jobs.head(10)
    .sort_values("match_score")
)

    fig = px.bar(
    chart,
    x="match_score",
    y="title",
    orientation="h",
    text="match_score",
    title="Top 10 Match Scores"
)

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    fig.update_layout(
    xaxis_title="Match Score (%)",
    yaxis_title="",
    height=500
)

    st.plotly_chart(
       fig,
       width="stretch"
)




    # ==========================
    # JOB RECOMMENDATIONS
    # ==========================
    st.subheader("🎯 Top Job Recommendations")

    for _, row in filtered_jobs.iterrows():

       with st.expander(
        f"💼 {row['title']} ({row['match_score']:.1f}%)"
    ):

        st.write(f"🏢 Company: {row['company_name']}")

        st.write(f"📍 Location: {row['location']}")

        st.write(f"💼 Work Type: {row['formatted_work_type']}")

        st.write(f"🎓 Experience: {row['formatted_experience_level']}")

        st.write(f"⭐ Match Score: {row['match_score']:.2f}%")

        st.progress(row["match_score"]/100)

        # ==========================
        # Required Skills
        # ==========================

        if (
            "skills_desc" in row
            and pd.notna(row["skills_desc"])
        ):

            st.markdown("### 🛠 Required Skills")

            st.write(row["skills_desc"])

        # ==========================
        # Job Description
        # ==========================

        if (
            "description" in row
            and pd.notna(row["description"])
        ):

            st.markdown("### 📄 Job Description")

            st.write(row["description"][:800] + "...")

        # ==========================
        # Apply Button
        # ==========================

        if (
            "job_posting_url" in row
            and pd.notna(row["job_posting_url"])
        ):

            st.link_button(
                "🚀 Apply Now",
                row["job_posting_url"]
            )

        st.markdown("---")

    # ==========================
    # DOWNLOAD
    # ==========================

    csv = filtered_jobs.to_csv(
      index=False
    )
    st.download_button(

        "📥 Download Recommendations",

        csv,

        "filtered_recommendations.csv",

        "text/csv"

    )

    # ==========================
    # FOOTER
    # ==========================


    with st.expander("⚙️ How It Works"):

     st.markdown("""
1. Upload your resume.

2. Resume text is extracted.

3. Sentence-BERT generates semantic embeddings.

4. Cosine similarity compares your resume with job descriptions.

5. The system recommends the most relevant jobs.
""")

    st.markdown("---")

    st.caption(
        "Built using Python • Sentence Transformers • Scikit-learn • Pandas • Streamlit"
    )



