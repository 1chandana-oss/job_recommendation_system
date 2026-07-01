"""
Resume Analysis Module
"""

import re

# ==========================
# MASTER SKILL LIST
# ==========================

SKILLS = {

    "Python",
    "SQL",
    "Excel",
    "Power BI",
    "Tableau",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Azure",
    "AWS",
    "GCP",
    "Spark",
    "Hadoop",
    "Databricks",
    "Airflow",
    "Docker",
    "Kubernetes",
    "Git",
    "GitHub",
    "Linux",
    "FastAPI",
    "Flask",
    "Streamlit",
    "MongoDB",
    "MySQL",
    "PostgreSQL",
    "Java",
    "C++",
    "JavaScript",
    "React",
    "HTML",
    "CSS"
}

# ==========================
# EXTRACT SKILLS
# ==========================

def extract_skills(resume_text):

    resume_text = resume_text.lower()

    detected = []

    for skill in SKILLS:

        if skill.lower() in resume_text:

            detected.append(skill)

    return sorted(detected)

# ==========================
# RESUME SCORE
# ==========================

def calculate_resume_score(resume_text, skills):

    score = 0

    # Resume Length

    words = len(resume_text.split())

    if words > 300:
        score += 20

    elif words > 150:
        score += 15

    else:
        score += 10

    # Skills

    score += min(len(skills) * 2, 30)

    # Projects

    if "project" in resume_text.lower():
        score += 15

    # Education

    if "education" in resume_text.lower():
        score += 10

    # Experience

    if (
        "experience" in resume_text.lower()
        or "internship" in resume_text.lower()
    ):
        score += 15

    # Certifications

    if "certification" in resume_text.lower():
        score += 10

    return min(score, 100)

# ==========================
# ATS SCORE
# ==========================

def calculate_ats_score(skills):

    return min(60 + len(skills) * 2, 100)

# ==========================
# MISSING SKILLS
# ==========================

def find_missing_skills(
    detected_skills,
    recommended_jobs
):

    required = set()

    if "skills_desc" not in recommended_jobs.columns:
        return []

    for text in recommended_jobs["skills_desc"].dropna():

        text = text.lower()

        for skill in SKILLS:

            if skill.lower() in text:

                required.add(skill)

    missing = sorted(
        required - set(detected_skills)
    )

    return missing