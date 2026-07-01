
"""
Utility functions for AI Job Recommendation System
"""

import pdfplumber


def extract_resume_text(uploaded_file):
    """
    Extract text from uploaded PDF resume.
    """

    resume_text = ""

    with pdfplumber.open(uploaded_file) as pdf:

        total_pages = len(pdf.pages)

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                resume_text += text + "\n"

    return resume_text, total_pages


def get_resume_statistics(resume_text, total_pages):
    """
    Generate resume statistics.
    """

    statistics = {
        "pages": total_pages,
        "characters": len(resume_text),
        "words": len(resume_text.split())
    }

    return statistics


def clean_text(text):
    """
    Clean extracted text.
    """

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text

