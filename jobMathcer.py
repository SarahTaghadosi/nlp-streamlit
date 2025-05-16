import streamlit as st
st.set_page_config(layout="wide")

import spacy
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import euclidean

nlp = spacy.load("en_core_web_lg")

def get_vector(keywords):
    vectors = [nlp(word).vector for word in keywords if nlp(word).has_vector]
    return np.mean(vectors, axis=0) if vectors else np.zeros(nlp.vocab.vectors_length)

@st.cache_data
def load_data():
    with open("/Users/sarahtgh/Desktop/nlp/task/job_opportunities.json", "r", encoding="utf-8") as jf:
        jobs = json.load(jf)
    with open("/Users/sarahtgh/Desktop/nlp/task/resumes.json", "r", encoding="utf-8") as rf:
        resumes = json.load(rf)
    return jobs, resumes

jobs, resumes = load_data()

#st.set_page_config(layout="wide")
st.sidebar.title("Job Opportunities")

selected_job = None
for job in jobs:
    if st.sidebar.button(f"Select Job: {job['title']}"):
        selected_job = job

if selected_job:
    st.title("Job and Resume Matching Application")
    st.subheader(f"Job Opportunity: {selected_job['title']}")

    st.markdown("**Key Words:** " + ", ".join(selected_job["key_words"]))
    st.markdown("**Job Description:** " + selected_job["text"])

    job_vector = get_vector(selected_job["key_words"])

    results = []
    for res in resumes:
        res_vector = get_vector(res["key_words"])
        dist = euclidean(job_vector, res_vector)
        results.append({
            "title": res["title"],
            "distance": dist,
            "id": res["id"],
            "text": res["text"],
            "key_words": res["key_words"]
        })

    sorted_resumes = sorted(results, key=lambda x: x["distance"])[:3]

    st.subheader("Top 3 Matched Resumes:")

    for i, r in enumerate(sorted_resumes, start=1):
        with st.container():
            st.markdown(f"### {i}. {r['title']}")
            st.markdown(f"**Similarity Score:** {round(10 - r['distance'], 2)}")
            st.markdown(f"**Key Words:** {', '.join(r['key_words'])}")
            st.markdown(f"**Resume ID:** `{r['id']}`")
            with st.expander("📄 Description"):
                st.write(r["text"])
            st.markdown("---")

else:
    st.title("Job and Resume Matching Application")
    st.markdown("👈 Select a job from the sidebar to begin.")
