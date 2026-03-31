import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {"and","the","for","with","that","this","are","was","you","have","from","your","our","will","can","has","but","not","all","they","been","its","more","also","their","we","an","in","of","to","a","is","it","on","at","be","as","or","by","do","if","so","up","no","my","he","she","his","her","we","us","me","him","who","how","what","when","where","why","which","would","could","should","about","into","than","then","there","these","those","such","each","any","may","had","him","did","get","got","make","made","use","used","just","like","over","after","before","between","through","during","while","other","some","both","few","more","most","other","same","than","too","very","just","because","since","though","although","however","therefore","thus","hence","either","neither","whether","unless","until","once","even","still","already","yet","again","further","once"}
    return set(w for w in words if w not in stopwords)

def get_match(resume_text, jd_text):
    embeddings = model.encode([resume_text, jd_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)
    
    strengths = resume_kw & jd_kw
    gaps = jd_kw - resume_kw
    
    return round(float(score) * 100, 2), list(strengths)[:15], list(gaps)[:15]

st.set_page_config(page_title="ResumeIQ", page_icon="📄")
st.title("📄 ResumeIQ — Resume Matcher")
st.subheader("See how well your resume matches a job description")

resume_text = ""
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("Resume uploaded!")

jd_text = st.text_area("Paste the Job Description here", height=200)

if st.button("Analyse"):
    if not resume_text:
        st.error("Please upload a resume.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analysing..."):
            score, strengths, gaps = get_match(resume_text, jd_text)
        
        st.markdown("---")
        color = "green" if score >= 60 else "orange" if score >= 40 else "red"
        st.markdown(f"## Match Score: :{color}[{score}%]")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Strengths")
            for s in strengths:
                st.markdown(f"- {s}")
        with col2:
            st.markdown("### ❌ Gaps")
            for g in gaps:
                st.markdown(f"- {g}")
