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
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stopwords = {"with","that","this","have","from","your","will","also","their","about","into","than","then","there","these","those","such","each","been","more","some","both","very","just","when","where","what","which","would","could","should","other","after","before","while","through","during","between","because","however","therefore","thus","either","whether","unless","until","even","still","already","further","once","they","were","over","only","need","make","made","used","like","does","doing","well","must","many","much","most","high","work","area","role","team","time","year","years","based","using","within","across","including","ensure","support","experience","ability","strong","skills","knowledge","understanding","working","provide","develop","manage","business","company","position","required","preferred","related","relevant","following","please","apply","including","candidate","candidates","responsibilities","qualifications","minimum","salary","benefits","employment","opportunity","equal","employer","applicants","without","regard","race","color","religion","national","origin","disability","veteran","status"}
    return set(w for w in words if w not in stopwords)

def get_match(resume_text, jd_text):
    embeddings = model.encode([resume_text, jd_text])
    raw_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    score = max(0, min(100, round(float(raw_score) * 100, 2)))
    
    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)
    
    strengths = list(resume_kw & jd_kw)[:15]
    gaps = list(jd_kw - resume_kw)[:15]
    
    return score, strengths, gaps

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

        
