# 📄 ResumeIQ — Resume Matcher

An AI-powered web app that matches your resume to a job description.

## Live App
🔗 [Click here to use the app](https://resumeiq-tki3cpg6rjcbnnmdetnac7.streamlit.app)
## Video Walkthrough
🎥 [Watch the demo](https://drive.google.com/file/d/1aeaitLyaHhf2X9LkrxO1pqdjiLROMuBP/view?usp=drivesdk)


## What it does
- Upload your resume (PDF) and paste a job description
- Get a match score from 0–100%
- See your strengths and skill gaps

## Tech Stack
- **Frontend/UI**: Streamlit
- **AI/NLP**: Sentence Transformers (all-MiniLM-L6-v2)
- **Similarity**: Cosine Similarity
- **PDF Parsing**: pdfplumber

## How it works
1. Resume and JD are converted to vector embeddings using Sentence Transformers
2. Cosine similarity is calculated between the two vectors
3. Keywords are extracted and compared to show strengths and gaps

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
