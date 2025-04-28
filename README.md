## AI Resume Screening System

 ## 📄 Overview
AI Resume Screening is a machine learning-powered application built to streamline the candidate shortlisting process for HR professionals. This project automates resume parsing, classification, and job-role matching using advanced Natural Language Processing (NLP) techniques. It offers real-time insights to both recruiters and job seekers, significantly reducing hiring time and effort.

## 💼 Key Features
- **Automated Resume Classification**: Uses TF-IDF and Word2Vec models to extract features and classify resumes based on job relevance.
- **Role Matching & Feedback**: Predicts the best-fit job roles for non-matching candidates and provides immediate feedback.
- **High Accuracy**: Achieved a classification accuracy of **91%** during evaluation.
- **Real-Time Processing**: Screens each resume in under **5 seconds**.
- **HR Dashboard**: Displays detailed candidate information including Name, Email, Location, and Resume match score.

## 🧠 Tech Stack & Tools
- **Languages & Libraries**: Python, Scikit-learn, Pandas, NumPy, Pickle, Joblib
- **Machine Learning Models**: TF-IDF, Word2Vec, Random Forest
- **Frameworks**: Streamlit (for UI), Flask (for backend APIs)
- **Database**: MySQL for storing shortlisted candidate information
- **Additional Tools**: Resume Parser, SQL Integration

 ## Project Components
- `Upload_Resume.py`: Streamlit UI for candidate resume uploads.
- `HR.py`: HR interface to input the job role and trigger candidate filtering.
- `model_training.ipynb`: Notebook detailing model training using multiple algorithms.
- `cv.pickle`: Pickled TF-IDF vectorizer trained on cleaned resume data.
- `RF.joblib.zip`: Compressed Random Forest model — the best performing model.
- `SQL.txt`: SQL script to create a candidate database schema.
- `pages/Show_Resumes.py`: Streamlit subpage displaying shortlisted candidates with scores.

##  Datasets Used
- [Resume Dataset – Gaurav Dutta (Kaggle)](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- [Resume Dataset – Snehaan Bhawal (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

## Usage Instructions
1. HR Input  
   Launch the HR portal:  
   ```bash
   streamlit run HR.py
   Candidate Upload
   
## Share the candidate page or run it locally:

streamlit run Upload_Resume.py
Resume Review
View filtered resumes (sorted by match score) via:
pages/Show_Resumes.py
Ensure the pages/ folder is in the same directory.

## Outcome
This project automated resume screening and significantly enhanced candidate-job role matching by using pre-trained NLP models. It reduced resume processing time by 45%, improved hiring precision, and delivered a seamless experience for both HR teams and job applicants.
