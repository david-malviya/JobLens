# JobLens — LinkedIn Job Insights Platform

![JobLens Banner](https://img.shields.io/badge/AI--Powered-Job%20Insights-blue?style=for-the-badge)

## Overview
**JobLens** is a full-stack job discovery platform designed to analyze and explore over 31,000 LinkedIn job postings. It goes beyond simple keyword searches by integrating machine learning models for semantic search, unsupervised job clustering, market trend forecasting, and direct resume-to-job matching.

## Features
*   **Smart Filtering:** Browse thousands of jobs with dynamic filters for Location, Function, Seniority, Employment Type, and Hiring Status.
*   **Semantic Search:** Powered by **TF-IDF** and **Cosine Similarity**, find jobs by their underlying meaning and context rather than strict keyword matches.
*   **Job Clustering:** Uses **K-Means** and **PCA** (Principal Component Analysis) to group jobs into distinct clusters, helping you discover related roles visually.
*   **Trend Forecasting:** Employs **Polynomial Regression** to analyze job posting volumes over time and predict future 6-month hiring trends.
*   **Resume Matching:** Paste your resume or skill set and let the platform rank jobs based on how well they match your experience.
*   **AI Data Analyst Chat:** An intelligent natural language interface powered by Groq and Llama 3 (or Gemini) to ask complex questions about the job market, trends, and specific companies directly from the dataset.

## Tech Stack
*   **Frontend:** Vanilla HTML5, CSS3, JavaScript, Chart.js
*   **Backend:** Python (Flask)
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **LLM Integration:** Groq API, Llama 3 / Gemini
*   **Data:** Cleaned and processed LinkedIn job postings dataset

## The Dataset
The dataset consists of one primary sheet. Each record in the dataset represents a single job posting, and the attributes captured include:

*   **job_title:** The job title for the position being advertised (e.g., "Software Engineer," "Marketing Manager").
*   **company_name:** The name of the company that is advertising the job (e.g., "Google," "Microsoft").
*   **location:** The geographic location of the job posting, including city, state, and country.
*   **hiring_status:** The current hiring status indicating whether the position is currently open or closed.
*   **date:** The date on which the job posting was created or last updated.
*   **seniority_level:** The level of seniority of the position (e.g., "Entry-Level," "Senior-Level").
*   **job_function:** The functional area of the job being advertised (e.g., "Marketing," "Engineering").
*   **employment_type:** The type of employment being offered (e.g., "Full-Time," "Contract").
*   **industry:** The industry of the company advertising the job (e.g., "Technology," "Healthcare").

## Getting Started

### Prerequisites
*   Python 3.8+
*   Node.js (for the frontend dev server)

### Backend Setup
1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend server (runs on port 5000):
   ```bash
   python app.py
   ```

### Frontend Setup
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the node modules:
   ```bash
   npm install
   ```
3. Start the frontend development server:
   ```bash
   npm run dev
   ```

## Project Structure
*   `/frontend` - Contains the UI, stylesheets, and client-side logic (`app.js`, `ml.js`, `llm_chat.js`, `auth.js`).
*   `/backend` - Contains the server (`app.py`), the Machine Learning engine (`ml_engine.py`), the LLM Engine (`llm_engine.py`), and API endpoints.
*   `/Dataset` - Stores the cleaned `.csv` and insights `.xlsx` files used to train models and serve data.
*   `clean_data.py` - Script used for initial dataset cleaning and preprocessing.
