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

## Tech Stack
*   **Frontend:** Vanilla HTML5, CSS3, JavaScript, Chart.js
*   **Backend:** Python (Flask)
*   **Machine Learning:** Scikit-Learn, Pandas, NumPy
*   **Data:** Cleaned and processed LinkedIn job postings dataset

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
*   `/frontend` - Contains the UI, stylesheets, and client-side logic (`app.js`, `ml.js`, `auth.js`).
*   `/backend` - Contains the server (`app.py`), the Machine Learning engine (`ml_engine.py`), and API endpoints.
*   `/Dataset` - Stores the cleaned `.csv` and insights `.xlsx` files used to train models and serve data.
*   `clean_data.py` - Script used for initial dataset cleaning and preprocessing.
