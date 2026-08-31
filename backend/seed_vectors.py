"""
JobLens Vector Seeding Script
Pre-calculates numerical vector representation for all LinkedIn job postings
and uploads them to MongoDB Atlas (Database: JobLens -> Collection: job_vectors).
"""

import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv(override=True)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Dataset",
    "linkedin_jobs_cleaned.csv"
)

MONGO_URI = os.environ.get("MONGO_URI", "")
if not MONGO_URI:
    print("[ERROR] MONGO_URI environment variable not found in .env")
    exit(1)

print(f"[INFO] Connecting to MongoDB Atlas ({MONGO_URI[:30]}...)...")
client = MongoClient(MONGO_URI)
db = client["JobLens"]
vector_col = db["job_vectors"]

print(f"[INFO] Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.fillna("Not Specified")
total_records = len(df)
print(f"[INFO] Loaded {total_records} job postings.")

# Combine relevant text fields for vector representation
df["_doc"] = (
    df["job_title"].astype(str) + " " +
    df["company_name"].astype(str) + " " +
    df["job_function"].astype(str) + " " +
    df["industry"].astype(str) + " " +
    df["location"].astype(str)
)

print("[INFO] Vectorizing documents with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=500,  # 500 dense feature dimensions for compact MongoDB storage
    stop_words="english",
    ngram_range=(1, 2)
)
tfidf_matrix = vectorizer.fit_transform(df["_doc"]).toarray()
print(f"[INFO] Vectorization complete. Dimensions: {tfidf_matrix.shape}")

# Batch insert into MongoDB Atlas
BATCH_SIZE = 1000
print(f"[INFO] Seeding vectors into MongoDB Atlas collection 'job_vectors' in batches of {BATCH_SIZE}...")

vector_col.delete_many({})  # Clear old vectors if re-seeding

documents = []
for idx, row in df.iterrows():
    vec = tfidf_matrix[idx].tolist()
    doc = {
        "job_id": int(idx),
        "job_title": str(row["job_title"]),
        "company_name": str(row["company_name"]),
        "location": str(row["location"]),
        "job_function": str(row["job_function"]),
        "industry": str(row["industry"]),
        "hiring_status": str(row["hiring_status"]),
        "seniority_level": str(row["seniority_level"]),
        "employment_type": str(row["employment_type"]),
        "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
        "vector": vec
    }
    documents.append(doc)

    if len(documents) >= BATCH_SIZE:
        vector_col.insert_many(documents)
        print(f"  Processed {idx + 1} / {total_records} records...")
        documents = []

if documents:
    vector_col.insert_many(documents)
    print(f"  Processed {total_records} / {total_records} records.")

# Create index on job_id for fast retrieval
vector_col.create_index("job_id", unique=True)
print(f"[SUCCESS] Successfully seeded {vector_col.count_documents({})} vector documents into MongoDB Atlas!")
