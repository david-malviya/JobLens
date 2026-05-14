"""
LinkedIn Job Insights Platform — Flask Backend
Loads cleaned CSV into memory and serves REST API endpoints for job filtering and statistics.
Includes JWT-based authentication with registration and login.
"""

import os
import json
import datetime
import functools

import bcrypt
import jwt
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SECRET_KEY = "joblens_super_secret_key_2026"  # In production, use env variable
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

# ---------------------------------------------------------------------------
# Load cleaned dataset into memory
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Dataset",
    "linkedin_jobs_cleaned.csv",
)

print(f"[INFO] Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.fillna("Not Specified")
print(f"[INFO] Loaded {len(df)} job records.")

# ---------------------------------------------------------------------------
# Initialize ML Engine (pre-compute models at startup)
# ---------------------------------------------------------------------------
from ml_engine import SemanticSearch, JobClusterer, TrendForecaster, ResumeJobMatcher

semantic_search = SemanticSearch(df)
job_clusterer = JobClusterer(df, n_clusters=8, sample_size=3000)
trend_forecaster = TrendForecaster(df)
resume_matcher = ResumeJobMatcher(df)
print("[INFO] All ML models initialized.")


# ---------------------------------------------------------------------------
# User storage helpers (JSON file-based)
# ---------------------------------------------------------------------------
def load_users():
    """Load users from the JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    """Save users to the JSON file."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def create_token(email, name):
    """Create a JWT token for the user."""
    payload = {
        "email": email,
        "name": name,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def token_required(f):
    """Decorator to protect routes with JWT authentication."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"error": "Authentication token is missing"}), 401

        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user = data
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired, please login again"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

@app.route("/api/auth/register", methods=["POST"])
def register():
    """Register a new user."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    # Validation
    if not name or len(name) < 2:
        return jsonify({"error": "Name must be at least 2 characters"}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required"}), 400
    if not password or len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    users = load_users()

    if email in users:
        return jsonify({"error": "An account with this email already exists"}), 409

    # Hash password and save
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    users[email] = {
        "name": name,
        "email": email,
        "password": hashed,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    save_users(users)

    token = create_token(email, name)
    return jsonify({
        "message": "Registration successful",
        "token": token,
        "user": {"name": name, "email": email},
    }), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Login an existing user."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    users = load_users()
    user = users.get(email)

    if not user:
        return jsonify({"error": "No account found with this email"}), 404

    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        return jsonify({"error": "Incorrect password"}), 401

    token = create_token(email, user["name"])
    return jsonify({
        "message": "Login successful",
        "token": token,
        "user": {"name": user["name"], "email": user["email"]},
    }), 200


@app.route("/api/auth/me", methods=["GET"])
@token_required
def get_me():
    """Get current authenticated user info."""
    return jsonify({"user": {"name": request.user["name"], "email": request.user["email"]}}), 200


# ---------------------------------------------------------------------------
# Helper: apply filters to dataframe
# ---------------------------------------------------------------------------
def apply_filters(data):
    """Apply query-string filters to the dataframe and return filtered copy."""
    filtered = data.copy()

    # Text search across job_title and company_name
    search = request.args.get("search", "").strip()
    if search:
        mask = (
            filtered["job_title"].str.contains(search, case=False, na=False)
            | filtered["company_name"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    # Exact-match filters
    location = request.args.get("location", "").strip()
    if location:
        filtered = filtered[
            filtered["location"].str.contains(location, case=False, na=False)
        ]

    role = request.args.get("role", "").strip()
    if role:
        filtered = filtered[
            filtered["job_title"].str.contains(role, case=False, na=False)
        ]

    job_function = request.args.get("job_function", "").strip()
    if job_function:
        filtered = filtered[
            filtered["job_function"].str.contains(job_function, case=False, na=False)
        ]

    hiring_status = request.args.get("hiring_status", "").strip()
    if hiring_status:
        filtered = filtered[filtered["hiring_status"] == hiring_status]

    seniority = request.args.get("seniority", "").strip()
    if seniority:
        filtered = filtered[filtered["seniority_level"] == seniority]

    employment_type = request.args.get("employment_type", "").strip()
    if employment_type:
        filtered = filtered[filtered["employment_type"] == employment_type]

    industry = request.args.get("industry", "").strip()
    if industry:
        filtered = filtered[
            filtered["industry"].str.contains(industry, case=False, na=False)
        ]

    return filtered


# ---------------------------------------------------------------------------
# Job Routes (protected — require login)
# ---------------------------------------------------------------------------

@app.route("/api/jobs", methods=["GET"])
@token_required
def get_jobs():
    """
    Return paginated, filtered job listings.
    Query params: search, location, role, job_function, hiring_status,
                  seniority, employment_type, industry, page, per_page
    """
    filtered = apply_filters(df)

    # Pagination
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(1, int(request.args.get("per_page", 20))))
    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered.iloc[start:end]

    jobs = []
    for _, row in page_data.iterrows():
        jobs.append(
            {
                "job_title": row["job_title"],
                "company_name": row["company_name"],
                "location": row["location"],
                "hiring_status": row["hiring_status"],
                "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
                "seniority_level": row["seniority_level"],
                "job_function": row["job_function"],
                "employment_type": row["employment_type"],
                "industry": row["industry"],
            }
        )

    return jsonify(
        {
            "jobs": jobs,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": max(1, -(-total // per_page)),  # ceil division
            },
        }
    )


@app.route("/api/jobs/stats", methods=["GET"])
@token_required
def get_stats():
    """Return dashboard-level statistics."""
    return jsonify(
        {
            "total_jobs": len(df),
            "total_companies": df["company_name"].nunique(),
            "total_locations": df["location"].nunique(),
            "actively_hiring": int((df["hiring_status"] == "Actively Hiring").sum()),
            "early_applicant": int((df["hiring_status"] == "Early Applicant").sum()),
            "top_locations": df["location"]
                .value_counts()
                .head(10)
                .to_dict(),
            "top_roles": df["job_title"]
                .value_counts()
                .head(10)
                .to_dict(),
            "top_companies": df["company_name"]
                .value_counts()
                .head(10)
                .to_dict(),
            "employment_type_dist": df["employment_type"]
                .value_counts()
                .to_dict(),
            "seniority_dist": df["seniority_level"]
                .value_counts()
                .to_dict(),
            "hiring_status_dist": df["hiring_status"]
                .value_counts()
                .to_dict(),
        }
    )


@app.route("/api/jobs/filters", methods=["GET"])
@token_required
def get_filters():
    """Return unique values for each filterable column (for populating dropdowns)."""
    return jsonify(
        {
            "locations": sorted(df["location"].unique().tolist()),
            "job_functions": sorted(
                [x for x in df["job_function"].unique().tolist() if x != "Not Specified"]
            ),
            "seniority_levels": sorted(
                [x for x in df["seniority_level"].unique().tolist() if x != "Not Specified"]
            ),
            "employment_types": sorted(
                [x for x in df["employment_type"].unique().tolist() if x != "Not Specified"]
            ),
            "hiring_statuses": sorted(df["hiring_status"].unique().tolist()),
            "industries": sorted(
                [x for x in df["industry"].unique().tolist() if x != "Not Specified"]
            )[:100],  # cap at 100 for UI
        }
    )


# ---------------------------------------------------------------------------
# ML Routes
# ---------------------------------------------------------------------------

@app.route("/api/ml/semantic-search", methods=["POST"])
@token_required
def ml_semantic_search():
    """NLP-powered semantic job search using TF-IDF + Cosine Similarity."""
    data = request.get_json()
    if not data or not data.get("query"):
        return jsonify({"error": "Query is required"}), 400

    query = data["query"].strip()
    top_k = min(50, max(1, int(data.get("top_k", 20))))
    results = semantic_search.search(query, top_k=top_k)

    return jsonify({
        "query": query,
        "results": results,
        "total": len(results),
        "method": "TF-IDF + Cosine Similarity",
    })


@app.route("/api/ml/clusters", methods=["GET"])
@token_required
def ml_clusters():
    """Return K-Means cluster assignments + PCA 2D coordinates."""
    data = job_clusterer.get_cluster_data()
    return jsonify(data)


@app.route("/api/ml/trends", methods=["GET"])
@token_required
def ml_trends():
    """Time-series job posting trends + polynomial regression forecast."""
    role = request.args.get("role", "").strip() or None
    months_ahead = min(12, max(1, int(request.args.get("months", 6))))
    data = trend_forecaster.get_trends(role=role, months_ahead=months_ahead)
    return jsonify(data)


@app.route("/api/ml/resume-match", methods=["POST"])
@token_required
def ml_resume_match():
    """Match resume/skills text against jobs using TF-IDF similarity."""
    data = request.get_json()
    if not data or not data.get("resume_text"):
        return jsonify({"error": "resume_text is required"}), 400

    resume_text = data["resume_text"].strip()
    top_k = min(30, max(1, int(data.get("top_k", 15))))
    results = resume_matcher.match(resume_text, top_k=top_k)

    return jsonify({
        "results": results,
        "total": len(results),
        "method": "TF-IDF + Cosine Similarity (Resume Matching)",
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
