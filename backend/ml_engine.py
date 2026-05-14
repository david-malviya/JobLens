"""
JobLens ML Engine
Contains 4 ML modules:
  1. SemanticSearch  — TF-IDF + Cosine Similarity for NLP-powered job search
  2. JobClusterer    — K-Means clustering + PCA for job grouping & visualization
  3. TrendForecaster — Time-series analysis + regression for demand forecasting
  4. ResumeJobMatcher — TF-IDF similarity for resume-to-job matching
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class SemanticSearch:
    """
    NLP-powered search using TF-IDF vectorization and cosine similarity.
    Unlike keyword matching, this understands semantic relationships:
    e.g. "ML engineer" also finds "Machine Learning", "Deep Learning" roles.
    """

    def __init__(self, df):
        print("[ML] Training SemanticSearch (TF-IDF)...")
        self.df = df.copy()

        # Combine text fields into a single searchable document per job
        self.df["_doc"] = (
            self.df["job_title"].fillna("")
            + " " + self.df["company_name"].fillna("")
            + " " + self.df["job_function"].fillna("")
            + " " + self.df["industry"].fillna("")
            + " " + self.df["location"].fillna("")
        )

        # Fit TF-IDF vectorizer on the job corpus
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams
            min_df=2,
            max_df=0.95,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["_doc"])
        print(f"[ML] SemanticSearch ready. Vocabulary: {len(self.vectorizer.vocabulary_)} terms")

    def search(self, query, top_k=20):
        """
        Find jobs semantically similar to the query.
        Returns list of (job_dict, similarity_score) tuples.
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-K indices sorted by similarity
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.01:
                continue  # Skip irrelevant results
            row = self.df.iloc[idx]
            results.append({
                "job_title": row["job_title"],
                "company_name": row["company_name"],
                "location": row["location"],
                "hiring_status": row["hiring_status"],
                "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
                "seniority_level": row["seniority_level"],
                "employment_type": row["employment_type"],
                "industry": row["industry"],
                "match_score": round(score * 100, 1),
            })
        return results


class JobClusterer:
    """
    Unsupervised job clustering using K-Means on TF-IDF features.
    PCA reduces dimensions to 2D for scatter plot visualization.
    """

    def __init__(self, df, n_clusters=8, sample_size=3000):
        print(f"[ML] Training JobClusterer (K-Means, k={n_clusters})...")
        self.n_clusters = n_clusters

        # Sample for performance (K-Means on 31K is slow)
        if len(df) > sample_size:
            self.df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            self.df = df.copy()

        # Build document features
        docs = (
            self.df["job_title"].fillna("")
            + " " + self.df["job_function"].fillna("")
            + " " + self.df["industry"].fillna("")
        )

        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
        )
        tfidf_matrix = self.vectorizer.fit_transform(docs)

        # K-Means clustering
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        self.labels = self.kmeans.fit_predict(tfidf_matrix)

        # PCA for 2D visualization
        self.pca = PCA(n_components=2, random_state=42)
        self.coords = self.pca.fit_transform(tfidf_matrix.toarray())

        # Generate cluster names from top terms
        self.cluster_names = self._name_clusters(tfidf_matrix)

        print(f"[ML] JobClusterer ready. {n_clusters} clusters on {len(self.df)} samples")

    def _name_clusters(self, tfidf_matrix):
        """Derive cluster names from the most frequent job titles in each cluster."""
        names = {}
        for k in range(self.n_clusters):
            mask = self.labels == k
            cluster_titles = self.df.loc[mask, "job_title"].value_counts().head(3)
            top_titles = list(cluster_titles.index)
            names[k] = ", ".join(top_titles[:2]) if top_titles else f"Cluster {k}"
        return names

    def get_cluster_data(self):
        """Return cluster assignments + 2D PCA coordinates for visualization."""
        points = []
        for i in range(len(self.df)):
            points.append({
                "x": round(float(self.coords[i, 0]), 4),
                "y": round(float(self.coords[i, 1]), 4),
                "cluster": int(self.labels[i]),
                "job_title": self.df.iloc[i]["job_title"],
                "company": self.df.iloc[i]["company_name"],
            })

        return {
            "points": points,
            "cluster_names": self.cluster_names,
            "n_clusters": self.n_clusters,
            "pca_variance": [round(float(v), 4) for v in self.pca.explained_variance_ratio_],
        }


class TrendForecaster:
    """
    Time-series analysis of job posting volume.
    Uses polynomial regression to forecast future trends.
    """

    def __init__(self, df):
        print("[ML] Training TrendForecaster...")
        self.df = df.copy()
        self.df["month"] = self.df["date"].dt.to_period("M")
        print("[ML] TrendForecaster ready.")

    def get_trends(self, role=None, months_ahead=6):
        """
        Get historical monthly job counts + forecast.
        Optionally filter by role (job title contains role string).
        """
        filtered = self.df
        if role:
            filtered = filtered[
                filtered["job_title"].str.contains(role, case=False, na=False)
            ]

        if len(filtered) < 10:
            return {"historical": [], "forecast": [], "role": role}

        # Aggregate monthly counts
        monthly = filtered.groupby("month").size().reset_index(name="count")
        monthly["month_str"] = monthly["month"].astype(str)
        monthly = monthly.sort_values("month")

        # Prepare features for regression (month index as feature)
        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly["count"].values

        # Polynomial regression (degree 2) for trend
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        # Trend line for historical data
        trend_hist = model.predict(X_poly).tolist()

        # Forecast future months
        future_X = np.arange(len(monthly), len(monthly) + months_ahead).reshape(-1, 1)
        future_X_poly = poly.transform(future_X)
        forecast_values = model.predict(future_X_poly)
        forecast_values = np.maximum(forecast_values, 0)  # no negative counts

        # Generate future month labels
        last_period = monthly["month"].iloc[-1]
        future_labels = []
        for i in range(1, months_ahead + 1):
            future_period = last_period + i
            future_labels.append(str(future_period))

        # Top roles breakdown
        top_roles = filtered["job_title"].value_counts().head(8).to_dict()

        return {
            "historical": [
                {"month": row["month_str"], "count": int(row["count"]), "trend": round(t, 1)}
                for (_, row), t in zip(monthly.iterrows(), trend_hist)
            ],
            "forecast": [
                {"month": label, "count": round(float(val), 1)}
                for label, val in zip(future_labels, forecast_values)
            ],
            "role": role,
            "total_jobs": len(filtered),
            "top_roles": top_roles,
            "r_squared": round(float(model.score(X_poly, y)), 4),
        }


class ResumeJobMatcher:
    """
    Matches resume text against the job corpus using TF-IDF + Cosine Similarity.
    The TF-IDF vectorizer is fit on the job corpus, then the resume is transformed
    into the same space to find the most relevant jobs.
    """

    def __init__(self, df):
        print("[ML] Training ResumeJobMatcher (TF-IDF)...")
        self.df = df.copy()

        # Build rich job documents
        self.df["_doc"] = (
            self.df["job_title"].fillna("")
            + " " + self.df["job_function"].fillna("")
            + " " + self.df["industry"].fillna("")
            + " " + self.df["seniority_level"].fillna("")
            + " " + self.df["location"].fillna("")
        )

        self.vectorizer = TfidfVectorizer(
            max_features=6000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["_doc"])
        print(f"[ML] ResumeJobMatcher ready. Vocabulary: {len(self.vectorizer.vocabulary_)} terms")

    def match(self, resume_text, top_k=15):
        """
        Match resume text against jobs. Returns ranked list with match scores.
        """
        resume_vec = self.vectorizer.transform([resume_text])
        similarities = cosine_similarity(resume_vec, self.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.005:
                continue
            row = self.df.iloc[idx]
            results.append({
                "job_title": row["job_title"],
                "company_name": row["company_name"],
                "location": row["location"],
                "hiring_status": row["hiring_status"],
                "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
                "seniority_level": row["seniority_level"],
                "employment_type": row["employment_type"],
                "industry": row["industry"],
                "match_score": round(score * 100, 1),
            })
        return results
