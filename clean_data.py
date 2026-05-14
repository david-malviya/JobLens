"""
Data Cleaning Script for LinkedIn Job Posts Dataset
Cleans whitespace artifacts, normalizes categories, and exports cleaned CSV.
"""

import pandas as pd
import re
import os

INPUT_FILE = os.path.join("Dataset", "linkedin_job_posts_insights.xlsx")
OUTPUT_FILE = os.path.join("Dataset", "linkedin_jobs_cleaned.csv")


def clean_whitespace(value):
    """Strip \\n, \\t, and excess whitespace from a string value."""
    if pd.isna(value):
        return value
    cleaned = re.sub(r'[\n\r\t]+', ' ', str(value))
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()


def normalize_hiring_status(status):
    """Normalize hiring status to clean categories."""
    if pd.isna(status):
        return "Not Specified"
    status = clean_whitespace(status).lower()
    if "actively hiring" in status:
        return "Actively Hiring"
    elif "early applicant" in status:
        return "Early Applicant"
    elif "promoted" in status:
        return "Promoted"
    else:
        return status.title()


def normalize_seniority(level):
    """Normalize seniority level."""
    if pd.isna(level):
        return "Not Specified"
    level = clean_whitespace(level)
    # Map common variations
    level_map = {
        "mid-senior level": "Mid-Senior Level",
        "entry level": "Entry Level",
        "not applicable": "Not Applicable",
        "associate": "Associate",
        "executive": "Executive",
        "director": "Director",
        "internship": "Internship",
    }
    lower = level.lower()
    for key, val in level_map.items():
        if key in lower:
            return val
    return level.title()


def normalize_employment_type(emp_type):
    """Normalize employment type."""
    if pd.isna(emp_type):
        return "Not Specified"
    emp_type = clean_whitespace(emp_type)
    type_map = {
        "full-time": "Full-time",
        "part-time": "Part-time",
        "contract": "Contract",
        "temporary": "Temporary",
        "internship": "Internship",
        "volunteer": "Volunteer",
        "other": "Other",
    }
    lower = emp_type.lower()
    for key, val in type_map.items():
        if key in lower:
            return val
    return emp_type.title()


def is_malformed_row(row):
    """
    Detect rows where all data is crammed into one column.
    These rows typically have \\n delimiters inside the job_title field.
    """
    job_title = str(row.get("job_title", ""))
    # If job_title contains multiple \\n separated segments, it's malformed
    if job_title.count("\\n") > 2:
        return True
    # Also check if it contains patterns like "Company Name\\n" mixed in
    if "\\n" in job_title and any(
        field in job_title.lower()
        for field in ["full-time", "part-time", "actively hiring", "early applicant"]
    ):
        return True
    return False


def main():
    print("📂 Loading dataset...")
    df = pd.read_excel(INPUT_FILE)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # --- Step 1: Drop malformed rows ---
    print("\n🧹 Removing malformed rows...")
    malformed_mask = df.apply(is_malformed_row, axis=1)
    malformed_count = malformed_mask.sum()
    df = df[~malformed_mask].copy()
    print(f"   Removed {malformed_count} malformed rows")

    # --- Step 2: Clean whitespace from all string columns ---
    print("\n✨ Cleaning whitespace artifacts...")
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].apply(clean_whitespace)

    # --- Step 3: Normalize categorical columns ---
    print("🏷️  Normalizing categories...")
    df["hiring_status"] = df["hiring_status"].apply(normalize_hiring_status)
    df["seniority_level"] = df["seniority_level"].apply(normalize_seniority)
    df["employment_type"] = df["employment_type"].apply(normalize_employment_type)
    df["job_function"] = df["job_function"].apply(
        lambda x: clean_whitespace(x) if pd.notna(x) else "Not Specified"
    )
    df["industry"] = df["industry"].apply(
        lambda x: clean_whitespace(x) if pd.notna(x) else "Not Specified"
    )
    df["company_name"] = df["company_name"].fillna("Not Specified")
    df["location"] = df["location"].fillna("Not Specified")
    df["job_title"] = df["job_title"].fillna("Not Specified")

    # Fix seniority values that are actually employment types (data leakage)
    employment_in_seniority = ["Full-Time", "Part-Time", "Contract", "Volunteer", "Other"]
    df.loc[
        df["seniority_level"].isin(employment_in_seniority), "seniority_level"
    ] = "Not Specified"

    # Normalize rare hiring_status variants (Medical Insurance benefits → Actively Hiring)
    df.loc[
        df["hiring_status"].str.contains("Medical Insurance|Benefits", case=False, na=False),
        "hiring_status",
    ] = "Actively Hiring"

    # --- Step 4: Drop rows with missing job title ---
    print("🗑️  Dropping rows with empty job titles...")
    df = df[df["job_title"] != "Not Specified"].copy()
    df = df[df["job_title"].str.len() > 3].copy()

    # --- Step 5: Sort by date descending ---
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # --- Step 6: Export ---
    print(f"\n💾 Exporting to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # --- Summary ---
    print(f"\n✅ Done! Cleaned dataset: {len(df)} rows")
    print(f"\n📊 Column summary:")
    for col in df.columns:
        unique = df[col].nunique()
        nulls = df[col].isnull().sum()
        print(f"   {col}: {unique} unique values, {nulls} nulls")

    print(f"\n📈 Hiring status distribution:")
    print(df["hiring_status"].value_counts().to_string())

    print(f"\n📈 Employment type distribution:")
    print(df["employment_type"].value_counts().to_string())

    print(f"\n📈 Seniority level distribution:")
    print(df["seniority_level"].value_counts().to_string())


if __name__ == "__main__":
    main()
