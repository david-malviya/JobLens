"""
JobLens LLM Engine — Gemini Integration
Uses Google Gemini API to answer natural language questions about the LinkedIn job dataset.
Retrieves relevant data from the dataset and provides it as context to the LLM.
"""

import os
import json
from groq import Groq
import pandas as pd


class JobLensLLM:
    """
    LLM-powered Q&A engine for the LinkedIn jobs dataset.
    Automatically extracts relevant dataset context based on the user's question
    and sends it to Gemini for an intelligent, data-driven answer.
    """

    def __init__(self, df, api_key=None, retriever=None):
        print("[LLM] Initializing Groq LLM Engine...")
        self.df = df.copy()
        self.retriever = retriever
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        print(f"[DEBUG] Final API Key length: {len(self.api_key)}")

        if not self.api_key:
            print("[LLM] WARNING: No Groq/Gemini API key provided. LLM features will be disabled.")
            self.client = None
            return

        self.client = Groq(api_key=self.api_key)

        # Pre-compute dataset summary for context
        self._dataset_summary = self._build_dataset_summary()
        print("[LLM] Groq LLM Engine ready.")

    def _build_dataset_summary(self):
        """Build a concise statistical summary of the dataset."""
        summary = {
            "total_jobs": len(self.df),
            "total_companies": self.df["company_name"].nunique(),
            "total_locations": self.df["location"].nunique(),
            "date_range": f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}",
            "top_10_job_titles": self.df["job_title"].value_counts().head(10).to_dict(),
            "top_10_companies": self.df["company_name"].value_counts().head(10).to_dict(),
            "top_10_locations": self.df["location"].value_counts().head(10).to_dict(),
            "employment_types": self.df["employment_type"].value_counts().to_dict(),
            "seniority_levels": self.df["seniority_level"].value_counts().to_dict(),
            "hiring_status": self.df["hiring_status"].value_counts().to_dict(),
            "top_industries": self.df["industry"].value_counts().head(15).to_dict(),
            "top_job_functions": self.df["job_function"].value_counts().head(15).to_dict(),
        }
        return summary

    def _extract_relevant_data(self, question):
        """
        Extract relevant slices of data based on keywords in the question.
        This gives the LLM focused context rather than the entire dataset.
        """
        question_lower = question.lower()
        context_parts = []

        # Always include the dataset summary
        context_parts.append(f"DATASET OVERVIEW:\n{json.dumps(self._dataset_summary, indent=2, default=str)}")

        # Check for specific company mentions
        companies = self.df["company_name"].unique()
        mentioned_companies = [c for c in companies if c.lower() in question_lower and c != "Not Specified"]
        if mentioned_companies:
            for company in mentioned_companies[:3]:
                company_data = self.df[self.df["company_name"] == company]
                company_info = {
                    "company": company,
                    "total_jobs": len(company_data),
                    "job_titles": company_data["job_title"].value_counts().head(5).to_dict(),
                    "locations": company_data["location"].value_counts().head(5).to_dict(),
                    "hiring_status": company_data["hiring_status"].value_counts().to_dict(),
                    "seniority_levels": company_data["seniority_level"].value_counts().to_dict(),
                    "employment_types": company_data["employment_type"].value_counts().to_dict(),
                }
                context_parts.append(f"COMPANY DATA - {company}:\n{json.dumps(company_info, indent=2, default=str)}")

        # Check for specific location mentions
        locations = self.df["location"].unique()
        mentioned_locations = [l for l in locations if l.lower() in question_lower and l != "Not Specified"]
        if mentioned_locations:
            for location in mentioned_locations[:3]:
                loc_data = self.df[self.df["location"] == location]
                loc_info = {
                    "location": location,
                    "total_jobs": len(loc_data),
                    "top_job_titles": loc_data["job_title"].value_counts().head(8).to_dict(),
                    "top_companies": loc_data["company_name"].value_counts().head(8).to_dict(),
                    "hiring_status": loc_data["hiring_status"].value_counts().to_dict(),
                    "employment_types": loc_data["employment_type"].value_counts().to_dict(),
                }
                context_parts.append(f"LOCATION DATA - {location}:\n{json.dumps(loc_info, indent=2, default=str)}")

        # Check for role/title keywords
        role_keywords = ["engineer", "developer", "analyst", "manager", "designer",
                         "scientist", "intern", "director", "consultant", "architect",
                         "lead", "senior", "junior", "associate", "data", "software",
                         "product", "marketing", "sales", "finance", "hr", "devops",
                         "fullstack", "frontend", "backend", "cloud", "ai", "ml",
                         "machine learning", "deep learning", "python", "java", "react"]
        matched_roles = [kw for kw in role_keywords if kw in question_lower]

        if matched_roles:
            for keyword in matched_roles[:3]:
                role_data = self.df[self.df["job_title"].str.contains(keyword, case=False, na=False)]
                if len(role_data) > 0:
                    role_info = {
                        "keyword": keyword,
                        "matching_jobs": len(role_data),
                        "top_titles": role_data["job_title"].value_counts().head(8).to_dict(),
                        "top_companies": role_data["company_name"].value_counts().head(8).to_dict(),
                        "top_locations": role_data["location"].value_counts().head(8).to_dict(),
                        "hiring_status": role_data["hiring_status"].value_counts().to_dict(),
                        "seniority_levels": role_data["seniority_level"].value_counts().to_dict(),
                        "employment_types": role_data["employment_type"].value_counts().to_dict(),
                    }
                    context_parts.append(f"ROLE DATA - '{keyword}':\n{json.dumps(role_info, indent=2, default=str)}")

        # Check for trend/time-related questions
        time_keywords = ["trend", "growth", "decline", "increase", "decrease", "month", "year",
                         "over time", "growing", "demand", "forecast", "future", "prediction"]
        if any(kw in question_lower for kw in time_keywords):
            monthly = self.df.groupby(self.df["date"].dt.to_period("M")).size()
            trend_data = {
                "monthly_job_postings": {str(k): int(v) for k, v in monthly.tail(12).items()},
                "total_months_covered": len(monthly),
            }
            # If a role was mentioned, also show trend for that role
            if matched_roles:
                for keyword in matched_roles[:2]:
                    role_df = self.df[self.df["job_title"].str.contains(keyword, case=False, na=False)]
                    if len(role_df) > 5:
                        role_monthly = role_df.groupby(role_df["date"].dt.to_period("M")).size()
                        trend_data[f"monthly_{keyword}_postings"] = {str(k): int(v) for k, v in role_monthly.tail(12).items()}

            context_parts.append(f"TREND DATA:\n{json.dumps(trend_data, indent=2, default=str)}")

        # Check for comparison questions
        compare_keywords = ["compare", "vs", "versus", "difference", "better", "which", "best",
                            "top", "highest", "lowest", "most", "least", "popular", "common"]
        if any(kw in question_lower for kw in compare_keywords):
            comparison_data = {
                "jobs_by_seniority": self.df["seniority_level"].value_counts().to_dict(),
                "jobs_by_employment_type": self.df["employment_type"].value_counts().to_dict(),
                "top_20_companies_by_jobs": self.df["company_name"].value_counts().head(20).to_dict(),
                "top_20_locations_by_jobs": self.df["location"].value_counts().head(20).to_dict(),
                "top_20_titles_by_count": self.df["job_title"].value_counts().head(20).to_dict(),
            }
            context_parts.append(f"COMPARISON DATA:\n{json.dumps(comparison_data, indent=2, default=str)}")

        # Sample of raw data (always include a small sample for context)
        sample_size = min(10, len(self.df))
        sample = self.df.sample(n=sample_size, random_state=42)[
            ["job_title", "company_name", "location", "hiring_status",
             "seniority_level", "employment_type", "industry", "job_function"]
        ].to_dict(orient="records")
        context_parts.append(f"SAMPLE RECORDS (random sample of {sample_size} jobs):\n{json.dumps(sample, indent=2, default=str)}")

        return "\n\n".join(context_parts)

    def chat(self, question, chat_history=None):
        """
        Answer a question about the job dataset using Groq (Llama 3).
        
        Args:
            question: The user's natural language question
            chat_history: Optional list of previous messages for context
            
        Returns:
            dict with 'answer', 'context_used', and 'status'
        """
        if not self.client:
            return {
                "answer": "⚠️ Groq API key is not configured. Please provide your API key to use the AI chat feature.",
                "status": "error",
                "context_used": "none",
            }

        # Extract relevant data context
        data_context = self._extract_relevant_data(question)

        # RAG Retrieval
        retrieved_context = ""
        if self.retriever:
            try:
                results = self.retriever.search(question, top_k=5)
                if results:
                    retrieved_context = "RETRIEVED RELEVANT JOB POSTINGS (Use these to answer specific queries):\n"
                    for i, r in enumerate(results, 1):
                        retrieved_context += f"{i}. {r['job_title']} at {r['company_name']} ({r['location']}) - {r['industry']}\n"
            except Exception as e:
                print(f"[LLM ERROR] Retrieval failed: {e}")

        # Build the system prompt
        system_prompt = f"""You are JobLens AI — an expert data analyst assistant for a LinkedIn job postings dataset.
You have access to a dataset of 31,000+ LinkedIn job postings.

DATASET SUMMARY (High-level stats):
Total Records: {self._dataset_summary['total_jobs']}
Top Industries: {', '.join(list(self._dataset_summary['top_industries'].keys()))}
Top Job Functions: {', '.join(list(self._dataset_summary['top_job_functions'].keys()))}
Top Locations: {', '.join(list(self._dataset_summary['top_10_locations'].keys()))}

{retrieved_context}

Your role:
1. Answer questions about the job market based ONLY on the data provided above.
2. If RETRIEVED RELEVANT JOB POSTINGS are provided, prioritize using them to answer the user's specific query. Give precise details from these postings.
3. Provide specific numbers, percentages, and data-driven insights.
4. Be concise but thorough — use bullet points and formatting for clarity.
5. If the user asks an off-topic question unrelated to jobs or the job market, reply with a simple, brief sentence (e.g., "I can only help with questions related to the job market."). Do NOT mention a "dataset" or generate a long explanation, and do NOT suggest follow-up questions.
6. Suggest follow-up questions the user might find interesting (only for on-topic questions)
7. Use markdown formatting for better readability (bold, lists, headers)
8. When providing statistics, always mention the data source context

IMPORTANT: Base your answers strictly on the context provided. Do not make up data."""

        # Build conversation with history
        user_message = f"DATASET CONTEXT:\n{data_context}\n\n"

        if chat_history:
            user_message += "PREVIOUS CONVERSATION:\n"
            for msg in chat_history[-6:]:  # Keep last 6 messages for context
                role = "User" if msg.get("role") == "user" else "Assistant"
                user_message += f"{role}: {msg.get('content', '')}\n\n"

        user_message += f"USER QUESTION: {question}"

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "status": "success",
                "context_used": "dataset_analysis",
            }
        except Exception as e:
            print("[LLM ERROR]:", repr(e))
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "PERMISSION" in error_msg.upper() or "INVALID" in error_msg.upper():
                return {
                    "answer": "⚠️ Invalid or expired API key. Please check your API key and try again.",
                    "status": "error",
                    "context_used": "none",
                }
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg.upper() or "QUOTA" in error_msg.upper():
                return {
                    "answer": "⏳ **Rate Limit Reached:** You have exceeded your Groq API rate limit. Please wait a moment and try again.",
                    "status": "error",
                    "context_used": "none",
                }
            return {
                "answer": f"❌ An error occurred while processing your question: {error_msg}",
                "status": "error",
                "context_used": "none",
            }
