import json
import pandas as pd
from typing import List, Dict, Any, Optional
from ..resume_processing.models import ProcessedResume
from ..matching_engine.models import RequirementMatchResult, MatchClassification
from ..scoring.models import CandidateEvaluation
from ..jd_processing.models import JobDescription

class ScreeningReportGenerator:
    """
    Generates structured individual candidate reports, multi-candidate comparison tables,
    and exportable JSON / CSV datasets.
    """
    @staticmethod
    def generate_candidate_report(
        resume: ProcessedResume,
        evaluation: CandidateEvaluation,
        match_results: List[RequirementMatchResult],
        jd: JobDescription,
        llm_evaluation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Constructs an explainable, comprehensive report dictionary for a single candidate."""
        matched_reqs = [
            r.to_dict() for r in match_results
            if r.priority.lower() == "required" and r.classification != MatchClassification.NOT_FOUND
        ]
        missing_reqs = [
            r.to_dict() for r in match_results
            if r.priority.lower() == "required" and r.classification == MatchClassification.NOT_FOUND
        ]
        matched_prefs = [
            r.to_dict() for r in match_results
            if r.priority.lower() == "preferred" and r.classification != MatchClassification.NOT_FOUND
        ]
        related_skills = [
            r.to_dict() for r in match_results
            if r.classification == MatchClassification.RELATED_SKILL_NOT_EQUIVALENT
        ]

        eval_summary = evaluation.to_dict()

        # If Groq LLM evaluation is present and active, attach qualitative details
        if llm_evaluation and llm_evaluation.get("llm_applied"):
            eval_summary["llm_score"] = llm_evaluation.get("overall_score")
            eval_summary["llm_suggested_recommendation"] = llm_evaluation.get("llm_suggested_recommendation", llm_evaluation.get("recommendation"))
            eval_summary["llm_strengths"] = llm_evaluation.get("strengths", [])
            eval_summary["llm_weaknesses"] = llm_evaluation.get("weaknesses", [])
            eval_summary["project_and_internship_analysis"] = llm_evaluation.get("project_and_internship_analysis", "")
            eval_summary["uncertain_areas"] = llm_evaluation.get("uncertain_areas", [])
            eval_summary["llm_model"] = llm_evaluation.get("model_used")
            eval_summary["llm_synthesis"] = llm_evaluation.get("executive_summary")

        report = {
            "candidate_name": resume.candidate_name or "Unknown Candidate",
            "filename": resume.filename,
            "processing_status": {
                "success": resume.extraction_success,
                "is_scanned": resume.is_scanned,
                "ocr_applied": resume.ocr_applied,
                "page_count": resume.page_count,
                "extraction_confidence": getattr(resume, "extraction_confidence", 1.0),
                "is_low_confidence": getattr(resume, "is_low_confidence", False),
                "confidence_reasons": getattr(resume, "confidence_reasons", []),
                "warnings": resume.extraction_warnings
            },
            "verified_experience": {
                "verified_years": getattr(resume, "verified_experience_years", 0.0),
                "employment_periods": getattr(resume, "employment_periods", []),
                "timeline_summary": getattr(resume, "timeline_summary", "Unverified")
            },
            "job_position": jd.title,
            "evaluation_summary": eval_summary,
            "llm_evaluation": llm_evaluation,
            "skills_analysis": {
                "matched_required_count": len(matched_reqs),
                "total_required_count": evaluation.total_required_count,
                "matched_preferred_count": len(matched_prefs),
                "total_preferred_count": evaluation.total_preferred_count,
                "related_skills_count": len(related_skills)
            },
            "detailed_requirements": [r.to_dict() for r in match_results],
            "evidence_breakdown": [
                {
                    "requirement": r.requirement_text,
                    "priority": r.priority,
                    "classification": r.classification.value,
                    "evidence_sentence": r.evidence_sentence,
                    "section": r.resume_section.value if r.resume_section else "N/A",
                    "strength": r.evidence_strength.value,
                    "method": r.matching_method,
                    "explanation": r.explanation
                }
                for r in match_results if r.evidence_sentence
            ]
        }
        return report

    @staticmethod
    def generate_comparison_dataframe(candidate_reports: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts screening reports into a comparative overview table."""
        rows = []
        for cr in candidate_reports:
            ev = cr["evaluation_summary"]
            proc = cr["processing_status"]
            exp = cr.get("verified_experience", {})
            rows.append({
                "Candidate Name": cr["candidate_name"],
                "Filename": cr["filename"],
                "Overall Score": ev["overall_score"],
                "Groq LLM Score (30%)": ev.get("llm_score", "N/A"),
                "LLM Suggestion": ev.get("llm_suggested_recommendation", "N/A"),
                "Essential Coverage (%)": ev.get("essential_coverage_pct", ev.get("required_coverage_pct", 0.0)),
                "Practical Depth (%)": ev.get("practical_evidence_score", ev.get("evidence_quality_score", 0.0)),
                "Project Depth (%)": ev.get("project_depth_score", 0.0),
                "Preferred Coverage (%)": ev.get("preferred_coverage_pct", 0.0),
                "Verified Tenure (Yrs)": exp.get("verified_years", ev.get("verified_tenure_years", 0.0)),
                "Recommendation": ev["recommendation"],
                "Req Matched": f"{ev['matched_required_count']}/{ev['total_required_count']}",
                "Extraction Confidence": f"{int(proc.get('extraction_confidence', 1.0) * 100)}%"
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="Overall Score", ascending=False)
        return df

    @staticmethod
    def export_reports_to_json(candidate_reports: List[Dict[str, Any]], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(candidate_reports, f, indent=2)

    @staticmethod
    def export_comparison_to_csv(df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False)
