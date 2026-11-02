import json
import pandas as pd
from typing import List, Dict, Any, Optional
from ..resume_processing.models import ProcessedResume
from ..matching_engine.models import RequirementMatchResult, MatchClassification, EvidenceStrength
from ..scoring.models import CandidateEvaluation
from ..jd_processing.models import JobDescription


DEMONSTRATED_EVIDENCE = frozenset({
    EvidenceStrength.STRONG_IMPLEMENTATION,
    EvidenceStrength.EXPERIENCE_EVIDENCE,
    EvidenceStrength.PROJECT_EVIDENCE,
})

CONFIRMED_CLASSIFICATIONS = frozenset({
    MatchClassification.EXACT_MATCH,
    MatchClassification.STRONG_SEMANTIC_MATCH,
    MatchClassification.FUZZY_MATCH,
})


class ScreeningReportGenerator:
    """
    Generates structured, explainable candidate reports.

    Reports focus on:
    - What the candidate actually demonstrated (evidence-grounded)
    - Which JD requirements were satisfied and how strongly
    - Genuine gaps that matter (not just absent keywords)
    - Clear, human-readable recommendation rationale
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

        # ── Requirement breakdown ────────────────────────────────────────────
        matched_reqs = [
            r.to_dict() for r in match_results
            if r.priority.lower() in ("mandatory", "required")
            and r.classification != MatchClassification.NOT_FOUND
        ]
        missing_reqs = [
            r.to_dict() for r in match_results
            if r.priority.lower() in ("mandatory", "required")
            and r.classification == MatchClassification.NOT_FOUND
        ]
        matched_prefs = [
            r.to_dict() for r in match_results
            if r.priority.lower() == "preferred"
            and r.classification != MatchClassification.NOT_FOUND
        ]
        related_skills = [
            r.to_dict() for r in match_results
            if r.classification == MatchClassification.RELATED_SKILL_NOT_EQUIVALENT
        ]

        # ── Evidence breakdown: what evidence was actually found ─────────────
        evidence_breakdown = []
        for r in match_results:
            if not r.evidence_sentence:
                continue
            evidence_breakdown.append({
                "requirement": r.requirement_text,
                "priority": r.priority,
                "classification": r.classification.value,
                "evidence_sentence": r.evidence_sentence,
                "section": r.resume_section.value if r.resume_section else "N/A",
                "strength": r.evidence_strength.value,
                "method": r.matching_method,
                "explanation": r.explanation,
                "weight": round(r.weight, 4),
                "fulfillment_ratio": round(r.fulfillment_ratio, 3)
            })

        # ── Demonstrated vs listed capabilities ─────────────────────────────
        demonstrated = [
            {
                "requirement": r.requirement_text,
                "evidence": r.evidence_sentence,
                "section": r.resume_section.value if r.resume_section else "N/A",
                "strength": r.evidence_strength.value
            }
            for r in match_results
            if r.classification in CONFIRMED_CLASSIFICATIONS
            and r.evidence_strength in DEMONSTRATED_EVIDENCE
        ]
        listed_only = [
            {
                "requirement": r.requirement_text,
                "section": r.resume_section.value if r.resume_section else "N/A"
            }
            for r in match_results
            if r.classification in CONFIRMED_CLASSIFICATIONS
            and r.evidence_strength not in DEMONSTRATED_EVIDENCE
            and r.evidence_strength != EvidenceStrength.NONE
        ]

        # ── Build evaluation summary ─────────────────────────────────────────
        eval_summary = evaluation.to_dict()

        if llm_evaluation and llm_evaluation.get("llm_applied"):
            eval_summary["llm_score"] = llm_evaluation.get("overall_score")
            eval_summary["llm_suggested_recommendation"] = llm_evaluation.get(
                "llm_suggested_recommendation", llm_evaluation.get("recommendation")
            )
            eval_summary["llm_strengths"]    = llm_evaluation.get("strengths", [])
            eval_summary["llm_weaknesses"]   = llm_evaluation.get("weaknesses", [])
            eval_summary["project_and_internship_analysis"] = llm_evaluation.get("project_and_internship_analysis", "")
            eval_summary["uncertain_areas"]  = llm_evaluation.get("uncertain_areas", [])
            eval_summary["llm_model"]        = llm_evaluation.get("model_used")
            eval_summary["llm_synthesis"]    = llm_evaluation.get("executive_summary")
            eval_summary["requirement_assessments"]     = llm_evaluation.get("requirement_assessments", [])
            eval_summary["hallucinated_quotes_detected"] = llm_evaluation.get("hallucinated_quotes_detected", 0)

        # ── Assemble the report ──────────────────────────────────────────────
        report = {
            "candidate_name": resume.candidate_name or "Unknown Candidate",
            "filename": resume.filename,
            "job_position": jd.title,

            # Processing metadata
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

            # Verified experience
            "verified_experience": {
                "verified_years": getattr(resume, "verified_experience_years", 0.0),
                "employment_periods": getattr(resume, "employment_periods", []),
                "timeline_summary": getattr(resume, "timeline_summary", "Unverified")
            },

            # Evaluation outcome
            "evaluation_summary": eval_summary,

            # LLM evaluation (raw)
            "llm_evaluation": llm_evaluation,

            # Skills analysis counts
            "skills_analysis": {
                "matched_required_count": len(matched_reqs),
                "total_required_count": evaluation.total_required_count,
                "matched_preferred_count": len(matched_prefs),
                "total_preferred_count": evaluation.total_preferred_count,
                "related_skills_count": len(related_skills)
            },

            # Human-readable capability summary
            "capability_summary": {
                "demonstrated_capabilities": demonstrated,
                "listed_only_capabilities": listed_only,
                "transferable_skills": evaluation.transferable_skills,
                "satisfied_requirements": matched_reqs,
                "missing_requirements": missing_reqs,
                "satisfied_preferred": matched_prefs,
                "related_but_not_equivalent": related_skills
            },

            # Full requirement details
            "detailed_requirements": [r.to_dict() for r in match_results],

            # Evidence breakdown (only where evidence was found)
            "evidence_breakdown": evidence_breakdown
        }
        return report

    @staticmethod
    def generate_comparison_dataframe(candidate_reports: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts screening reports into a comparative overview table."""
        rows = []
        for cr in candidate_reports:
            ev   = cr["evaluation_summary"]
            proc = cr["processing_status"]
            exp  = cr.get("verified_experience", {})
            rows.append({
                "Candidate Name":        cr["candidate_name"],
                "Filename":              cr["filename"],
                "Overall Score":         ev["overall_score"],
                "Essential Coverage (%)": ev.get("essential_coverage_pct", ev.get("required_coverage_pct", 0.0)),
                "Practical Depth (%)":   ev.get("practical_evidence_score", ev.get("evidence_quality_score", 0.0)),
                "Project Depth (%)":     ev.get("project_depth_score", 0.0),
                "Preferred Coverage (%)": ev.get("preferred_coverage_pct", 0.0),
                "Verified Tenure (Yrs)": exp.get("verified_years", ev.get("verified_tenure_years", 0.0)),
                "Recommendation":        ev["recommendation"],
                "Req Matched":           f"{ev['matched_required_count']}/{ev['total_required_count']}",
                "Extraction Confidence": f"{int(proc.get('extraction_confidence', 1.0) * 100)}%",
                "AI Validated Assessment": ev.get("llm_score", "N/A")
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
