import json
import logging
import time
import random
from typing import Dict, Any, List, Optional
from .groq_client import GroqClientManager
from ..jd_processing.models import JobDescription
from ..resume_processing.models import ProcessedResume, SectionType
from ..matching_engine.models import RequirementMatchResult, MatchClassification
from ..scoring.models import CandidateEvaluation, Recommendation

logger = logging.getLogger(__name__)

REC_SELECTION = "Recommended for Selection"
REC_MANUAL_REVIEW = "Manual Review Required"
REC_NOT_RECOMMENDED = "Not Recommended"

class GroqContextualEvaluator:
    """
    Evaluates candidate resumes against Job Descriptions using Groq LLMs.
    Uses targeted evidence snippets rather than full resumes, enforces strict call budgeting,
    and supports an optional conditional 2nd call for genuinely ambiguous items.
    """
    def __init__(self, client_manager: Optional[GroqClientManager] = None):
        self.client_manager = client_manager or GroqClientManager()
        self.call_count: int = 0

    @property
    def is_available(self) -> bool:
        return self.client_manager.is_available

    def reset_call_counter(self):
        """Resets the total LLM API call counter."""
        self.call_count = 0

    def get_call_stats(self, total_candidates: int) -> Dict[str, Any]:
        """Calculates total calls and average calls per candidate."""
        avg_calls = round(self.call_count / max(1, total_candidates), 2)
        return {
            "total_calls": self.call_count,
            "total_candidates": total_candidates,
            "average_calls_per_candidate": avg_calls
        }

    def evaluate_candidate(
        self,
        jd: JobDescription,
        resume: ProcessedResume,
        deterministic_matches: List[RequirementMatchResult],
        deterministic_eval: CandidateEvaluation
    ) -> Dict[str, Any]:
        """
        Combines deterministic hybrid matches with LLM contextual evaluation.
        Follows strict call budgeting:
        - Extraction anomalies: 0 calls (routed to manual review)
        - Obvious hard mismatches: 0 calls (direct reject)
        - Viable candidates: 1 batched call using targeted snippets
        - Genuinely ambiguous cases: max 1 additional targeted probe
        """
        if not self.is_available:
            return self._build_fallback_result(deterministic_eval, "Groq API key not configured. Using deterministic hybrid scoring.")

        # Budget Rule 1: Extraction anomalies or low text -> 0 calls, route to manual review
        if getattr(deterministic_eval, "is_low_confidence_extraction", False) or getattr(resume, "is_low_confidence", False):
            return self._build_fallback_result(
                deterministic_eval,
                "Extraction confidence is low. Skipped LLM call to save tokens; routed to manual review."
            )

        # Budget Rule 2: Obvious hard mismatches with clean extraction -> 0 calls, direct reject
        if (
            deterministic_eval.essential_coverage_pct < 45.0
            and deterministic_eval.overall_score < 40.0
            and getattr(resume, "verified_experience_years", 0.0) == 0.0
        ):
            return self._build_fallback_result(
                deterministic_eval,
                "Obvious mismatch on mandatory competencies with 0 relevant experience. Skipped LLM call."
            )

        client = self.client_manager.get_client()
        if client is None:
            return self._build_fallback_result(deterministic_eval, "Failed to initialize Groq client. Using deterministic hybrid scoring.")

        try:
            # Prepare compact, targeted evidence snippets (never full resume text)
            prompt_data = self._prepare_targeted_payload(jd, resume, deterministic_matches, deterministic_eval)
            system_prompt = (
                "You are an expert Senior Technical Recruiter and Hiring Committee Member. "
                "Evaluate the candidate's practical skill demonstration, project impact, and experience context "
                "based strictly on the provided evidence snippets against the target Job Description.\n\n"
                "CRITICAL RULES:\n"
                "1. STRICT JD RELEVANCE: Candidates with general technical experience (e.g., general IT, web development, "
                "scripting) who lack core mandatory competencies for this specific role MUST receive low scores (below 45) "
                "and 'Not Recommended'. General technical experience CANNOT compensate for missing mandatory requirements.\n"
                "2. MANDATORY VS PREFERRED: Prioritize mandatory requirements above all. Preferred skills add incremental value "
                "only when core competencies are satisfied.\n"
                "3. TRANSFERABLE SKILLS: Recognize transferable skills (e.g. Angular for React, PyTorch for TensorFlow) "
                "as transferable context rather than direct equivalents, and recommend 'Manual Review Required' if viable.\n"
                "4. PROJECT QUALITY & DEPTH: Evaluate projects based on BOTH implementation depth (architecture, complexity) "
                "and direct relevance to the JD requirements. Projects in unrelated stacks do not satisfy role requirements.\n"
                "5. STRICT EVIDENCE GROUNDING: Do NOT invent, assume, or hallucinate skills.\n"
                "6. NO KEYWORD STUFFING REWARD: Distinguish listed skills from demonstrated execution.\n"
                "7. UNCERTAINTY HANDLING: If any claim is contradictory, vague, or lacks verifiable context, "
                "list it explicitly in 'uncertain_areas'.\n"
                "8. Return strictly a valid JSON object matching the requested schema."
            )

            user_prompt = f"Evaluate this candidate against the requirements based on these verified evidence snippets:\n\n{json.dumps(prompt_data, indent=2)}"

            # Call #1: Primary Batched Contextual Evaluation (with Exponential Backoff)
            self.call_count += 1
            call_kwargs = {
                "model": self.client_manager.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 1500,
                "timeout": 15.0
            }
            response = self._call_groq_with_backoff(client, call_kwargs)
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)

            uncertain_areas = parsed.get("uncertain_areas", [])
            secondary_call_applied = False
            clarified_notes = ""

            # Call #2 (Conditional only): If genuine ambiguity exists in a borderline candidate
            if (
                uncertain_areas
                and len(uncertain_areas) >= 1
                and deterministic_eval.recommendation == Recommendation.MANUAL_REVIEW_REQUIRED
            ):
                secondary_call_applied = True
                clarified_notes = self._execute_targeted_ambiguity_probe(
                    client, resume, uncertain_areas
                )

            # Reconcile score between deterministic engine and LLM qualitative assessment
            llm_score = float(parsed.get("overall_score", deterministic_eval.overall_score))
            llm_score = max(0.0, min(100.0, round(llm_score, 1)))

            # Capture LLM suggested decision and explainable reasoning
            llm_suggested_rec = parsed.get("suggested_recommendation") or parsed.get("recommendation") or deterministic_eval.recommendation.value
            final_rec = deterministic_eval.recommendation.value
            exec_summary = parsed.get("executive_summary", deterministic_eval.decision_rationale)
            if clarified_notes:
                exec_summary += f" [Targeted Verification: {clarified_notes}]"

            return {
                "llm_applied": True,
                "model_used": self.client_manager.model,
                "recommendation": final_rec,
                "llm_suggested_recommendation": llm_suggested_rec,
                "overall_score": llm_score,
                "executive_summary": exec_summary,
                "strengths": parsed.get("strengths", []),
                "weaknesses": parsed.get("weaknesses", []),
                "project_and_internship_analysis": parsed.get("project_and_internship_analysis", "Analyzed projects and experience based on evidence snippets."),
                "uncertain_areas": uncertain_areas,
                "secondary_call_applied": secondary_call_applied,
                "error": None
            }

        except Exception as e:
            logger.warning(f"Groq API evaluation failed: {e}. Falling back to deterministic matching.")
            return self._build_fallback_result(deterministic_eval, f"Groq evaluation error: {str(e)}. Preserved hybrid engine results.")

    def _call_groq_with_backoff(
        self,
        client: Any,
        create_kwargs: Dict[str, Any],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0
    ) -> Any:
        """
        Executes a Groq API call with exponential backoff and jitter.
        Handles rate limits (429), server errors (500/502/503/504), and transient timeouts.
        Non-retriable auth errors fail fast without redundant delays.
        """
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return client.chat.completions.create(**create_kwargs)
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()

                # Fail fast on non-retriable auth issues
                if "authentication" in err_str or "unauthorized" in err_str or "invalid api key" in err_str:
                    logger.error(f"Non-retriable Groq auth error encountered: {e}")
                    raise e

                # Detect retriable conditions
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many requests" in err_str
                is_server_error = any(code in err_str for code in ["500", "502", "503", "504", "overloaded", "service unavailable"])
                is_timeout = "timeout" in err_str or "timed out" in err_str or "connection" in err_str

                if attempt < max_retries and (is_rate_limit or is_server_error or is_timeout):
                    # Exponential backoff with random jitter: delay = min(max_delay, base * 2^attempt) + jitter
                    backoff = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0.1, 0.4)
                    delay = round(backoff + jitter, 2)
                    logger.warning(
                        f"Groq API call encountered transient error: {e}. "
                        f"Retrying attempt {attempt + 1}/{max_retries} in {delay}s (Exponential Backoff)..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(f"Groq API call attempt {attempt + 1} failed: {e}")
                    raise e

        if last_exception:
            raise last_exception

    def _execute_targeted_ambiguity_probe(
        self,
        client: Any,
        resume: ProcessedResume,
        uncertain_areas: List[str]
    ) -> str:
        """
        Executes a targeted, minimal second LLM call strictly to resolve specific ambiguous points.
        Never sends whole resume; sends only the ambiguous questions and candidate's experience sentences.
        Uses exponential backoff for resilience against transient errors.
        """
        try:
            self.call_count += 1
            # Collect up to 4 sentences from experience or projects relevant to the ambiguity
            exp_sentences = [s.text for s in resume.get_sentences_by_section(SectionType.EXPERIENCE)[:4]]
            proj_sentences = [s.text for s in resume.get_sentences_by_section(SectionType.PROJECTS)[:3]]

            probe_payload = {
                "ambiguities_to_verify": uncertain_areas[:3],
                "candidate_evidence_sentences": exp_sentences + proj_sentences
            }

            probe_kwargs = {
                "model": self.client_manager.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior verifier. Determine if the evidence sentences confirm or refute the ambiguous items. Return JSON: {'clarification': '1-2 concise sentences', 'is_resolved': bool}"
                    },
                    {"role": "user", "content": json.dumps(probe_payload)}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.05,
                "max_tokens": 400,
                "timeout": 10.0
            }
            resp = self._call_groq_with_backoff(client, probe_kwargs)
            data = json.loads(resp.choices[0].message.content)
            return data.get("clarification", "Ambiguity probed against resume text.")
        except Exception as e:
            logger.info(f"Targeted ambiguity probe skipped or failed: {e}")
            return "Could not conclusively resolve ambiguity via secondary probe."

    def _prepare_targeted_payload(
        self,
        jd: JobDescription,
        resume: ProcessedResume,
        deterministic_matches: List[RequirementMatchResult],
        deterministic_eval: CandidateEvaluation
    ) -> Dict[str, Any]:
        """Packages only targeted evidence snippets (never full resume text) into structured input."""
        essential_evidence = []
        preferred_evidence = []

        for m in deterministic_matches:
            item = {
                "requirement": m.requirement_text,
                "classification": m.classification.value,
                "evidence": m.evidence_sentence or "None found",
                "depth": m.evidence_strength.value
            }
            if m.priority.lower() == "required":
                essential_evidence.append(item)
            else:
                preferred_evidence.append(item)

        # Extract top 3 project sentences
        proj_sentences = [s.text for s in resume.get_sentences_by_section(SectionType.PROJECTS)[:3]]

        return {
            "job_position": jd.title,
            "candidate_name": resume.candidate_name or "Candidate",
            "verified_experience_years": getattr(resume, "verified_experience_years", 0.0),
            "timeline_summary": getattr(resume, "timeline_summary", "Unverified"),
            "essential_requirements_evidence": essential_evidence,
            "preferred_requirements_evidence": preferred_evidence,
            "project_portfolio_snippets": proj_sentences,
            "deterministic_baseline_score": deterministic_eval.overall_score,
            "essential_coverage_pct": deterministic_eval.essential_coverage_pct,
            "expected_output_json_schema": {
                "overall_score": "float from 0.0 to 100.0",
                "suggested_recommendation": "Recommended for Selection | Manual Review Required | Not Recommended",
                "executive_summary": "2-3 sentences explaining the hiring recommendation",
                "strengths": ["list of candidate strengths backed by verified evidence"],
                "weaknesses": ["list of genuine gaps or missing skills"],
                "project_and_internship_analysis": "Assessment of hands-on project depth and real execution scale",
                "uncertain_areas": ["list of contradictory, vague, or unverifiable claims"]
            }
        }

    def _build_fallback_result(self, deterministic_eval: CandidateEvaluation, note: str) -> Dict[str, Any]:
        """Provides a safe fallback response using deterministic evaluation when LLM is unavailable or skipped."""
        return {
            "llm_applied": False,
            "model_used": None,
            "recommendation": deterministic_eval.recommendation.value,
            "overall_score": deterministic_eval.overall_score,
            "executive_summary": deterministic_eval.decision_rationale,
            "strengths": [
                f"Matched {deterministic_eval.matched_essential_count}/{deterministic_eval.total_essential_count} essential competencies."
            ],
            "weaknesses": [
                f"Missing {deterministic_eval.total_essential_count - deterministic_eval.matched_essential_count} essential items."
            ] if deterministic_eval.total_essential_count > deterministic_eval.matched_essential_count else [],
            "project_and_internship_analysis": f"Verified experience: {deterministic_eval.verified_tenure_years} yrs. Practical evidence score: {deterministic_eval.practical_evidence_score:.1f}%.",
            "uncertain_areas": [],
            "secondary_call_applied": False,
            "error": note
        }
