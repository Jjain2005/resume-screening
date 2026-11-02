import json
import logging
import time
import random
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from .groq_client import GroqClientManager
from ..jd_processing.models import JobDescription, JDEvaluationStrategy
from ..resume_processing.models import ProcessedResume, SectionType
from ..matching_engine.models import RequirementMatchResult, MatchClassification
from ..scoring.models import CandidateEvaluation, Recommendation

logger = logging.getLogger(__name__)

REC_SELECTION = "Recommended for Selection"
REC_MANUAL_REVIEW = "Manual Review Required"
REC_NOT_RECOMMENDED = "Not Recommended"


class APICallTracker:
    """Thread-safe tracker for LLM API calls, cache hits, retries, and evaluation outcomes."""
    def __init__(self):
        self._lock = threading.Lock()
        self.jd_analysis_calls: int = 0
        self.resume_eval_calls: int = 0
        self.successful_evaluations: int = 0
        self.failed_evaluations: int = 0
        self.retry_count: int = 0
        self.cache_hits: int = 0

    @property
    def total_api_calls(self) -> int:
        with self._lock:
            return self.jd_analysis_calls + self.resume_eval_calls

    def record_jd_analysis_call(self):
        with self._lock:
            self.jd_analysis_calls += 1

    def record_resume_eval_call(self):
        with self._lock:
            self.resume_eval_calls += 1

    def record_success(self):
        with self._lock:
            self.successful_evaluations += 1

    def record_failure(self):
        with self._lock:
            self.failed_evaluations += 1

    def record_retry(self):
        with self._lock:
            self.retry_count += 1

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def reset(self):
        with self._lock:
            self.jd_analysis_calls = 0
            self.resume_eval_calls = 0
            self.successful_evaluations = 0
            self.failed_evaluations = 0
            self.retry_count = 0
            self.cache_hits = 0

    def get_stats(self, total_candidates: int = 0) -> Dict[str, Any]:
        with self._lock:
            total_calls = self.jd_analysis_calls + self.resume_eval_calls
            avg_per_cand = round(self.resume_eval_calls / max(1, total_candidates), 2) if total_candidates > 0 else 0.0
            return {
                "total_calls": total_calls,
                "jd_analysis_calls": self.jd_analysis_calls,
                "resume_eval_calls": self.resume_eval_calls,
                "successful_evaluations": self.successful_evaluations,
                "failed_evaluations": self.failed_evaluations,
                "retry_count": self.retry_count,
                "cache_hits": self.cache_hits,
                "total_candidates": total_candidates,
                "average_calls_per_candidate": avg_per_cand
            }


class ContentCache:
    """
    Content-hash-based caching for JD analysis and individual candidate evaluations.
    Invalidates cached results whenever JD, resume content, or evaluation strategy changes.
    Never silently substitutes a cached or failed result belonging to another candidate.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._jd_cache: Dict[str, Any] = {}
        self._candidate_cache: Dict[str, Any] = {}

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()

    def get_jd_strategy(self, jd_text: str) -> Optional[Any]:
        jd_hash = self.hash_text(jd_text)
        with self._lock:
            return self._jd_cache.get(jd_hash)

    def set_jd_strategy(self, jd_text: str, strategy: Any):
        jd_hash = self.hash_text(jd_text)
        with self._lock:
            self._jd_cache[jd_hash] = strategy

    def make_candidate_key(self, jd_hash: str, strategy_hash: str, resume_hash: str, filename: str) -> str:
        return f"{jd_hash}:{strategy_hash}:{resume_hash}:{filename}"

    def get_candidate_eval(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cached = self._candidate_cache.get(key)
            if cached:
                return dict(cached)
            return None

    def set_candidate_eval(self, key: str, eval_result: Dict[str, Any]):
        with self._lock:
            self._candidate_cache[key] = dict(eval_result)

    def clear(self):
        with self._lock:
            self._jd_cache.clear()
            self._candidate_cache.clear()


class GroqContextualEvaluator:
    """
    Evaluates candidate resumes against Job Descriptions using Groq LLMs.

    One-to-many architecture:
    1. JD Analysis: Analyzes the JD once to produce a structured evaluation strategy
       (domain, essential capabilities, relevant skills, expected practical knowledge, desirable qualifications).
    2. Independent Resume Evaluation: Exactly one independent LLM call per candidate receiving the
       shared JD strategy and that candidate's complete extracted resume content.
    3. Python Validation: Validates evidence quotes against resume text, calculates scores and recommendations.
    4. Reliability: Thread-safe tracking of JD calls, resume calls, retries, cache hits, and outcomes.
    """

    def __init__(self, client_manager: Optional[GroqClientManager] = None):
        self.client_manager = client_manager or GroqClientManager()
        self._tracker = APICallTracker()
        self._cache = ContentCache()

    @property
    def tracker(self) -> APICallTracker:
        if not hasattr(self, "_tracker") or self._tracker is None:
            self._tracker = APICallTracker()
        return self._tracker

    @tracker.setter
    def tracker(self, val: APICallTracker):
        self._tracker = val

    @property
    def cache(self) -> ContentCache:
        if not hasattr(self, "_cache") or self._cache is None:
            self._cache = ContentCache()
        return self._cache

    @cache.setter
    def cache(self, val: ContentCache):
        self._cache = val

    @property
    def is_available(self) -> bool:
        return self.client_manager.is_available

    @property
    def call_count(self) -> int:
        return self.tracker.total_api_calls

    @call_count.setter
    def call_count(self, val: int):
        with self.tracker._lock:
            self.tracker.resume_eval_calls = val

    def reset_call_counter(self):
        """Resets the total LLM API call counter and tracker metrics."""
        self.tracker.reset()

    def get_call_stats(self, total_candidates: int = 0) -> Dict[str, Any]:
        """Calculates total calls and metrics per candidate."""
        return self.tracker.get_stats(total_candidates)

    def evaluate_candidate(
        self,
        jd: JobDescription,
        resume: ProcessedResume,
        deterministic_matches: List[RequirementMatchResult],
        deterministic_eval: CandidateEvaluation,
        shared_strategy: Optional[JDEvaluationStrategy] = None
    ) -> Dict[str, Any]:
        """
        Performs an independent LLM evaluation of a single candidate against the shared JD strategy.

        - Each call receives the shared JD evaluation strategy and the candidate's complete content.
        - Checks content-hash cache first.
        - Validates evidence quotes against resume text in Python.
        - Calculates final scores and recommendations consistently in Python.
        """
        if not self.is_available:
            return self._build_fallback_result(
                deterministic_eval,
                "Groq API key not configured. Using deterministic hybrid scoring."
            )

        # Budget Rule 1: Low-confidence extraction → 0 calls, route to manual review
        if getattr(deterministic_eval, "is_low_confidence_extraction", False) or getattr(resume, "is_low_confidence", False):
            return self._build_fallback_result(
                deterministic_eval,
                "Extraction confidence is low. Skipped LLM call; routed to manual review."
            )

        # Budget Rule 2: Obvious hard mismatches → 0 calls, direct reject
        if (
            deterministic_eval.essential_coverage_pct < 40.0
            and deterministic_eval.overall_score < 35.0
            and getattr(resume, "verified_experience_years", 0.0) == 0.0
        ):
            return self._build_fallback_result(
                deterministic_eval,
                "Obvious mismatch on mandatory competencies with 0 relevant experience. Skipped LLM call."
            )

        client = self.client_manager.get_client()
        if client is None:
            return self._build_fallback_result(
                deterministic_eval,
                "Failed to initialize Groq client. Using deterministic hybrid scoring."
            )

        # Content-hash-based cache check
        strategy = shared_strategy or getattr(jd, "evaluation_strategy", None)
        jd_text = jd.raw_text or ""
        jd_hash = ContentCache.hash_text(jd_text)
        strategy_hash = getattr(strategy, "strategy_hash", "") or ContentCache.hash_text(
            json.dumps(getattr(strategy, "to_dict", lambda: {})(), sort_keys=True)
        )
        resume_text = (getattr(resume, "cleaned_text", "") or getattr(resume, "raw_text", "")).strip()
        resume_hash = ContentCache.hash_text(resume_text)
        candidate_key = self.cache.make_candidate_key(
            jd_hash=jd_hash,
            strategy_hash=strategy_hash,
            resume_hash=resume_hash,
            filename=resume.filename or "unknown"
        )

        cached_res = self.cache.get_candidate_eval(candidate_key)
        if cached_res is not None:
            self.tracker.record_cache_hit()
            res = dict(cached_res)
            res["from_cache"] = True
            return res

        try:
            payload = self._prepare_holistic_payload(jd, resume, deterministic_matches, deterministic_eval, strategy)

            system_prompt = (
                "You are an experienced Senior Technical Recruiter evaluating a candidate for a job opening.\n\n"
                "Your role:\n"
                "1. Review the shared Job Description evaluation strategy (domain, essential capabilities, relevant skills, "
                "expected practical knowledge, desirable qualifications).\n"
                "2. Understand the candidate's actual profile (skills, projects, experience, practical implementations). "
                "A skill demonstrated in a project or experience is genuine evidence even if not listed separately.\n"
                "3. Assess technical relevance, domain knowledge, skills, projects, practical implementations, experience, and overall suitability.\n"
                "4. Identify:\n"
                "   - direct_evidence: skills/capabilities directly demonstrated in experience/projects\n"
                "   - reasonable_inferences: capabilities that can be reasonably inferred from demonstrated tech stack\n"
                "   - genuine_gaps: critical qualifications genuinely missing\n"
                "   - unknown_information: behavioural or unverifiable qualities absent from resume (treated as UNKNOWN, never as gaps)\n\n"
                "CRITICAL RULES:\n"
                "- For each requirement, provide status: VERIFIED (clearly demonstrated), PARTIAL (some evidence), "
                "TRANSFERABLE (related but not equivalent), or UNSUPPORTED (no evidence found).\n"
                "- evidence_quote MUST be an exact sentence from the candidate profile provided. "
                "Do NOT paraphrase, fabricate, or combine sentences. All quotes are verified by Python.\n"
                "- Generic soft skills (communication, teamwork, etc.) absent from a resume are UNKNOWN, not gaps.\n"
                "- Recognize equivalent technologies and related capabilities — do not penalise for terminology differences.\n"
                "- Return strictly valid JSON matching the schema."
            )

            user_prompt = (
                f"Evaluate this candidate for the position and provide your assessment:\n\n"
                f"{json.dumps(payload)}"
            )

            self.tracker.record_resume_eval_call()
            call_kwargs = {
                "model": self.client_manager.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 1800,
                "timeout": 25.0
            }
            response = self._call_groq_with_backoff(client, call_kwargs)
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)

            uncertain_areas = parsed.get("uncertain_areas", [])
            secondary_call_applied = False
            clarified_notes = ""

            # Conditional second call: probe genuine ambiguities in borderline candidates
            if (
                uncertain_areas
                and len(uncertain_areas) >= 1
                and deterministic_eval.recommendation == Recommendation.MANUAL_REVIEW_REQUIRED
            ):
                secondary_call_applied = True
                clarified_notes = self._execute_targeted_ambiguity_probe(
                    client, resume, uncertain_areas
                )

            # ── Python validation of LLM evidence quotes ──────────────────────
            raw_resume_text = (getattr(resume, "cleaned_text", "") or getattr(resume, "raw_text", "")).lower()
            all_resume_sentences_lower = [s.text.lower() for s in getattr(resume, "all_sentences", [])]
            hallucinated_count = 0
            validated_assessments = []

            req_weight_map = {r.id: r.weight for r in jd.requirements}
            total_weight = sum(req_weight_map.values()) or 1.0

            status_credits = {
                "VERIFIED": 1.0,
                "PARTIAL": 0.50,
                "TRANSFERABLE": 0.35,
                "UNSUPPORTED": 0.0
            }

            earned_score = 0.0
            for item in parsed.get("requirement_assessments", []):
                rid    = item.get("requirement_id") or item.get("id")
                status = (item.get("status") or "UNSUPPORTED").upper()
                quote  = (item.get("evidence_quote") or "").strip()

                quote_verified = False
                if quote:
                    q_lower = quote.lower()
                    if q_lower in raw_resume_text or any(q_lower in s for s in all_resume_sentences_lower):
                        quote_verified = True
                    else:
                        words = [w for w in q_lower.split() if len(w) > 3]
                        if words and sum(1 for w in words if w in raw_resume_text) >= (len(words) * 0.7):
                            quote_verified = True

                if not quote_verified and quote:
                    hallucinated_count += 1
                    status = "UNSUPPORTED"

                validated_assessments.append({
                    "requirement_id": rid,
                    "status": status,
                    "evidence_quote": quote,
                    "is_quote_verified": quote_verified,
                    "inference_type": item.get("inference_type", "DIRECT_EVIDENCE"),
                    "analysis": item.get("analysis", "")
                })

                w = req_weight_map.get(rid, 1.0 / max(1, len(jd.requirements)))
                earned_score += status_credits.get(status, 0.0) * w

            # Deterministic Python score from validated LLM assessments
            if validated_assessments and total_weight > 0:
                python_validated_llm_score = (earned_score / total_weight) * 100.0
                if hallucinated_count > 0:
                    python_validated_llm_score = max(0.0, python_validated_llm_score - (hallucinated_count * 5.0))
                python_validated_llm_score = round(min(100.0, max(0.0, python_validated_llm_score)), 1)
            elif "overall_score" in parsed:
                python_validated_llm_score = float(parsed["overall_score"])
            else:
                python_validated_llm_score = deterministic_eval.overall_score

            llm_suggested_rec = (
                parsed.get("suggested_recommendation")
                or parsed.get("recommendation")
                or deterministic_eval.recommendation.value
            )
            exec_summary = parsed.get("executive_summary", deterministic_eval.decision_rationale)
            if clarified_notes:
                exec_summary += f" [Targeted Verification: {clarified_notes}]"

            skills_assessment = parsed.get("skills_assessment", {})
            if not isinstance(skills_assessment, dict):
                skills_assessment = {}

            result = {
                "llm_applied": True,
                "model_used": self.client_manager.model,
                "recommendation": deterministic_eval.recommendation.value,
                "llm_suggested_recommendation": llm_suggested_rec,
                "overall_score": python_validated_llm_score,
                "requirement_assessments": validated_assessments,
                "hallucinated_quotes_detected": hallucinated_count,
                "executive_summary": exec_summary,
                "technical_relevance_summary": parsed.get("technical_relevance_summary", ""),
                "domain_fit": parsed.get("domain_fit", ""),
                "skills_assessment": {
                    "direct_evidence": skills_assessment.get("direct_evidence", []),
                    "reasonable_inferences": skills_assessment.get("reasonable_inferences", []),
                    "genuine_gaps": skills_assessment.get("genuine_gaps", []),
                    "unknown_information": skills_assessment.get("unknown_information", [])
                },
                "strengths": parsed.get("strengths", []),
                "weaknesses": parsed.get("weaknesses", []),
                "project_and_internship_analysis": parsed.get(
                    "project_and_internship_analysis",
                    "Analysed projects and experience based on evidence snippets."
                ),
                "uncertain_areas": uncertain_areas,
                "secondary_call_applied": secondary_call_applied,
                "error": None
            }

            self.tracker.record_success()
            self.cache.set_candidate_eval(candidate_key, result)
            return result

        except Exception as e:
            logger.warning(f"Groq API evaluation failed: {e}. Falling back to deterministic matching.")
            self.tracker.record_failure()
            return self._build_fallback_result(
                deterministic_eval,
                f"Groq evaluation error: {str(e)}. Preserved hybrid engine results."
            )

    def evaluate_candidates_batch(
        self,
        jd: JobDescription,
        candidates: List[Tuple[ProcessedResume, List[RequirementMatchResult], CandidateEvaluation]],
        max_workers: int = 2,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Tuple[ProcessedResume, CandidateEvaluation, Optional[Dict[str, Any]]]]:
        """
        Evaluates multiple candidates with controlled concurrency.
        Each candidate receives an independent evaluation call with the shared JD evaluation strategy.
        Failure in one candidate's evaluation never blocks other candidates.
        """
        import concurrent.futures
        from ..scoring.evaluator import CandidateEvaluator

        evaluator = CandidateEvaluator()
        shared_strategy = getattr(jd, "evaluation_strategy", None)
        results: List[Optional[Tuple[ProcessedResume, CandidateEvaluation, Optional[Dict[str, Any]]]]] = [None] * len(candidates)

        def _evaluate_single(idx: int, cand_tuple: Tuple[ProcessedResume, List[RequirementMatchResult], CandidateEvaluation]):
            resume, matches, det_eval = cand_tuple
            try:
                llm_eval = self.evaluate_candidate(
                    jd=jd,
                    resume=resume,
                    deterministic_matches=matches,
                    deterministic_eval=det_eval,
                    shared_strategy=shared_strategy
                )
                final_eval = det_eval
                if llm_eval and llm_eval.get("llm_applied"):
                    # Python calculates all scores and decisions using one consistent formula.
                    # Groq provides qualitative synthesis and evidence interpretation.
                    final_eval = evaluator.evaluate(
                        matches,
                        resume=resume,
                        jd=jd,
                        llm_score=None,
                        llm_recommendation=llm_eval.get("llm_suggested_recommendation"),
                        llm_synthesis=llm_eval.get("executive_summary")
                    )
                return idx, (resume, final_eval, llm_eval)
            except Exception as e:
                logger.error(f"Error evaluating candidate {resume.filename}: {e}")
                fallback = self._build_fallback_result(det_eval, f"Evaluation error: {str(e)}")
                return idx, (resume, det_eval, fallback)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_evaluate_single, i, cand)
                for i, cand in enumerate(candidates)
            ]
            completed_count = 0
            for fut in concurrent.futures.as_completed(futures):
                idx, res_tuple = fut.result()
                results[idx] = res_tuple
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, len(candidates), res_tuple[0].filename)
                    except Exception:
                        pass

        return [r for r in results if r is not None]

    def _call_groq_with_backoff(
        self,
        client: Any,
        create_kwargs: Dict[str, Any],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0
    ) -> Any:
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return client.chat.completions.create(**create_kwargs)
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()

                if "authentication" in err_str or "unauthorized" in err_str or "invalid api key" in err_str:
                    logger.error(f"Non-retriable Groq auth error: {e}")
                    raise e

                is_rate_limit   = "429" in err_str or "rate limit" in err_str or "too many requests" in err_str
                is_server_error = any(c in err_str for c in ["500", "502", "503", "504", "overloaded", "service unavailable"])
                is_timeout      = "timeout" in err_str or "timed out" in err_str or "connection" in err_str

                if attempt < max_retries and (is_rate_limit or is_server_error or is_timeout):
                    self.tracker.record_retry()
                    backoff = min(max_delay, base_delay * (2 ** attempt))
                    jitter  = random.uniform(0.1, 0.4)
                    delay   = round(backoff + jitter, 2)
                    logger.warning(
                        f"Groq API transient error: {e}. "
                        f"Retry {attempt + 1}/{max_retries} in {delay}s..."
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
        try:
            self.tracker.record_resume_eval_call()
            exp_sentences  = [s.text for s in resume.get_sentences_by_section(SectionType.EXPERIENCE)[:4]]
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
                        "content": (
                            "You are a senior verifier. Determine if the evidence sentences confirm or "
                            "refute the ambiguous items. Return JSON: "
                            "{'clarification': '1-2 concise sentences', 'is_resolved': bool}"
                        )
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

    def _prepare_holistic_payload(
        self,
        jd: JobDescription,
        resume: ProcessedResume,
        deterministic_matches: List[RequirementMatchResult],
        deterministic_eval: CandidateEvaluation,
        strategy: Optional[JDEvaluationStrategy] = None
    ) -> Dict[str, Any]:
        """Packages a compact, holistic payload for the LLM including the shared evaluation strategy."""
        strat = strategy or getattr(jd, "evaluation_strategy", None)
        strat_dict = strat.to_dict() if strat and hasattr(strat, "to_dict") else {}

        # Build compact requirement list
        req_items = []
        for m in deterministic_matches:
            ev = (m.evidence_sentence[:120] + "...") if m.evidence_sentence and len(m.evidence_sentence) > 120 else (m.evidence_sentence or "")
            req_items.append({
                "requirement_id": m.requirement_id,
                "requirement_text": m.requirement_text,
                "priority": m.priority,
                "weight": round(m.weight, 2),
                "deterministic_match": m.classification.value,
                "evidence_snippet": ev
            })

        # Compact resume content by section for holistic understanding
        skills_sentences  = [s.text[:120] for s in resume.get_sentences_by_section(SectionType.SKILLS)[:8]]
        exp_sentences     = [s.text[:180] for s in resume.get_sentences_by_section(SectionType.EXPERIENCE)[:5]]
        proj_sentences    = [s.text[:180] for s in resume.get_sentences_by_section(SectionType.PROJECTS)[:5]]
        edu_sentences     = [s.text[:100] for s in resume.get_sentences_by_section(SectionType.EDUCATION)[:2]]
        summary_sentences = [s.text[:120] for s in resume.get_sentences_by_section(SectionType.SUMMARY)[:2]]

        if not any([skills_sentences, exp_sentences, proj_sentences, edu_sentences, summary_sentences]):
            raw_content_snippet = (getattr(resume, "cleaned_text", "") or getattr(resume, "raw_text", "") or "")[:400].strip()
            if raw_content_snippet:
                summary_sentences = [raw_content_snippet]

        return {
            "job_position": jd.title,
            "job_domain": strat_dict.get("job_domain", "Technology"),
            "role_summary": strat_dict.get("role_summary", "") or getattr(jd, "role_summary", "") or jd.raw_text[:250],
            "shared_evaluation_strategy": {
                "essential_capabilities": strat_dict.get("essential_capabilities", []),
                "relevant_skills": strat_dict.get("relevant_skills", []),
                "expected_practical_knowledge": strat_dict.get("expected_practical_knowledge", []),
                "desirable_qualifications": strat_dict.get("desirable_qualifications", [])
            },
            "candidate_name": resume.candidate_name or "Candidate",
            "verified_experience_years": getattr(resume, "verified_experience_years", 0.0),
            "candidate_profile": {
                "summary": summary_sentences,
                "skills": skills_sentences,
                "experience": exp_sentences,
                "projects": proj_sentences,
                "education": edu_sentences
            },
            "requirements_to_assess": req_items,
            "deterministic_score": deterministic_eval.overall_score,
            "expected_output_schema": {
                "technical_relevance_summary": "1-2 sentences on domain and role fit",
                "domain_fit": "High | Medium | Low",
                "skills_assessment": {
                    "direct_evidence": ["skills demonstrated in experience/projects"],
                    "reasonable_inferences": ["reasonable inferences from candidate background"],
                    "genuine_gaps": ["critical requirements completely missing"],
                    "unknown_information": ["unverifiable qualities/soft skills not mentioned"]
                },
                "requirement_assessments": [
                    {
                        "requirement_id": "req_id",
                        "status": "VERIFIED|PARTIAL|TRANSFERABLE|UNSUPPORTED",
                        "evidence_quote": "exact sentence quote from candidate profile",
                        "inference_type": "DIRECT_EVIDENCE|REASONABLE_INFERENCE|UNKNOWN|GAP"
                    }
                ],
                "suggested_recommendation": "Recommended for Selection | Manual Review Required | Not Recommended",
                "executive_summary": "2-3 sentences summarising fit",
                "strengths": ["strengths with evidence"],
                "weaknesses": ["genuine gaps"],
                "project_and_internship_analysis": "Practical depth assessment",
                "uncertain_areas": ["ambiguities"]
            }
        }

    def _build_fallback_result(self, deterministic_eval: CandidateEvaluation, note: str) -> Dict[str, Any]:
        """Provides a safe fallback response using deterministic evaluation when LLM is unavailable."""
        missing = (
            deterministic_eval.total_essential_count
            - deterministic_eval.matched_essential_count
        )
        return {
            "llm_applied": False,
            "model_used": None,
            "recommendation": deterministic_eval.recommendation.value,
            "llm_suggested_recommendation": deterministic_eval.recommendation.value,
            "overall_score": deterministic_eval.overall_score,
            "requirement_assessments": [],
            "hallucinated_quotes_detected": 0,
            "executive_summary": deterministic_eval.decision_rationale,
            "technical_relevance_summary": "Evaluated using deterministic matching engine.",
            "domain_fit": "Deterministic",
            "skills_assessment": {
                "direct_evidence": [],
                "reasonable_inferences": [],
                "genuine_gaps": [f"Missing {missing} essential items."] if missing > 0 else [],
                "unknown_information": []
            },
            "strengths": [
                f"Matched {deterministic_eval.matched_essential_count}/"
                f"{deterministic_eval.total_essential_count} essential competencies."
            ],
            "weaknesses": (
                [f"Missing {missing} essential items."]
                if missing > 0 else []
            ),
            "project_and_internship_analysis": (
                f"Verified experience: {deterministic_eval.verified_tenure_years} yrs. "
                f"Practical evidence score: {deterministic_eval.practical_evidence_score:.1f}%."
            ),
            "uncertain_areas": [],
            "secondary_call_applied": False,
            "error": note
        }
