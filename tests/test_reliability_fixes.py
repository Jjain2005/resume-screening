import pytest
from unittest.mock import MagicMock
from modules.resume_processing.extractor import ResumeExtractor
from modules.resume_processing.models import ProcessedResume, EvidenceSentence, SectionType
from modules.knowledge_base.taxonomy import SkillTaxonomy
from modules.matching_engine.hybrid_engine import HybridMatchingEngine
from modules.matching_engine.semantic_matcher import SemanticMatcher
from modules.matching_engine.models import (
    RequirementMatchResult,
    MatchClassification,
    EvidenceStrength
)
from modules.jd_processing.models import (
    JobDescription,
    RequirementItem,
    RequirementType,
    RequirementPriority
)
from modules.scoring.evaluator import CandidateEvaluator, Recommendation
from modules.llm.evaluator import GroqContextualEvaluator, GroqClientManager
import os

TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "skills_taxonomy.json")

# ---------------------------------------------------------------------------
# 1. Date Parsing (Present, month/year, ISO) + Overlap Merging + Relevant Exp
# ---------------------------------------------------------------------------
def test_date_parsing_iso_month_year_present_and_relevant_exp():
    extractor = ResumeExtractor()
    resume = ProcessedResume(filename="candidate_dates.pdf")
    resume.all_sentences = [
        # ISO format with overlap: 2020-01 to 2022-06 and 2021-06 to Present
        EvidenceSentence(
            text="Senior Backend Engineer at CloudTech (2020-01 - 2022-06): Architected Python FastAPI microservices.",
            section=SectionType.EXPERIENCE
        ),
        EvidenceSentence(
            text="Staff Engineer at NovaCorp (2021-06 - Present): Led cloud infrastructure and Python distributed services.",
            section=SectionType.EXPERIENCE
        ),
        # Unrelated past experience: 05/2016 to 08/2018 (Month/Year format)
        EvidenceSentence(
            text="Customer Sales Associate at RetailMart (05/2016 - 08/2018): Managed floor inventory and sales transactions.",
            section=SectionType.EXPERIENCE
        )
    ]

    # Calculate experience with relevant keywords filtering
    years = extractor.calculate_verified_experience_years(
        resume,
        relevant_keywords=["python", "fastapi", "microservices", "cloud"]
    )

    # 1. Date parsing: Present should be dynamically computed (> 2024), 2020-01 is parsed properly
    # 2. Overlap merging: 2020-01 to Present merged into 1 continuous interval (~6+ years)
    # RetailMart is a separate interval (2.2 years)
    assert len(resume.employment_periods) == 2
    assert years >= 7.0

    # 3. Relevant experience: CloudTech & NovaCorp match keywords, RetailMart does NOT
    assert resume.relevant_experience_years >= 5.0
    assert resume.relevant_experience_years < years


# ---------------------------------------------------------------------------
# 2. Isolation of JD-specific embeddings and matching state between sessions
# ---------------------------------------------------------------------------
def test_session_state_matching_engine_isolation():
    taxonomy = SkillTaxonomy(TAXONOMY_PATH)
    shared_semantic_matcher = SemanticMatcher()

    # User 1 and User 2 create distinct HybridMatchingEngine instances sharing only immutable models
    engine_user1 = HybridMatchingEngine(taxonomy, semantic_matcher=shared_semantic_matcher, fuzzy_threshold=80.0)
    engine_user2 = HybridMatchingEngine(taxonomy, semantic_matcher=shared_semantic_matcher, fuzzy_threshold=95.0)

    jd1 = JobDescription(
        id="jd_python",
        title="Python Engineer",
        raw_text="...",
        requirements=[RequirementItem(id="r1", text="Python programming", req_type=RequirementType.SKILL, priority=RequirementPriority.REQUIRED)]
    )
    jd2 = JobDescription(
        id="jd_react",
        title="React Developer",
        raw_text="...",
        requirements=[RequirementItem(id="r2", text="React frontend development", req_type=RequirementType.SKILL, priority=RequirementPriority.REQUIRED)]
    )

    engine_user1.cache_jd(jd1)
    engine_user2.cache_jd(jd2)

    # State must be isolated: user 1's engine caches jd1, user 2's engine caches jd2
    assert engine_user1._cached_jd_id == "jd_python"
    assert engine_user2._cached_jd_id == "jd_react"
    assert "r1" in engine_user1._cached_jd_embeddings
    assert "r2" in engine_user2._cached_jd_embeddings
    assert "r2" not in engine_user1._cached_jd_embeddings

    # User 1 altering their threshold does not affect User 2
    engine_user1.fuzzy_threshold = 72.0
    assert engine_user2.fuzzy_threshold == 95.0


# ---------------------------------------------------------------------------
# 3. Candidate-Level Failure Isolation in Batch Processing
# ---------------------------------------------------------------------------
def test_candidate_failure_isolation_in_batch():
    evaluator = CandidateEvaluator()
    jd = JobDescription(
        id="jd_dev",
        title="Developer",
        raw_text="...",
        requirements=[RequirementItem(id="r1", text="Python", req_type=RequirementType.SKILL, priority=RequirementPriority.REQUIRED)]
    )

    # Candidate 1 succeeds
    cand1 = ProcessedResume(filename="good.pdf", candidate_name="Good Candidate")
    match1 = RequirementMatchResult(
        requirement_id="r1",
        requirement_text="Python",
        priority="Required",
        classification=MatchClassification.EXACT_MATCH,
        evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION
    )
    eval1 = evaluator.evaluate([match1], resume=cand1, jd=jd)

    # Candidate 2 fails (e.g. malformed PDF / extractor exception caught)
    cand2_failed = ProcessedResume(
        filename="corrupted.pdf",
        extraction_success=False,
        is_low_confidence=True,
        extraction_warnings=["Malformed PDF stream: syntax error"],
        confidence_reasons=["File parsing error"]
    )
    eval2 = evaluator.evaluate([], resume=cand2_failed, jd=jd)

    # Candidate 3 succeeds
    cand3 = ProcessedResume(filename="good2.pdf", candidate_name="Good Candidate 2")
    match3 = RequirementMatchResult(
        requirement_id="r1",
        requirement_text="Python",
        priority="Required",
        classification=MatchClassification.EXACT_MATCH,
        evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION
    )
    eval3 = evaluator.evaluate([match3], resume=cand3, jd=jd)

    # The batch preserves successful candidate evaluations while isolating candidate 2 as Unable to Evaluate
    assert eval1.recommendation == Recommendation.RECOMMENDED
    assert eval2.recommendation == Recommendation.UNABLE_TO_EVALUATE
    assert eval3.recommendation == Recommendation.RECOMMENDED


# ---------------------------------------------------------------------------
# 4. Groq Failure Uses Deterministic Fallback; Unable to Evaluate Only on True Failure
# ---------------------------------------------------------------------------
def test_groq_failure_uses_deterministic_fallback():
    mock_mgr = MagicMock(spec=GroqClientManager)
    mock_mgr.is_available = True
    mock_mgr.model = "llama-3.3-70b-versatile"
    mock_client = MagicMock()
    # Groq throws an unexpected network error / rate limit exhaustion
    mock_client.chat.completions.create.side_effect = RuntimeError("Groq 500 Internal Server Error")
    mock_mgr.get_client.return_value = mock_client

    groq_evaluator = GroqContextualEvaluator(mock_mgr)

    jd = JobDescription(
        id="jd_test",
        title="Senior Python Engineer",
        raw_text="...",
        requirements=[
            RequirementItem(id="r1", text="Python", req_type=RequirementType.SKILL, priority=RequirementPriority.REQUIRED),
            RequirementItem(id="r2", text="FastAPI", req_type=RequirementType.SKILL, priority=RequirementPriority.REQUIRED)
        ]
    )
    resume = ProcessedResume(
        filename="alice.pdf",
        candidate_name="Alice",
        raw_text="Experienced Python FastAPI developer with 4 years building APIs",
        cleaned_text="Experienced Python FastAPI developer with 4 years building APIs",
        extraction_success=True,
        is_low_confidence=False,
        verified_experience_years=4.0
    )
    matches = [
        RequirementMatchResult(
            requirement_id="r1", requirement_text="Python", priority="Required",
            classification=MatchClassification.EXACT_MATCH, evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION
        ),
        RequirementMatchResult(
            requirement_id="r2", requirement_text="FastAPI", priority="Required",
            classification=MatchClassification.EXACT_MATCH, evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION
        )
    ]
    det_eval = CandidateEvaluator().evaluate(matches, resume=resume, jd=jd)
    assert det_eval.recommendation == Recommendation.RECOMMENDED

    batch_res = groq_evaluator.evaluate_candidates_batch(
        jd=jd,
        candidates=[(resume, matches, det_eval)]
    )

    proc_res, final_eval, llm_rep = batch_res[0]
    # Deterministic evaluation is preserved; must NOT return Unable to Evaluate
    assert final_eval.recommendation == Recommendation.RECOMMENDED
    assert final_eval.overall_score >= 68.0
    assert llm_rep["llm_applied"] is False


# ---------------------------------------------------------------------------
# 5. Manual Review Required Applies Only to Material Ambiguity
#    Transferable skills alone must not trigger it
# ---------------------------------------------------------------------------
def test_transferable_skills_alone_cannot_trigger_manual_review():
    evaluator = CandidateEvaluator()

    # Candidate has 1 transferable skill (Angular for React), but is missing other critical requirements (Docker, Python)
    matches = [
        RequirementMatchResult(
            requirement_id="r1",
            requirement_text="Python",
            priority="Required",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="r2",
            requirement_text="Docker",
            priority="Required",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="r3",
            requirement_text="React",
            priority="Required",
            classification=MatchClassification.RELATED_SKILL_NOT_EQUIVALENT,
            matched_term="Angular",
            explanation="Angular is a related frontend framework, not React"
        )
    ]
    resume = ProcessedResume(filename="cand.pdf", verified_experience_years=3.0)
    res = evaluator.evaluate(matches, resume=resume)

    # Missing critical requirements with no ambiguity -> MUST be NOT_RECOMMENDED, NOT MANUAL_REVIEW_REQUIRED
    assert res.recommendation == Recommendation.NOT_RECOMMENDED
    assert "not recommended" in res.decision_rationale.lower()


def test_immaterial_ambiguity_cannot_trigger_manual_review():
    evaluator = CandidateEvaluator()

    # Candidate fails 3 requirements and has 1 uncertain requirement
    # Even if the 1 uncertain requirement were resolved, coverage would only be 1/4 (25%), well below passing
    matches = [
        RequirementMatchResult(
            requirement_id="r1", requirement_text="Skill 1", priority="Required",
            classification=MatchClassification.NOT_FOUND, evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="r2", requirement_text="Skill 2", priority="Required",
            classification=MatchClassification.NOT_FOUND, evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="r3", requirement_text="Skill 3", priority="Required",
            classification=MatchClassification.NOT_FOUND, evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="r4", requirement_text="Skill 4", priority="Required",
            classification=MatchClassification.NEEDS_MANUAL_REVIEW, evidence_strength=EvidenceStrength.MENTIONED
        )
    ]
    resume = ProcessedResume(filename="weak_ambiguous.pdf", verified_experience_years=1.0)
    res = evaluator.evaluate(matches, resume=resume)

    # Immaterial ambiguity cannot trigger Manual Review Required
    assert res.recommendation == Recommendation.NOT_RECOMMENDED


# ---------------------------------------------------------------------------
# 6. Responsibilities Provide Context But Do Not Independently Increase Competency Scores
# ---------------------------------------------------------------------------
def test_responsibilities_do_not_increase_competency_scores():
    evaluator = CandidateEvaluator()

    # Candidate matches 1 technical skill (Python: 100%) out of 1 skill
    skill_match = RequirementMatchResult(
        requirement_id="s1",
        requirement_text="Python Development",
        req_type="Skill",
        priority="Core",
        weight=2.0,
        classification=MatchClassification.EXACT_MATCH,
        evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION
    )

    eval_without_resp = evaluator.evaluate([skill_match])

    # Now add two broad responsibilities
    resp_match1 = RequirementMatchResult(
        requirement_id="r1",
        requirement_text="Responsible for cross-functional collaboration and agile ceremonies",
        req_type="Responsibility",
        priority="Core",
        weight=2.0,
        classification=MatchClassification.EXACT_MATCH,
        evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION
    )
    resp_match2 = RequirementMatchResult(
        requirement_id="r2",
        requirement_text="Lead project delivery timelines and sprint planning",
        req_type="Responsibility",
        priority="Core",
        weight=2.0,
        classification=MatchClassification.EXACT_MATCH,
        evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION
    )

    eval_with_resp = evaluator.evaluate([skill_match, resp_match1, resp_match2])

    # Essential coverage percentage must reflect competency requirements (100%), not diluted or inflated by responsibilities
    assert eval_without_resp.essential_coverage_pct == 100.0
    assert eval_with_resp.essential_coverage_pct == 100.0

    # Responsibilities provide context (recorded in dynamic_criteria_scores)
    assert "r1" in eval_with_resp.dynamic_criteria_scores
    assert "r2" in eval_with_resp.dynamic_criteria_scores
