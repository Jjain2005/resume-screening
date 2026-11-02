import os
import pytest
from modules.knowledge_base.taxonomy import SkillTaxonomy
from modules.knowledge_base.models import RelationshipType
from modules.resume_processing.extractor import ResumeExtractor
from modules.resume_processing.models import ProcessedResume, EvidenceSentence, SectionType
from modules.jd_processing.parser import JobDescriptionParser
from modules.jd_processing.models import JobDescription, RequirementItem, RequirementType as ReqType, RequirementPriority
from modules.matching_engine.hybrid_engine import HybridMatchingEngine
from modules.matching_engine.models import MatchClassification, EvidenceStrength, RequirementMatchResult
from modules.scoring.evaluator import CandidateEvaluator, ScoringWeights, Recommendation
from modules.reporting.reporter import ScreeningReportGenerator

TAXONOMY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "knowledge_base", "skills_taxonomy.json"
)
TEST_RESUMES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "test_resumes"
)

@pytest.fixture(scope="session")
def taxonomy():
    return SkillTaxonomy(TAXONOMY_PATH)

@pytest.fixture(scope="session")
def matching_engine(taxonomy):
    return HybridMatchingEngine(taxonomy)

@pytest.fixture(scope="session")
def resume_extractor():
    return ResumeExtractor()

@pytest.fixture(scope="session")
def jd_parser(taxonomy):
    return JobDescriptionParser(taxonomy)

# -------------------------------------------------------------
# 1. Exact Skill Matches
# -------------------------------------------------------------
def test_exact_canonical_match(taxonomy, matching_engine):
    resume = ProcessedResume(filename="dummy.pdf")
    resume.all_sentences = [
        EvidenceSentence(
            text="Developed microservices using Python and PostgreSQL databases.",
            section=SectionType.EXPERIENCE
        )
    ]
    req = RequirementItem(
        id="r1",
        text="Strong proficiency in Python",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Python"
    )
    result = matching_engine.match_single_requirement(req, resume)
    assert result.classification == MatchClassification.EXACT_MATCH
    assert result.matched_term.lower() == "python"
    assert result.evidence_strength in (EvidenceStrength.EXPERIENCE_EVIDENCE, EvidenceStrength.STRONG_IMPLEMENTATION)

# -------------------------------------------------------------
# 2. Known Aliases
# -------------------------------------------------------------
def test_known_alias_match(taxonomy, matching_engine):
    # JD asks for Natural Language Processing, resume mentions alias "NLP"
    resume = ProcessedResume(filename="dummy.pdf")
    resume.all_sentences = [
        EvidenceSentence(
            text="Led several enterprise NLP projects involving sentiment analysis.",
            section=SectionType.EXPERIENCE
        )
    ]
    req = RequirementItem(
        id="r2",
        text="Experience with Natural Language Processing",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Natural Language Processing"
    )
    result = matching_engine.match_single_requirement(req, resume)
    assert result.classification == MatchClassification.EXACT_MATCH
    assert result.matched_term == "NLP"
    assert result.canonical_skill == "Natural Language Processing"
    assert "known alias" in result.explanation.lower()

# -------------------------------------------------------------
# 3. Spelling Variations (Fuzzy Matching)
# -------------------------------------------------------------
def test_fuzzy_spelling_variation(taxonomy, matching_engine):
    # Typo: "Kubernetes" written as "Kubernets" or "PostgreSQ"
    resume = ProcessedResume(filename="dummy.pdf")
    resume.all_sentences = [
        EvidenceSentence(
            text="Deployed microservices on Kubernets clusters and managed pods.",
            section=SectionType.EXPERIENCE
        )
    ]
    req = RequirementItem(
        id="r3",
        text="Experience with Kubernetes",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Kubernetes"
    )
    result = matching_engine.match_single_requirement(req, resume)
    assert result.classification in (MatchClassification.FUZZY_MATCH, MatchClassification.NEEDS_MANUAL_REVIEW)
    assert "kubernets" in result.matched_term.lower()
    assert result.similarity_score >= 0.85

# -------------------------------------------------------------
# 4. Related But Non-Equivalent Skills
# -------------------------------------------------------------
def test_related_skill_not_equivalent(taxonomy, matching_engine):
    # JD asks for React, candidate only has Angular
    resume = ProcessedResume(filename="dummy.pdf")
    resume.all_sentences = [
        EvidenceSentence(
            text="Developed high performance web interfaces using Angular and TypeScript.",
            section=SectionType.EXPERIENCE
        )
    ]
    req = RequirementItem(
        id="r4",
        text="Proficiency in React frontend development",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="React"
    )
    result = matching_engine.match_single_requirement(req, resume)
    assert result.classification == MatchClassification.RELATED_SKILL_NOT_EQUIVALENT
    assert result.matched_term == "Angular"
    assert "not fully equivalent" in result.explanation.lower()

# -------------------------------------------------------------
# 5. Semantic Matches (Paraphrased Evidence)
# -------------------------------------------------------------
def test_semantic_match_paraphrase(taxonomy, matching_engine):
    # JD requirement phrasing vs resume phrasing with different words
    resume = ProcessedResume(filename="dummy.pdf")
    resume.all_sentences = [
        EvidenceSentence(
            text="Trained convolutional neural networks using PyTorch on AWS GPU clusters.",
            section=SectionType.PROJECTS
        )
    ]
    req = RequirementItem(
        id="r5",
        text="Experience in deep learning and training neural network architectures",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Deep Learning"
    )
    result = matching_engine.match_single_requirement(req, resume)
    assert result.classification in (MatchClassification.STRONG_SEMANTIC_MATCH, MatchClassification.EXACT_MATCH)
    assert result.similarity_score >= 0.65

# -------------------------------------------------------------
# 6 & 7. Skills Found in Project vs Professional Experience
# -------------------------------------------------------------
def test_project_vs_experience_evidence(taxonomy, matching_engine):
    # Skill in Project section
    proj_resume = ProcessedResume(filename="proj.pdf")
    proj_resume.all_sentences = [
        EvidenceSentence(
            text="Implemented biometric facial recognition using PyTorch in academic research.",
            section=SectionType.PROJECTS
        )
    ]
    # Skill in Experience section with metrics
    exp_resume = ProcessedResume(filename="exp.pdf")
    exp_resume.all_sentences = [
        EvidenceSentence(
            text="Architected and deployed PyTorch inference pipelines serving 5,000 QPS with 99.9% uptime.",
            section=SectionType.EXPERIENCE
        )
    ]

    req = RequirementItem(
        id="r6",
        text="PyTorch",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="PyTorch"
    )

    proj_res = matching_engine.match_single_requirement(req, proj_resume)
    exp_res = matching_engine.match_single_requirement(req, exp_resume)

    assert proj_res.classification == MatchClassification.EXACT_MATCH
    assert exp_res.classification == MatchClassification.EXACT_MATCH
    assert exp_res.evidence_strength == EvidenceStrength.STRONG_IMPLEMENTATION

# -------------------------------------------------------------
# 8. Missing Requirements
# -------------------------------------------------------------
def test_missing_requirement(taxonomy, matching_engine):
    resume = ProcessedResume(filename="dummy.pdf")
    resume.all_sentences = [
        EvidenceSentence(
            text="Accountant with 4 years managing payroll and ledger reconciliations.",
            section=SectionType.EXPERIENCE
        )
    ]
    req = RequirementItem(
        id="r7",
        text="Kubernetes cluster management",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Kubernetes"
    )
    result = matching_engine.match_single_requirement(req, resume)
    assert result.classification == MatchClassification.NOT_FOUND
    assert result.matched_term is None
    assert result.similarity_score == 0.0

# -------------------------------------------------------------
# 9. Weak or Ambiguous Evidence (Listed Only)
# -------------------------------------------------------------
def test_weak_evidence_listed_only(taxonomy, matching_engine):
    resume = ProcessedResume(filename="dummy.pdf")
    resume.all_sentences = [
        EvidenceSentence(
            text="Python, Docker, Kubernetes, AWS",
            section=SectionType.SKILLS
        )
    ]
    req = RequirementItem(
        id="r8",
        text="Docker",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Docker"
    )
    result = matching_engine.match_single_requirement(req, resume)
    assert result.classification == MatchClassification.EXACT_MATCH
    assert result.evidence_strength == EvidenceStrength.LISTED_ONLY

# -------------------------------------------------------------
# 10. Different Industry Requirements (HR / Marketing / Finance)
# -------------------------------------------------------------
def test_different_industry_requirements(taxonomy, matching_engine):
    # Test Marketing Requirement
    resume_mkt = ProcessedResume(filename="mkt.pdf")
    resume_mkt.all_sentences = [
        EvidenceSentence(
            text="Managed a $50k monthly budget for Google Ads campaigns with focus on keyword bidding.",
            section=SectionType.EXPERIENCE
        )
    ]
    req_mkt = RequirementItem(
        id="mkt_1",
        text="Google Ads PPC management",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Google Ads"
    )
    res_mkt = matching_engine.match_single_requirement(req_mkt, resume_mkt)
    assert res_mkt.classification in (MatchClassification.EXACT_MATCH, MatchClassification.STRONG_SEMANTIC_MATCH)

    # Test HR Requirement
    resume_hr = ProcessedResume(filename="hr.pdf")
    resume_hr.all_sentences = [
        EvidenceSentence(
            text="Led technical recruiting and candidate sourcing, closing 40+ engineering candidates annually utilizing LinkedIn Recruiter and ATS.",
            section=SectionType.EXPERIENCE
        )
    ]
    req_hr = RequirementItem(
        id="hr_1",
        text="Talent Acquisition and technical recruiting",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Talent Acquisition"
    )
    res_hr = matching_engine.match_single_requirement(req_hr, resume_hr)
    assert res_hr.classification in (MatchClassification.EXACT_MATCH, MatchClassification.STRONG_SEMANTIC_MATCH)

# -------------------------------------------------------------
# 11. Multiple Resumes Against One JD (End-to-End Pipeline)
# -------------------------------------------------------------
def test_multiple_resumes_screening(taxonomy, resume_extractor, jd_parser, matching_engine):
    jd_text = """
    Senior AI & Backend Engineer
    Requirements:
    • 4+ years experience in Python backend development
    • Natural Language Processing (NLP) or Machine Learning model training
    • Containerization with Docker and Kubernetes
    • Relational database design with PostgreSQL
    Preferred:
    • FastAPI microservices development
    • CI/CD pipeline automation
    """
    jd = jd_parser.parse(jd_text, jd_id="jd_ai_test")
    assert len(jd.requirements) >= 4

    # Extract our generated PDF test resumes
    alex_pdf = os.path.join(TEST_RESUMES_DIR, "alex_morgan_ai_engineer.pdf")
    sam_pdf = os.path.join(TEST_RESUMES_DIR, "sam_rivera_marketing.pdf")

    assert os.path.exists(alex_pdf), "Alex Morgan test resume PDF must exist"
    assert os.path.exists(sam_pdf), "Sam Rivera test resume PDF must exist"

    alex_resume = resume_extractor.extract_from_pdf(alex_pdf, "alex_morgan.pdf")
    sam_resume = resume_extractor.extract_from_pdf(sam_pdf, "sam_rivera.pdf")

    evaluator = CandidateEvaluator()

    # Screen Alex
    alex_matches = matching_engine.match_resume_to_jd(alex_resume, jd)
    alex_eval = evaluator.evaluate(alex_matches)

    # Screen Sam
    sam_matches = matching_engine.match_resume_to_jd(sam_resume, jd)
    sam_eval = evaluator.evaluate(sam_matches)

    # Validate relative ranking
    assert alex_eval.overall_score > sam_eval.overall_score
    assert alex_eval.recommendation == Recommendation.RECOMMENDED_FOR_SELECTION
    assert sam_eval.recommendation == Recommendation.NOT_RECOMMENDED

    # Test report generation & comparison dataframe
    reports = [
        ScreeningReportGenerator.generate_candidate_report(alex_resume, alex_eval, alex_matches, jd),
        ScreeningReportGenerator.generate_candidate_report(sam_resume, sam_eval, sam_matches, jd)
    ]
    df = ScreeningReportGenerator.generate_comparison_dataframe(reports)
    assert len(df) == 2
    assert df.iloc[0]["Candidate Name"] == "Alex Morgan"

# -------------------------------------------------------------
# 12. Resume Extraction Failure / Warning Handling
# -------------------------------------------------------------
def test_minimal_resume_extraction_warning(resume_extractor):
    minimal_pdf = os.path.join(TEST_RESUMES_DIR, "morgan_vance_minimal.pdf")
    resume = resume_extractor.extract_from_pdf(minimal_pdf, "morgan_vance.pdf")
    assert resume.filename == "morgan_vance.pdf"
    # Even if very short, extractor handles without crashing and records warnings or sections
    assert resume.page_count >= 1

# -------------------------------------------------------------
# 13. LLM Evaluator Fallback & Augmented Report Generation
# -------------------------------------------------------------
def test_groq_evaluator_fallback_and_report():
    from modules.llm.evaluator import GroqContextualEvaluator
    from modules.llm.groq_client import GroqClientManager

    # Test with dummy key to verify graceful error handling
    dummy_manager = GroqClientManager(api_key="gsk_invalid_test_key_12345", model="qwen/qwen3.8-27b")
    evaluator = GroqContextualEvaluator(dummy_manager)

    resume = ProcessedResume(filename="test.pdf", candidate_name="Test Candidate")
    jd = JobDescription(id="jd_test", title="Software Engineer", raw_text="Test JD", requirements=[])
    det_eval = CandidateEvaluator().evaluate([])

    # Must not raise an exception; falls back gracefully
    result = evaluator.evaluate_candidate(jd, resume, [], det_eval)
    assert "llm_applied" in result
    assert "overall_score" in result
    assert "strengths" in result
    assert "weaknesses" in result

    # Test report integration
    report = ScreeningReportGenerator.generate_candidate_report(
        resume=resume,
        evaluation=det_eval,
        match_results=[],
        jd=jd,
        llm_evaluation=result
    )
    assert report["candidate_name"] == "Test Candidate"
    assert "evaluation_summary" in report

# -------------------------------------------------------------
# 14. Requirement 8.1: Overlapping Employment Date Deduplication
# -------------------------------------------------------------
def test_overlapping_employment_date_deduplication(resume_extractor):
    # Overlapping intervals: Jan 2020 - Jan 2022 (2 yrs) and Jan 2021 - Jan 2023 (2 yrs)
    # Merged timeline: Jan 2020 to Jan 2023 = 3.0 years, not naive 4.0 years
    resume_overlap = ProcessedResume(filename="overlap.pdf")
    resume_overlap.all_sentences = [
        EvidenceSentence(text="Software Engineer at TechCorp (Jan 2020 - Jan 2022) - Built backend APIs.", section=SectionType.EXPERIENCE),
        EvidenceSentence(text="Senior Engineer at BetaSoft (Jan 2021 - Jan 2023) - Led migration.", section=SectionType.EXPERIENCE)
    ]
    years_overlap = resume_extractor.calculate_verified_experience_years(resume_overlap)
    assert 2.8 <= years_overlap <= 3.2
    assert len(resume_overlap.employment_periods) == 1

    # Non-overlapping intervals: 2018 - 2020 (2 yrs) and 2021 - 2024 (3 yrs) = 5.0 years
    resume_sep = ProcessedResume(filename="sep.pdf")
    resume_sep.all_sentences = [
        EvidenceSentence(text="Developer at Alpha (2018 - 2020)", section=SectionType.EXPERIENCE),
        EvidenceSentence(text="Consultant at Gamma (2021 - 2024)", section=SectionType.EXPERIENCE)
    ]
    years_sep = resume_extractor.calculate_verified_experience_years(resume_sep)
    assert 4.8 <= years_sep <= 5.2
    assert len(resume_sep.employment_periods) == 2

    # Missing or unclear dates: should not assume or invent experience
    resume_unclear = ProcessedResume(filename="unclear.pdf")
    resume_unclear.all_sentences = [
        EvidenceSentence(text="Software Engineer at Startup (Date unavailable / duration unspecified)", section=SectionType.EXPERIENCE)
    ]
    years_unclear = resume_extractor.calculate_verified_experience_years(resume_unclear)
    assert years_unclear == 0.0
    assert len(resume_unclear.employment_periods) == 0

# -------------------------------------------------------------
# 15. Requirement 8.2: Low Extraction Confidence Routing to Manual Review
# -------------------------------------------------------------
def test_low_extraction_confidence_routing_to_manual_review(resume_extractor):
    # Broken/scanned/low-text resume (< 150 characters)
    scanned_text = "Img scan err 123"
    resume = ProcessedResume(
        filename="scanned_resume.pdf",
        raw_text=scanned_text,
        page_count=1
    )
    conf, is_low, reasons = resume_extractor.check_extraction_confidence(resume)
    resume.extraction_confidence = conf
    resume.is_low_confidence = is_low
    resume.confidence_reasons = reasons

    assert is_low is True
    assert conf < 0.50

    evaluator = CandidateEvaluator()
    # Even if matches is empty or low, must NEVER automatically reject due to extraction failure
    evaluation = evaluator.evaluate(match_results=[], resume=resume)
    assert evaluation.recommendation == Recommendation.MANUAL_REVIEW_REQUIRED
    assert evaluation.is_low_confidence_extraction is True
    assert "Low extraction confidence" in evaluation.decision_rationale

# -------------------------------------------------------------
# 16. Requirement 8.3: Transferable Skills Routing to Manual Review
# -------------------------------------------------------------
def test_transferable_skills_routing_to_manual_review(matching_engine):
    # Candidate knows Angular, JD requires React
    resume = ProcessedResume(filename="angular_dev.pdf", verified_experience_years=4.0)
    resume.all_sentences = [
        EvidenceSentence(
            text="Developed responsive web applications and dashboards using Angular and TypeScript.",
            section=SectionType.EXPERIENCE
        )
    ]
    req_react = RequirementItem(
        id="r_react",
        text="Proficiency in React frontend development",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="React"
    )
    match = matching_engine.match_single_requirement(req_react, resume)
    assert match.classification == MatchClassification.RELATED_SKILL_NOT_EQUIVALENT
    assert match.matched_term == "Angular"

    evaluator = CandidateEvaluator()
    eval_result = evaluator.evaluate([match], resume=resume)

    # Strictly not counted as essential coverage (0%)
    assert eval_result.essential_coverage_pct == 0.0
    # Flagged as transferable skill
    assert len(eval_result.transferable_skills) >= 1
    assert eval_result.transferable_skills[0]["candidate_has"] == "Angular"
    # Not rejected automatically; routed to manual review for human assessment
    assert eval_result.recommendation == Recommendation.MANUAL_REVIEW_REQUIRED
    assert "transferable" in eval_result.decision_rationale.lower()

# -------------------------------------------------------------
# 17. Requirement 8.4: Skills Listed Without Practical Evidence Penalty
# -------------------------------------------------------------
def test_skills_listed_without_practical_evidence_penalty():
    evaluator = CandidateEvaluator()

    # Case A: Keyword stuffer - skills in skills section only, no practical sentences
    keyword_stuffer_matches = [
        RequirementMatchResult(
            requirement_id="k1",
            requirement_text="Python",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Python",
            evidence_strength=EvidenceStrength.LISTED_ONLY,
            resume_section=SectionType.SKILLS
        ),
        RequirementMatchResult(
            requirement_id="k2",
            requirement_text="Docker",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Docker",
            evidence_strength=EvidenceStrength.LISTED_ONLY,
            resume_section=SectionType.SKILLS
        ),
        RequirementMatchResult(
            requirement_id="k3",
            requirement_text="Kubernetes",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Kubernetes",
            evidence_strength=EvidenceStrength.LISTED_ONLY,
            resume_section=SectionType.SKILLS
        )
    ]
    eval_stuffer = evaluator.evaluate(keyword_stuffer_matches)
    assert eval_stuffer.unsupported_keyword_penalty > 0.0

    # Case B: Candidate with skills list BUT supported by practical experience sentences
    legit_candidate_matches = [
        RequirementMatchResult(
            requirement_id="k1",
            requirement_text="Python",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Python",
            evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION,
            resume_section=SectionType.EXPERIENCE,
            evidence_sentence="Architected Python microservices handling 10k RPS."
        ),
        RequirementMatchResult(
            requirement_id="k2",
            requirement_text="Docker",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Docker",
            evidence_strength=EvidenceStrength.EXPERIENCE_EVIDENCE,
            resume_section=SectionType.EXPERIENCE,
            evidence_sentence="Containerized backend services with Docker."
        ),
        RequirementMatchResult(
            requirement_id="k3",
            requirement_text="Kubernetes",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Kubernetes",
            evidence_strength=EvidenceStrength.EXPERIENCE_EVIDENCE,
            resume_section=SectionType.EXPERIENCE,
            evidence_sentence="Deployed Helm charts to production Kubernetes clusters."
        )
    ]
    resume_legit = ProcessedResume(filename="legit.pdf", verified_experience_years=3.0)
    eval_legit = evaluator.evaluate(legit_candidate_matches, resume=resume_legit)
    # Must NOT be penalized just for listing many skills when supported by experience
    assert eval_legit.unsupported_keyword_penalty == 0.0

# -------------------------------------------------------------
# 18. Requirement 8.5: Strong Practical Evidence With Fewer Preferred Skills
# -------------------------------------------------------------
def test_strong_practical_evidence_with_fewer_preferred_skills():
    evaluator = CandidateEvaluator()

    # Candidate meets 100% of Essential Skills with Strong Implementation & Experience
    # but meets 0% of Preferred Skills
    matches = [
        RequirementMatchResult(
            requirement_id="e1",
            requirement_text="Python",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION,
            resume_section=SectionType.EXPERIENCE
        ),
        RequirementMatchResult(
            requirement_id="e2",
            requirement_text="FastAPI",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION,
            resume_section=SectionType.EXPERIENCE
        ),
        RequirementMatchResult(
            requirement_id="e3",
            requirement_text="PostgreSQL",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            evidence_strength=EvidenceStrength.EXPERIENCE_EVIDENCE,
            resume_section=SectionType.EXPERIENCE
        ),
        # Preferred skills NOT found
        RequirementMatchResult(
            requirement_id="p1",
            requirement_text="GraphQL",
            priority="Preferred",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="p2",
            requirement_text="Redis Caching",
            priority="Preferred",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        )
    ]
    resume = ProcessedResume(filename="strong_dev.pdf", verified_experience_years=4.0)
    jd = JobDescription(id="jd_test", title="Backend Dev", raw_text="...", requirements=[])

    result = evaluator.evaluate(matches, resume=resume, jd=jd)
    assert result.essential_coverage_pct == 100.0
    assert result.preferred_coverage_pct == 0.0
    assert result.practical_evidence_score >= 80.0
    assert result.overall_score >= 72.0
    assert result.recommendation == Recommendation.RECOMMENDED_FOR_SELECTION

# -------------------------------------------------------------
# 19. Requirement 8.6: Groq Call Limits & Conditional Second Call
# -------------------------------------------------------------
def test_groq_call_limits_and_conditional_second_call():
    import json
    from unittest.mock import MagicMock
    from modules.llm.evaluator import GroqContextualEvaluator
    from modules.llm.groq_client import GroqClientManager
    from modules.scoring.models import CandidateEvaluation

    mock_client_mgr = MagicMock(spec=GroqClientManager)
    mock_client_mgr.is_available = True
    mock_client_mgr.model = "qwen/qwen3.8-27b"
    mock_groq_client = MagicMock()
    mock_client_mgr.get_client.return_value = mock_groq_client

    evaluator = GroqContextualEvaluator(mock_client_mgr)
    evaluator.reset_call_counter()

    jd = JobDescription(id="jd1", title="ML Engineer", raw_text="ML Engineer", requirements=[])
    clean_resume = ProcessedResume(filename="clean.pdf", verified_experience_years=0.0)

    # 1. Obvious hard mismatch with clean extraction -> 0 calls
    mismatch_eval = CandidateEvaluation(
        overall_score=20.0,
        essential_coverage_pct=20.0,
        preferred_coverage_pct=0.0,
        practical_evidence_score=0.0,
        experience_duration_score=0.0,
        project_depth_score=0.0,
        unsupported_keyword_penalty=0.0,
        verified_tenure_years=0.0,
        recommendation=Recommendation.NOT_RECOMMENDED,
        total_essential_count=5,
        matched_essential_count=1,
        total_preferred_count=2,
        matched_preferred_count=0,
        score_breakdown_formula="Formula",
        decision_rationale="Rationale"
    )
    res_mismatch = evaluator.evaluate_candidate(jd, clean_resume, [], mismatch_eval)
    assert res_mismatch["llm_applied"] is False
    assert evaluator.call_count == 0

    # 2. Low confidence extraction -> 0 calls, routed to manual review
    low_conf_resume = ProcessedResume(filename="low.pdf", is_low_confidence=True, verified_experience_years=2.0)
    low_eval = CandidateEvaluation(
        overall_score=50.0,
        essential_coverage_pct=50.0,
        preferred_coverage_pct=50.0,
        practical_evidence_score=50.0,
        experience_duration_score=50.0,
        project_depth_score=50.0,
        unsupported_keyword_penalty=0.0,
        verified_tenure_years=2.0,
        recommendation=Recommendation.MANUAL_REVIEW_REQUIRED,
        total_essential_count=4,
        matched_essential_count=2,
        total_preferred_count=2,
        matched_preferred_count=1,
        is_low_confidence_extraction=True,
        score_breakdown_formula="Formula",
        decision_rationale="Rationale"
    )
    res_low = evaluator.evaluate_candidate(jd, low_conf_resume, [], low_eval)
    assert res_low["llm_applied"] is False
    assert evaluator.call_count == 0

    # 3. Viable candidate without ambiguity -> Exactly 1 batched call
    mock_choice1 = MagicMock()
    mock_choice1.message.content = json.dumps({
        "overall_score": 85.0,
        "executive_summary": "Strong fit for role.",
        "strengths": ["Python expertise"],
        "weaknesses": [],
        "uncertain_areas": []
    })
    mock_groq_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice1])

    viable_resume = ProcessedResume(filename="viable.pdf", verified_experience_years=4.0)
    viable_eval = CandidateEvaluation(
        overall_score=80.0,
        essential_coverage_pct=90.0,
        preferred_coverage_pct=70.0,
        practical_evidence_score=85.0,
        experience_duration_score=80.0,
        project_depth_score=75.0,
        unsupported_keyword_penalty=0.0,
        verified_tenure_years=4.0,
        recommendation=Recommendation.RECOMMENDED_FOR_SELECTION,
        total_essential_count=5,
        matched_essential_count=5,
        total_preferred_count=2,
        matched_preferred_count=2,
        score_breakdown_formula="Formula",
        decision_rationale="Rationale"
    )
    res_viable = evaluator.evaluate_candidate(jd, viable_resume, [], viable_eval)
    assert res_viable["llm_applied"] is True
    assert evaluator.call_count == 1
    assert res_viable["secondary_call_applied"] is False

    # 4. Borderline candidate with unresolved ambiguity -> Triggers conditional 2nd probe call
    mock_choice2_primary = MagicMock()
    mock_choice2_primary.message.content = json.dumps({
        "overall_score": 62.0,
        "executive_summary": "Borderline skills with uncertain cloud deployment experience.",
        "strengths": ["Backend skills"],
        "weaknesses": ["Unclear AWS production scale"],
        "uncertain_areas": ["Is AWS production experience verified or merely personal sandbox?"]
    })
    mock_choice2_probe = MagicMock()
    mock_choice2_probe.message.content = json.dumps({
        "clarification": "Evidence indicates commercial AWS production deployment.",
        "is_resolved": True
    })
    mock_groq_client.chat.completions.create.side_effect = [
        MagicMock(choices=[mock_choice2_primary]),
        MagicMock(choices=[mock_choice2_probe])
    ]

    ambiguous_resume = ProcessedResume(filename="ambiguous.pdf", verified_experience_years=2.5)
    ambiguous_eval = CandidateEvaluation(
        overall_score=60.0,
        essential_coverage_pct=65.0,
        preferred_coverage_pct=50.0,
        practical_evidence_score=60.0,
        experience_duration_score=50.0,
        project_depth_score=40.0,
        unsupported_keyword_penalty=0.0,
        verified_tenure_years=2.5,
        recommendation=Recommendation.MANUAL_REVIEW_REQUIRED,
        total_essential_count=4,
        matched_essential_count=3,
        total_preferred_count=2,
        matched_preferred_count=1,
        score_breakdown_formula="Formula",
        decision_rationale="Rationale"
    )
    res_ambiguous = evaluator.evaluate_candidate(jd, ambiguous_resume, [], ambiguous_eval)
    assert res_ambiguous["secondary_call_applied"] is True
    # 1 initial call + 1 probe call = 2 calls for this candidate, 3 calls total in session
    assert evaluator.call_count == 3

    # Check stats reporting
    stats = evaluator.get_call_stats(total_candidates=4)
    assert stats["total_calls"] == 3
    assert stats["total_candidates"] == 4
    assert stats["average_calls_per_candidate"] == 0.75

# -------------------------------------------------------------
# 20. Requirement 8.7: Score Boundaries & Decision-Rule Consistency
# -------------------------------------------------------------
def test_score_boundaries_and_decision_consistency():
    evaluator = CandidateEvaluator()

    # Empty match results
    empty_eval = evaluator.evaluate([])
    assert empty_eval.overall_score == 0.0
    assert empty_eval.recommendation == Recommendation.NOT_RECOMMENDED

    # Perfect candidate matches
    perfect_matches = [
        RequirementMatchResult(
            requirement_id=f"e_{i}",
            requirement_text=f"Essential {i}",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION,
            resume_section=SectionType.EXPERIENCE
        )
        for i in range(4)
    ] + [
        RequirementMatchResult(
            requirement_id=f"p_{i}",
            requirement_text=f"Preferred {i}",
            priority="Preferred",
            classification=MatchClassification.EXACT_MATCH,
            evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION,
            resume_section=SectionType.PROJECTS
        )
        for i in range(2)
    ]
    resume_perf = ProcessedResume(filename="perfect.pdf", verified_experience_years=5.0)
    jd_perf = JobDescription(id="jd_p", title="Lead Engineer", raw_text="...", requirements=[])

    perf_eval_1 = evaluator.evaluate(perfect_matches, resume=resume_perf, jd=jd_perf)
    perf_eval_2 = evaluator.evaluate(perfect_matches, resume=resume_perf, jd=jd_perf)

    # Score bounded at 100.0
    assert 0.0 <= perf_eval_1.overall_score <= 100.0
    assert perf_eval_1.recommendation == Recommendation.RECOMMENDED_FOR_SELECTION

    # Determinism: exact match across identical inputs
    assert perf_eval_1.overall_score == perf_eval_2.overall_score
    assert perf_eval_1.recommendation == perf_eval_2.recommendation
    assert perf_eval_1.score_breakdown_formula == perf_eval_2.score_breakdown_formula

# -------------------------------------------------------------
# 21. Exponential Backoff & Transient Error Recovery
# -------------------------------------------------------------
def test_groq_exponential_backoff_retry():
    import json
    from unittest.mock import MagicMock
    from modules.llm.evaluator import GroqContextualEvaluator
    from modules.llm.groq_client import GroqClientManager
    from modules.scoring.models import CandidateEvaluation

    mock_client_mgr = MagicMock(spec=GroqClientManager)
    mock_client_mgr.is_available = True
    mock_client_mgr.model = "qwen/qwen3.8-27b"
    mock_groq_client = MagicMock()
    mock_client_mgr.get_client.return_value = mock_groq_client

    evaluator = GroqContextualEvaluator(mock_client_mgr)

    # Simulate first attempt raising 429 RateLimitError, second attempt succeeding
    success_resp = MagicMock()
    success_resp.choices = [MagicMock(message=MagicMock(content=json.dumps({
        "overall_score": 88.0,
        "executive_summary": "Recovered fit after backoff.",
        "strengths": [],
        "weaknesses": [],
        "uncertain_areas": []
    })))]

    mock_groq_client.chat.completions.create.side_effect = [
        Exception("429: Too Many Requests - rate limit exceeded"),
        success_resp
    ]

    jd = JobDescription(id="jd_b", title="AI Engineer", raw_text="...", requirements=[])
    resume = ProcessedResume(filename="resume.pdf", verified_experience_years=3.0)
    det_eval = CandidateEvaluation(
        overall_score=75.0,
        essential_coverage_pct=80.0,
        preferred_coverage_pct=60.0,
        practical_evidence_score=80.0,
        experience_duration_score=80.0,
        project_depth_score=60.0,
        unsupported_keyword_penalty=0.0,
        verified_tenure_years=3.0,
        recommendation=Recommendation.RECOMMENDED_FOR_SELECTION,
        total_essential_count=4,
        matched_essential_count=3,
        total_preferred_count=2,
        matched_preferred_count=1,
        score_breakdown_formula="Formula",
        decision_rationale="Rationale"
    )

    result = evaluator.evaluate_candidate(jd, resume, [], det_eval)
    assert result["llm_applied"] is True
    assert result["overall_score"] == 88.0
    assert mock_groq_client.chat.completions.create.call_count == 2

# -------------------------------------------------------------
# 21. Education / Study Years Must NOT Be Counted as Work Tenure
# -------------------------------------------------------------
def test_education_dates_not_counted_as_work_experience(resume_extractor):
    # Candidate with B.Tech study dates (2020 - 2024) and High School (2018 - 2020)
    # Must yield 0.0 verified work experience years
    resume_student = ProcessedResume(filename="student_cv.pdf")
    resume_student.all_sentences = [
        EvidenceSentence(
            text="Bachelor of Technology in Computer Science, ABC University (2020 - 2024), CGPA 8.8",
            section=SectionType.EDUCATION
        ),
        EvidenceSentence(
            text="Senior Secondary School, CBSE Board (2018 - 2020), Percentage 92%",
            section=SectionType.EDUCATION
        ),
        EvidenceSentence(
            text="Developed an end-to-end fullstack e-commerce app using React and Node.js.",
            section=SectionType.PROJECTS
        )
    ]
    years = resume_extractor.calculate_verified_experience_years(resume_student)
    assert years == 0.0
    assert len(resume_student.employment_periods) == 0

# -------------------------------------------------------------
# 22. Intern Mode: 0% Weight on Experience Duration
# -------------------------------------------------------------
def test_intern_mode_scoring_zero_tenure_weight():
    intern_weights = ScoringWeights.intern_mode()
    assert intern_weights.relevant_experience_weight == 0.0
    assert intern_weights.llm_evaluation_weight == 0.30
    assert intern_weights.essential_skills_weight == 0.25
    assert intern_weights.practical_evidence_weight == 0.20
    assert intern_weights.project_depth_weight == 0.15
    assert intern_weights.preferred_skills_weight == 0.10
    assert abs(
        intern_weights.essential_skills_weight +
        intern_weights.preferred_skills_weight +
        intern_weights.practical_evidence_weight +
        intern_weights.relevant_experience_weight +
        intern_weights.project_depth_weight +
        intern_weights.llm_evaluation_weight - 1.0
    ) < 0.01

    evaluator = CandidateEvaluator(intern_weights)
    resume = ProcessedResume(filename="intern.pdf", verified_experience_years=0.0)

    # Candidate has all essential skills and strong practical projects, but 0.0 years work experience
    match_results = [
        RequirementMatchResult(
            requirement_id="r1",
            requirement_text="Proficiency with Python",
            priority="required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Python",
            canonical_skill="Python",
            resume_section=SectionType.PROJECTS,
            evidence_sentence="Built ML classification models in Python.",
            evidence_strength=EvidenceStrength.PROJECT_EVIDENCE,
            matching_method="exact"
        ),
        RequirementMatchResult(
            requirement_id="r2",
            requirement_text="Knowledge of SQL",
            priority="required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="SQL",
            canonical_skill="SQL",
            resume_section=SectionType.PROJECTS,
            evidence_sentence="Designed PostgreSQL database schemas and optimized SQL queries.",
            evidence_strength=EvidenceStrength.PROJECT_EVIDENCE,
            matching_method="exact"
        )
    ]

    jd = JobDescription(
        id="jd_intern",
        title="Software Intern",
        raw_text="...",
        requirements=[
            RequirementItem(id="r1", text="Proficiency with Python", req_type=ReqType.SKILL, priority=RequirementPriority.REQUIRED, min_years=1.0),
            RequirementItem(id="r2", text="Knowledge of SQL", req_type=ReqType.SKILL, priority=RequirementPriority.REQUIRED)
        ]
    )

    evaluation = evaluator.evaluate(match_results, resume=resume, jd=jd, llm_score=85.0)
    # 0 years tenure must NOT block selection under intern mode!
    assert evaluation.recommendation == Recommendation.RECOMMENDED_FOR_SELECTION
    assert evaluation.overall_score >= 68.0
    assert "Groq AI" in evaluation.decision_rationale

# -------------------------------------------------------------
# 23. Unrelated Technical Skills CANNOT Compensate for Missing Mandatory Requirements
# -------------------------------------------------------------
def test_unrelated_technical_skills_cannot_compensate_missing_mandatory():
    evaluator = CandidateEvaluator()

    # JD requires 4 critical mandatory competencies for ML Engineer
    # Candidate has 6 years general web dev experience (PHP, WordPress, CSS),
    # but only matched 1 peripheral skill (Git) and missed ML, PyTorch, Kubernetes.
    matches = [
        RequirementMatchResult(
            requirement_id="m1",
            requirement_text="Machine Learning",
            priority="Required",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="m2",
            requirement_text="PyTorch",
            priority="Required",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="m3",
            requirement_text="Kubernetes",
            priority="Required",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="m4",
            requirement_text="Git version control",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Git",
            evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION,
            resume_section=SectionType.EXPERIENCE,
            evidence_sentence="Used Git for team version control across 50 repositories."
        )
    ]
    resume = ProcessedResume(filename="web_dev.pdf", verified_experience_years=6.0)
    jd = JobDescription(
        id="jd_ml",
        title="ML Engineer",
        raw_text="...",
        requirements=[
            RequirementItem(id="m1", text="Machine Learning", req_type=ReqType.SKILL, priority=RequirementPriority.REQUIRED),
            RequirementItem(id="m2", text="PyTorch", req_type=ReqType.SKILL, priority=RequirementPriority.REQUIRED),
            RequirementItem(id="m3", text="Kubernetes", req_type=ReqType.SKILL, priority=RequirementPriority.REQUIRED),
            RequirementItem(id="m4", text="Git version control", req_type=ReqType.SKILL, priority=RequirementPriority.REQUIRED, min_years=2.0)
        ]
    )

    evaluation = evaluator.evaluate(matches, resume=resume, jd=jd)

    # Mandatory gating ceiling ensures candidate cannot pass
    assert evaluation.essential_coverage_pct == 25.0
    assert evaluation.overall_score <= 25.0
    assert evaluation.recommendation == Recommendation.NOT_RECOMMENDED
    assert "missing critical mandatory requirements" in evaluation.decision_rationale.lower()
    assert "cannot compensate" in evaluation.decision_rationale.lower()

# -------------------------------------------------------------
# 24. Project Quality Evaluates Implementation Depth AND Relevance to JD
# -------------------------------------------------------------
def test_project_quality_evaluates_depth_and_jd_relevance():
    evaluator = CandidateEvaluator()

    # Candidate 1: Built a project directly implementing mandatory JD skills with strong depth
    proj_matches_relevant = [
        RequirementMatchResult(
            requirement_id="p1",
            requirement_text="FastAPI backend",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="FastAPI",
            evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION,
            resume_section=SectionType.PROJECTS,
            evidence_sentence="Built and deployed high throughput FastAPI microservice handling 500 RPS."
        ),
        RequirementMatchResult(
            requirement_id="p2",
            requirement_text="PostgreSQL database",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="PostgreSQL",
            evidence_strength=EvidenceStrength.PROJECT_EVIDENCE,
            resume_section=SectionType.PROJECTS,
            evidence_sentence="Engineered optimized PostgreSQL database schemas with indexing."
        )
    ]
    eval_relevant = evaluator.evaluate(proj_matches_relevant)

    # Candidate 2: Only has 1 preferred skill in project with listed-only evidence
    proj_matches_weak = [
        RequirementMatchResult(
            requirement_id="p1",
            requirement_text="FastAPI backend",
            priority="Required",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="pref1",
            requirement_text="Redis",
            priority="Preferred",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Redis",
            evidence_strength=EvidenceStrength.LISTED_ONLY,
            resume_section=SectionType.PROJECTS
        )
    ]
    eval_weak = evaluator.evaluate(proj_matches_weak)

    assert eval_relevant.project_depth_score > eval_weak.project_depth_score
    assert eval_relevant.project_depth_score >= 80.0
    assert eval_weak.project_depth_score <= 40.0

# -------------------------------------------------------------
# 25. Unclear Evidence & Extraction Failure Route to Manual Review
# -------------------------------------------------------------
def test_unclear_evidence_and_extraction_failure_route_to_manual_review():
    evaluator = CandidateEvaluator()

    # Case A: Ambiguous evidence in mandatory requirement
    unclear_matches = [
        RequirementMatchResult(
            requirement_id="u1",
            requirement_text="Cloud Infrastructure",
            priority="Required",
            classification=MatchClassification.NEEDS_MANUAL_REVIEW,
            evidence_strength=EvidenceStrength.MENTIONED,
            explanation="Unclear if candidate managed production cloud or university lab environment."
        )
    ]
    eval_unclear = evaluator.evaluate(unclear_matches)
    assert eval_unclear.recommendation == Recommendation.MANUAL_REVIEW_REQUIRED
    assert "ambiguous" in eval_unclear.decision_rationale.lower() or "manual review" in eval_unclear.decision_rationale.lower()

    # Case B: PDF extraction failure / low confidence
    failed_resume = ProcessedResume(filename="scanned_unreadable.pdf", is_low_confidence=True)
    eval_failed = evaluator.evaluate([], resume=failed_resume)
    assert eval_failed.recommendation == Recommendation.MANUAL_REVIEW_REQUIRED
    assert eval_failed.is_low_confidence_extraction is True

# -------------------------------------------------------------
# 26. Composite Section Headings Recognition & Evidence Provenance
# -------------------------------------------------------------
def test_composite_section_heading_recognition(resume_extractor):
    raw_text = """John Doe
Software Engineer

TECHNICAL SKILLS
Python, Docker, SQL, Git

PROJECTS & EXPERIENCE
Software Engineer at Acme Corp (2021 - 2024)
• Architected high-throughput microservices using Python and Docker.
• Deployed scalable services on Kubernetes clusters.

AI Agent Platform (Personal Project)
• Built an autonomous ReAct AI agent using LangChain and FastAPI.
• Optimized latency and reduced API costs by 35%.

EDUCATION
B.S. in Computer Science (2017 - 2021)
"""
    processed = ProcessedResume(filename="composite.pdf", cleaned_text=raw_text, raw_text=raw_text)
    resume_extractor._segment_sections(processed)

    # Experience section must contain the employment role
    exp_sentences = processed.get_sentences_by_section(SectionType.EXPERIENCE)
    assert len(exp_sentences) >= 2
    # Projects section must contain the project sentences
    proj_sentences = processed.get_sentences_by_section(SectionType.PROJECTS)
    assert len(proj_sentences) >= 2
    # Verified experience must capture tenure
    assert processed.verified_experience_years >= 2.5

# -------------------------------------------------------------
# 27. JD Preamble & Responsibilities Non-Blocker Prioritization
# -------------------------------------------------------------
def test_jd_preamble_and_responsibilities_handling(jd_parser):
    jd_text = """Senior Machine Learning Engineer
About Acme AI:
We are a leading AI startup revolutionizing healthcare with generative models.
We are looking for a passionate Senior ML Engineer to join our collaborative team.

Responsibilities:
• Collaborate with cross-functional product and clinical teams.
• Design, evaluate, and maintain production inference pipelines.
• Participate in daily standups and code reviews.

Minimum Requirements:
• 4+ years of Python development experience
• Strong expertise in PyTorch or TensorFlow

Preferred Qualifications:
• Experience with Docker and Kubernetes
"""
    jd = jd_parser.parse(jd_text, jd_id="ai_preamble_test")

    # Preamble lines ("About Acme AI", "We are a leading AI startup") must NOT be parsed as requirements
    req_texts = [r.text.lower() for r in jd.requirements]
    assert not any("leading ai startup" in t for t in req_texts)
    assert not any("join our collaborative team" in t for t in req_texts)

    # Responsibilities must be tagged as RESPONSIBILITY with PREFERRED priority and weight 0.5
    resp_reqs = [r for r in jd.requirements if r.req_type == ReqType.RESPONSIBILITY]
    assert len(resp_reqs) >= 2
    for r in resp_reqs:
        assert r.priority == RequirementPriority.PREFERRED
        assert r.weight == 0.5

    # Only core technical competencies under Minimum Requirements act as REQUIRED
    mandatory_reqs = jd.get_requirements_by_priority(RequirementPriority.REQUIRED)
    assert len(mandatory_reqs) == 2
    assert any("python" in r.text.lower() for r in mandatory_reqs)
    assert any("pytorch" in r.text.lower() for r in mandatory_reqs)

# -------------------------------------------------------------
# 28. Semantic False Positive Prevention (Domain Concept Verification)
# -------------------------------------------------------------
def test_semantic_false_positive_prevention(matching_engine):
    # Candidate with database indexing skills in PostgreSQL
    resume = ProcessedResume(filename="db_dev.pdf")
    resume.all_sentences = [
        EvidenceSentence(
            text="Strong database optimization skills in PostgreSQL with index tuning and query planning.",
            section=SectionType.EXPERIENCE
        )
    ]

    # Requirement 1: NLP / Deep Learning
    req_nlp = RequirementItem(
        id="nlp_1",
        text="Strong expertise in Natural Language Processing (NLP) or Deep Learning",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="Natural Language Processing"
    )
    result_nlp = matching_engine.match_single_requirement(req_nlp, resume)
    # Must NOT match PostgreSQL as a strong semantic match!
    assert result_nlp.classification == MatchClassification.NOT_FOUND

    # Requirement 2: PyTorch / TensorFlow
    req_pytorch = RequirementItem(
        id="pt_1",
        text="Hands-on proficiency with PyTorch or TensorFlow frameworks",
        req_type=ReqType.SKILL,
        priority=RequirementPriority.REQUIRED,
        canonical_skill="PyTorch"
    )
    result_pt = matching_engine.match_single_requirement(req_pytorch, resume)
    # Must NOT match PostgreSQL as a strong semantic match!
    assert result_pt.classification == MatchClassification.NOT_FOUND

# -------------------------------------------------------------
# 29. Unqualified Candidates Default to NOT_RECOMMENDED
# -------------------------------------------------------------
def test_unqualified_candidates_fall_to_not_recommended():
    evaluator = CandidateEvaluator()

    # Candidate with 3 required skills, matches only 1 peripheral skill, missing 2 core skills
    matches = [
        RequirementMatchResult(
            requirement_id="r1",
            requirement_text="Python",
            priority="Required",
            classification=MatchClassification.EXACT_MATCH,
            matched_term="Python",
            evidence_strength=EvidenceStrength.STRONG_IMPLEMENTATION,
            resume_section=SectionType.EXPERIENCE,
            evidence_sentence="Developed backend scripts in Python."
        ),
        RequirementMatchResult(
            requirement_id="r2",
            requirement_text="PyTorch deep learning",
            priority="Required",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        ),
        RequirementMatchResult(
            requirement_id="r3",
            requirement_text="Distributed Systems Architecture",
            priority="Required",
            classification=MatchClassification.NOT_FOUND,
            evidence_strength=EvidenceStrength.NONE
        )
    ]
    resume = ProcessedResume(filename="weak.pdf", verified_experience_years=2.0)
    evaluation = evaluator.evaluate(matches, resume=resume)

    # Must be firmly NOT_RECOMMENDED, NEVER default to MANUAL_REVIEW_REQUIRED
    assert evaluation.recommendation == Recommendation.NOT_RECOMMENDED
    assert evaluation.overall_score <= 45.0
    assert "not recommended" in evaluation.decision_rationale.lower()



