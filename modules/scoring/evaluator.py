from typing import List, Optional, Dict, Any
from .models import (
    ScoringWeights,
    CandidateEvaluation,
    Recommendation
)
from ..matching_engine.models import (
    RequirementMatchResult,
    MatchClassification,
    EvidenceStrength
)
from ..resume_processing.models import SectionType

# ---------------------------------------------------------------------------
# Classification → credit mapping
# For mandatory requirements: how much "coverage credit" does each result give?
# ---------------------------------------------------------------------------
ESSENTIAL_CLASSIFICATION_CREDITS = {
    MatchClassification.EXACT_MATCH: 1.0,
    MatchClassification.STRONG_SEMANTIC_MATCH: 0.90,
    MatchClassification.FUZZY_MATCH: 0.80,
    MatchClassification.RELATED_SKILL_NOT_EQUIVALENT: 0.0,   # Shows transferable, not equivalent
    MatchClassification.PARTIAL_EVIDENCE: 0.35,
    MatchClassification.NEEDS_MANUAL_REVIEW: 0.50,
    MatchClassification.NOT_FOUND: 0.0
}

PREFERRED_CLASSIFICATION_CREDITS = {
    MatchClassification.EXACT_MATCH: 1.0,
    MatchClassification.STRONG_SEMANTIC_MATCH: 0.90,
    MatchClassification.FUZZY_MATCH: 0.80,
    MatchClassification.RELATED_SKILL_NOT_EQUIVALENT: 0.35,
    MatchClassification.PARTIAL_EVIDENCE: 0.30,
    MatchClassification.NEEDS_MANUAL_REVIEW: 0.50,
    MatchClassification.NOT_FOUND: 0.0
}

# Evidence strength → quality weight (how well-demonstrated is this match?)
EVIDENCE_QUALITY_WEIGHTS = {
    EvidenceStrength.STRONG_IMPLEMENTATION: 1.0,
    EvidenceStrength.EXPERIENCE_EVIDENCE: 0.80,
    EvidenceStrength.PROJECT_EVIDENCE: 0.70,
    EvidenceStrength.MENTIONED: 0.40,
    EvidenceStrength.LISTED_ONLY: 0.20,
    EvidenceStrength.NONE: 0.0
}

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

# Requirement types considered "behavioural" — their absence from a resume is
# UNKNOWN (candidate may have the quality but just didn't write about it).
# They contribute to score but must NEVER independently gate a rejection.
BEHAVIOURAL_REQ_TYPES = {"Behavioural", "Responsibility"}


class CandidateEvaluator:
    """
    Universal evaluation engine for all job roles and candidates.

    Philosophy:
    - Evaluate the WHOLE candidate profile against the WHOLE job.
    - Hard technical requirements are the primary signal.
    - Behavioural/soft requirements are real but unverifiable from resume text alone;
      their absence is UNKNOWN, not a confirmed gap.
    - Evidence quality (demonstrated vs. listed) matters.
    - A single missing requirement never auto-rejects; the overall picture does.
    - Score thresholds drive the recommendation, not cascading rule-chains.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()
        self.weights.validate()

    def evaluate(
        self,
        match_results: List[RequirementMatchResult],
        resume: Optional[Any] = None,
        jd: Optional[Any] = None,
        llm_score: Optional[float] = None,
        llm_recommendation: Optional[str] = None,
        llm_synthesis: Optional[str] = None
    ) -> CandidateEvaluation:

        is_low_conf = getattr(resume, "is_low_confidence", False) if resume else False

        # ── Extraction / API system failure ─────────────────────────────────
        # Kept strictly separate from candidate decisions.
        if is_low_conf or not match_results:
            return CandidateEvaluation(
                overall_score=0.0,
                essential_coverage_pct=0.0,
                preferred_coverage_pct=0.0,
                practical_evidence_score=0.0,
                experience_duration_score=0.0,
                project_depth_score=0.0,
                unsupported_keyword_penalty=0.0,
                verified_tenure_years=getattr(resume, "verified_experience_years", 0.0) if resume else 0.0,
                recommendation=Recommendation.UNABLE_TO_EVALUATE,
                total_essential_count=0,
                matched_essential_count=0,
                total_preferred_count=0,
                matched_preferred_count=0,
                is_low_confidence_extraction=is_low_conf,
                is_system_failure=True,
                total_mandatory_count=0,
                matched_mandatory_count=0,
                total_core_count=0,
                matched_core_count=0,
                demonstrated_essential_count=0,
                listed_only_essential_count=0,
                score_breakdown_formula="Evaluation incomplete.",
                decision_rationale=(
                    "[System] Resume extraction confidence too low to score reliably. "
                    "Re-upload a readable PDF or run OCR before rescreening."
                    if is_low_conf else
                    "[System] No JD requirements were matched — cannot produce a decision."
                ),
                llm_score=llm_score,
                llm_recommendation=llm_recommendation,
                llm_synthesis=llm_synthesis,
                dynamic_criteria_scores={}
            )

        # ── 1. PARTITION ─────────────────────────────────────────────────────
        # Every requirement maps to exactly ONE bucket. No double-counting.
        ESSENTIAL_PRIO = frozenset(("mandatory", "required", "core"))

        hard_essential  = [r for r in match_results
                           if r.priority.lower() in ESSENTIAL_PRIO
                           and r.req_type not in BEHAVIOURAL_REQ_TYPES]
        soft_essential  = [r for r in match_results
                           if r.priority.lower() in ESSENTIAL_PRIO
                           and r.req_type in BEHAVIOURAL_REQ_TYPES]
        hard_mandatory  = [r for r in hard_essential
                           if r.priority.lower() in ("mandatory", "required")]
        hard_core       = [r for r in hard_essential
                           if r.priority.lower() == "core"]
        preferred       = [r for r in match_results if r.priority.lower() == "preferred"]
        optional        = [r for r in match_results if r.priority.lower() == "optional"]

        total_hard_essential  = len(hard_essential)
        total_mandatory_count = len(hard_mandatory)
        total_core_count      = len(hard_core)
        total_pref            = len(preferred) + len(optional)
        total_essential       = total_hard_essential + len(soft_essential)   # for UI display

        # ── 2. PER-REQUIREMENT CREDIT WITH EVIDENCE MULTIPLIER ───────────────
        # credit  = how well the classification matches (exact, fuzzy, partial, none)
        # ev_mult = how well the evidence demonstrates the skill (not just listed)
        # score   = credit × ev_mult × fulfillment_ratio
        #
        # LISTED_ONLY for a mandatory/core requirement gets only 35% — the JD
        # demands demonstrated competency, not keyword appearance.
        # Transferable skills (RELATED_NOT_EQUIVALENT) get 0 credit for essential
        # requirements — they do not satisfy the requirement.

        def _ev_mult(r: RequirementMatchResult, is_essential: bool) -> float:
            if r.evidence_strength in DEMONSTRATED_EVIDENCE:
                return 1.0
            if r.evidence_strength == EvidenceStrength.MENTIONED:
                return 0.60
            if r.evidence_strength == EvidenceStrength.LISTED_ONLY:
                return 0.35 if is_essential else 0.55
            return 1.0  # NOT_FOUND / NONE already score 0 via credit

        def _score(r: RequirementMatchResult, is_essential: bool) -> float:
            credit_map = (ESSENTIAL_CLASSIFICATION_CREDITS
                          if is_essential else PREFERRED_CLASSIFICATION_CREDITS)
            credit      = credit_map.get(r.classification, 0.0)
            fulfillment = r.fulfillment_ratio if r.fulfillment_ratio > 0 else (
                1.0 if r.classification in CONFIRMED_CLASSIFICATIONS else 0.0
            )
            return credit * _ev_mult(r, is_essential) * fulfillment

        # ── 3. COVERAGE SCORES ───────────────────────────────────────────────
        hard_ess_earned = sum(_score(r, True)  for r in hard_essential)
        soft_ess_earned = sum(_score(r, False) for r in soft_essential)   # soft treated as preferred credit
        pref_earned     = sum(_score(r, False) for r in preferred)
        opt_earned      = sum(_score(r, False) * 0.5 for r in optional)

        hard_essential_pct = (hard_ess_earned / total_hard_essential * 100.0) if total_hard_essential > 0 else 100.0
        all_essential_pct  = (
            (hard_ess_earned + soft_ess_earned) / total_essential * 100.0
        ) if total_essential > 0 else 100.0
        preferred_pct = (
            (pref_earned + opt_earned) / max(1, total_pref) * 100.0
        ) if total_pref > 0 else 100.0

        # Confirmed = classification is a match (exact, semantic, fuzzy)
        confirmed_essential    = [r for r in hard_essential if r.classification in CONFIRMED_CLASSIFICATIONS]
        matched_essential_count = len(confirmed_essential)
        matched_mandatory_count = sum(1 for r in hard_mandatory if r.classification in CONFIRMED_CLASSIFICATIONS)
        matched_core_count      = sum(1 for r in hard_core      if r.classification in CONFIRMED_CLASSIFICATIONS)
        matched_pref_count      = sum(1 for r in preferred      if r.classification in CONFIRMED_CLASSIFICATIONS)

        demonstrated_essential = [r for r in confirmed_essential if r.evidence_strength in DEMONSTRATED_EVIDENCE]
        listed_only_essential  = [r for r in confirmed_essential if r.evidence_strength not in DEMONSTRATED_EVIDENCE]

        # Missing / uncertain — only counted on hard essential (not behavioural)
        missing_critical   = [r for r in hard_essential if r.classification == MatchClassification.NOT_FOUND]
        uncertain_essential = [r for r in hard_essential if r.classification == MatchClassification.NEEDS_MANUAL_REVIEW]
        # Transferable: related skill exists but does NOT satisfy the requirement
        transferable_skills = [
            {"required": r.requirement_text, "candidate_has": r.matched_term, "explanation": r.explanation}
            for r in hard_essential
            if r.classification == MatchClassification.RELATED_SKILL_NOT_EQUIVALENT
        ]

        n_missing        = len(missing_critical)
        n_uncertain      = len(uncertain_essential)
        n_transferable   = len(transferable_skills)

        # ── 4. PRACTICAL EVIDENCE QUALITY ────────────────────────────────────
        # Weighted average of evidence quality over ALL confirmed requirements,
        # grounded by what fraction of hard essentials were actually demonstrated.
        confirmed_all   = [r for r in match_results if r.classification in CONFIRMED_CLASSIFICATIONS]
        demonstrated_all = [r for r in confirmed_all if r.evidence_strength in DEMONSTRATED_EVIDENCE]
        listed_only_all  = [r for r in confirmed_all if r.evidence_strength not in DEMONSTRATED_EVIDENCE
                            and r.evidence_strength != EvidenceStrength.NONE]

        if demonstrated_all or listed_only_all:
            demo_sum   = sum(EVIDENCE_QUALITY_WEIGHTS.get(r.evidence_strength, 0.0) for r in demonstrated_all)
            listed_sum = sum(EVIDENCE_QUALITY_WEIGHTS.get(r.evidence_strength, 0.0) * 0.4 for r in listed_only_all)
            avg_ev     = (demo_sum + listed_sum) / max(1, len(demonstrated_all) + len(listed_only_all))
            if total_hard_essential > 0:
                ground = len([r for r in demonstrated_essential
                               if r.req_type not in BEHAVIOURAL_REQ_TYPES]) / total_hard_essential
                practical_score = ((avg_ev * 0.65) + (ground * 0.35)) * 100.0
            else:
                practical_score = avg_ev * 100.0
        else:
            practical_score = 0.0

        # ── 5. PROJECT RELEVANCE SCORE ───────────────────────────────────────
        # Only project/experience evidence that maps to a hard essential JD requirement
        # gets credit. Unrelated projects do not contribute.
        relevant_hands_on = [
            r for r in confirmed_all
            if r.classification in CONFIRMED_CLASSIFICATIONS
            and r.evidence_strength in DEMONSTRATED_EVIDENCE
            and r.priority.lower() in ESSENTIAL_PRIO
            and r.req_type not in BEHAVIOURAL_REQ_TYPES
            and (r.resume_section in (SectionType.PROJECTS, SectionType.EXPERIENCE)
                 or r.evidence_strength in (EvidenceStrength.PROJECT_EVIDENCE,
                                            EvidenceStrength.STRONG_IMPLEMENTATION))
        ]
        if relevant_hands_on:
            depth_scores = [EVIDENCE_QUALITY_WEIGHTS.get(r.evidence_strength, 0.7) for r in relevant_hands_on]
            avg_depth    = sum(depth_scores) / len(depth_scores)
            relevance    = min(1.0, len(relevant_hands_on) / max(1, min(total_hard_essential, 4)))
            project_score = min(100.0, ((avg_depth * 0.40) + (relevance * 0.60)) * 100.0)
        else:
            project_score = 0.0

        # ── 6. EXPERIENCE DURATION ───────────────────────────────────────────
        verified_tenure = getattr(resume, "verified_experience_years", 0.0) if resume else 0.0
        min_years_req   = None
        if jd and hasattr(jd, "requirements"):
            exp_years = [r.min_years for r in jd.requirements
                         if getattr(r, "min_years", None) and r.min_years > 0]
            if exp_years:
                min_years_req = max(exp_years)

        jd_requires_experience = bool(min_years_req)
        if jd_requires_experience:
            exp_duration_score = min(100.0, (verified_tenure / min_years_req) * 100.0)
            tenure_satisfied   = verified_tenure >= (min_years_req - 0.25)
        else:
            exp_duration_score = 100.0
            tenure_satisfied   = True

        # ── 7. COMPOSITE SCORE ───────────────────────────────────────────────
        # Dynamic weights from JD parser take priority.
        # Falls back to uniform formula if weights are absent.
        has_dynamic_weights = any(r.weight > 0 for r in match_results)
        dynamic_criteria_scores: Dict[str, float] = {}

        if has_dynamic_weights:
            total_dyn_w = sum(r.weight for r in match_results) or 1.0
            dyn_sum     = 0.0
            for r in match_results:
                is_ess    = r.priority.lower() in ESSENTIAL_PRIO
                w_norm    = r.weight / total_dyn_w
                item_s    = _score(r, is_ess) * 100.0
                dyn_sum  += item_s * w_norm
                dynamic_criteria_scores[r.requirement_id] = round(item_s, 1)
            base_overall = (dyn_sum * 0.60) + (practical_score * 0.25) + (project_score * 0.15)
        else:
            w_exp = 0.15 if jd_requires_experience else 0.0
            w_ess = 0.45 + (0.0 if jd_requires_experience else 0.15)
            base_overall = (
                (all_essential_pct * w_ess)
                + (preferred_pct   * 0.15)
                + (practical_score * 0.25)
                + (exp_duration_score * w_exp)
                + (project_score   * 0.15)
            )

        if llm_score is not None and getattr(self.weights, "llm_evaluation_weight", 0.0) > 0:
            w_llm   = min(0.30, self.weights.llm_evaluation_weight)
            overall = (base_overall * (1.0 - w_llm)) + (llm_score * w_llm)
        else:
            overall = base_overall

        # ── 8. SCORE CEILING ─────────────────────────────────────────────────
        # Prevent unrelated/transferable skills from inflating the score past what
        # actual JD-relevant competency justifies.
        if total_hard_essential > 0:
            if hard_essential_pct < 50.0:
                overall = min(overall, hard_essential_pct, 44.0)
            elif hard_essential_pct < 65.0:
                overall = min(overall, hard_essential_pct + 10.0, 64.0)
            # Zero demonstrated essentials: no practical competency shown at all
            if len(demonstrated_essential) == 0 and total_hard_essential >= 2:
                overall = min(overall, 30.0)

        # Explicit eligibility gates (degree, clearance, etc.)
        failed_eligibility = [
            r for r in match_results
            if getattr(r, "is_eligibility_condition", False)
            and r.classification == MatchClassification.NOT_FOUND
        ]
        if failed_eligibility:
            overall = min(overall, 30.0)

        overall = min(100.0, max(0.0, overall))

        # ── 9. RECOMMENDATION ────────────────────────────────────────────────
        # Three outcomes for candidates (system failures already returned above):
        #   RECOMMENDED_FOR_SELECTION  — strong, evidenced match
        #   FURTHER_REVIEW             — borderline; needs secondary screening
        #   NOT_RECOMMENDED            — clear mismatch on critical requirements
        #
        # Rules:
        #   • Transferable skills do NOT compensate for missing core requirements.
        #   • Rejection gates apply only to hard essential requirements + clean extraction.
        #   • Behavioural / soft misses never trigger rejection.
        #   • FURTHER_REVIEW is an automated outcome, not a human-review referral.

        if failed_eligibility:
            elig_names = [r.requirement_text for r in failed_eligibility[:2]]
            rec = Recommendation.NOT_RECOMMENDED
            decision_caveat = (
                f"Not Recommended: Fails non-negotiable eligibility criteria "
                f"({', '.join(elig_names)})."
            )

        elif total_hard_essential > 0 and (
            # Missing more than half the hard essential requirements — clear mismatch
            n_missing > (total_hard_essential / 2)
            # Nothing confirmed and nothing uncertain (transferable alone is NOT enough)
            or (matched_essential_count == 0 and n_uncertain == 0)
            # Score below floor — transferable skills present but don't close gap
            or (overall < 40.0)
        ):
            rec = Recommendation.NOT_RECOMMENDED
            miss_names = [r.requirement_text for r in missing_critical[:3]]
            decision_caveat = (
                f"Not Recommended: Insufficient match on critical competencies "
                f"(score {overall:.0f}/100; missing: {', '.join(miss_names) if miss_names else f'{n_missing} core requirements'})."
            )

        elif (
            hard_essential_pct >= 70.0
            and overall >= 68.0
            and tenure_satisfied
            and n_missing == 0
            and len(demonstrated_essential) >= max(1, total_hard_essential // 2)
        ):
            rec = Recommendation.RECOMMENDED_FOR_SELECTION
            tenure_note = f", {verified_tenure:.1f} yrs exp (required {min_years_req})" if jd_requires_experience else ""
            decision_caveat = (
                f"Recommended: {matched_essential_count}/{total_hard_essential} essential requirements matched "
                f"({hard_essential_pct:.0f}% coverage), {len(demonstrated_essential)} demonstrated in work/projects"
                f"{tenure_note}."
            )

        else:
            # Borderline: some requirements met, some uncertain, or score in mid-range.
            rec = Recommendation.FURTHER_REVIEW
            reasons = []
            if n_uncertain > 0:
                u_names = [r.requirement_text for r in uncertain_essential[:2]]
                reasons.append(f"Ambiguous evidence for: {', '.join(u_names)}")
            if n_transferable > 0 and n_missing > 0:
                reasons.append(f"{n_transferable} transferable skill(s) present but {n_missing} core requirement(s) not satisfied")
            if jd_requires_experience and not tenure_satisfied:
                reasons.append(f"Experience {verified_tenure:.1f} yrs vs required {min_years_req}")
            if not reasons:
                reasons.append(f"Overall fit {overall:.0f}/100 — borderline match requiring secondary screening")
            decision_caveat = f"Further Review: {'; '.join(reasons)}."

        # ── 10. RATIONALE ────────────────────────────────────────────────────
        formula = (
            f"Score={overall:.1f}/100 "
            f"[HardEssential={hard_essential_pct:.0f}%, "
            f"PracticalEvidence={practical_score:.0f}%, "
            f"ProjectRelevance={project_score:.0f}%]"
        )

        rationale_parts = [decision_caveat]
        rationale_parts.append(
            f"Hard essential: {matched_essential_count}/{total_hard_essential} matched "
            f"({len(demonstrated_essential)} demonstrated, {len(listed_only_essential)} listed-only)."
        )
        if missing_critical:
            rationale_parts.append(
                f"Missing: {', '.join(r.requirement_text for r in missing_critical[:3])}."
            )
        if transferable_skills:
            transfer_items = [f"{t['required']} -> {t['candidate_has']}" for t in transferable_skills[:2]]
            rationale_parts.append(f"Transferable (not equivalent): {'; '.join(transfer_items)}.")
        rationale = " ".join(rationale_parts)

        return CandidateEvaluation(
            overall_score=overall,
            essential_coverage_pct=all_essential_pct,
            preferred_coverage_pct=preferred_pct,
            practical_evidence_score=practical_score,
            experience_duration_score=exp_duration_score,
            project_depth_score=project_score,
            unsupported_keyword_penalty=0.0,   # absorbed into ev_mult; no separate penalty
            verified_tenure_years=verified_tenure,
            min_years_required=min_years_req,
            recommendation=rec,
            total_essential_count=total_essential,
            matched_essential_count=matched_essential_count,
            total_preferred_count=total_pref,
            matched_preferred_count=matched_pref_count,
            transferable_skills=transferable_skills,
            is_low_confidence_extraction=is_low_conf,
            is_system_failure=False,
            total_mandatory_count=total_mandatory_count,
            matched_mandatory_count=matched_mandatory_count,
            total_core_count=total_core_count,
            matched_core_count=matched_core_count,
            demonstrated_essential_count=len(demonstrated_essential),
            listed_only_essential_count=len(listed_only_essential),
            score_breakdown_formula=formula,
            decision_rationale=rationale,
            llm_score=llm_score,
            llm_recommendation=llm_recommendation,
            llm_synthesis=llm_synthesis,
            dynamic_criteria_scores=dynamic_criteria_scores
        )

