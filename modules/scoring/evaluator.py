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

ESSENTIAL_CLASSIFICATION_CREDITS = {
    MatchClassification.EXACT_MATCH: 1.0,
    MatchClassification.STRONG_SEMANTIC_MATCH: 0.90,
    MatchClassification.FUZZY_MATCH: 0.80,
    # Related skills earn 0% essential coverage, but are flagged as transferable
    MatchClassification.RELATED_SKILL_NOT_EQUIVALENT: 0.0,
    MatchClassification.PARTIAL_EVIDENCE: 0.30,
    MatchClassification.NEEDS_MANUAL_REVIEW: 0.50,
    MatchClassification.NOT_FOUND: 0.0
}

PREFERRED_CLASSIFICATION_CREDITS = {
    MatchClassification.EXACT_MATCH: 1.0,
    MatchClassification.STRONG_SEMANTIC_MATCH: 0.90,
    MatchClassification.FUZZY_MATCH: 0.80,
    # Related skills can earn partial transferability credit for preferred requirements
    MatchClassification.RELATED_SKILL_NOT_EQUIVALENT: 0.35,
    MatchClassification.PARTIAL_EVIDENCE: 0.30,
    MatchClassification.NEEDS_MANUAL_REVIEW: 0.50,
    MatchClassification.NOT_FOUND: 0.0
}

PRACTICAL_EVIDENCE_WEIGHTS = {
    EvidenceStrength.STRONG_IMPLEMENTATION: 1.0,
    EvidenceStrength.EXPERIENCE_EVIDENCE: 0.80,
    EvidenceStrength.PROJECT_EVIDENCE: 0.70,
    EvidenceStrength.MENTIONED: 0.40,
    EvidenceStrength.LISTED_ONLY: 0.20,
    EvidenceStrength.NONE: 0.0
}

# Evidence strengths that represent actual demonstrated work (not just listed/mentioned in a skills section)
DEMONSTRATED_EVIDENCE = frozenset({
    EvidenceStrength.STRONG_IMPLEMENTATION,
    EvidenceStrength.EXPERIENCE_EVIDENCE,
    EvidenceStrength.PROJECT_EVIDENCE,
})

# Confirmed match classifications (skill is genuinely present in resume)
CONFIRMED_CLASSIFICATIONS = frozenset({
    MatchClassification.EXACT_MATCH,
    MatchClassification.STRONG_SEMANTIC_MATCH,
    MatchClassification.FUZZY_MATCH,
})


class CandidateEvaluator:
    """
    Computes explainable 5-factor candidate scores based on configurable weights for
    essential skills, preferred skills, practical demonstration, verified experience
    duration, and project implementation depth and JD relevance.

    Decision rules enforce strict priority:
      1. Low-confidence extraction → Manual Review
      2. Clear mandatory gap (majority NOT_FOUND) → Not Recommended
      3. Ambiguous/uncertain evidence on mandatory items → Manual Review
      4. Comprehensive JD match → Recommended for Selection
      5. Borderline / transferable / tenure gap → Manual Review
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

        # --- Empty match results: exit early ---
        if not match_results:
            return CandidateEvaluation(
                overall_score=0.0,
                essential_coverage_pct=0.0,
                preferred_coverage_pct=0.0,
                practical_evidence_score=0.0,
                experience_duration_score=0.0,
                project_depth_score=0.0,
                unsupported_keyword_penalty=0.0,
                verified_tenure_years=getattr(resume, "verified_experience_years", 0.0) if resume else 0.0,
                recommendation=Recommendation.MANUAL_REVIEW_REQUIRED if is_low_conf else Recommendation.NOT_RECOMMENDED,
                total_essential_count=0,
                matched_essential_count=0,
                total_preferred_count=0,
                matched_preferred_count=0,
                is_low_confidence_extraction=is_low_conf,
                score_breakdown_formula="No requirements evaluated.",
                decision_rationale=(
                    "Low extraction confidence detected. Manual PDF review required before decision."
                    if is_low_conf else "No requirements provided for screening."
                ),
                llm_score=llm_score,
                llm_recommendation=llm_recommendation,
                llm_synthesis=llm_synthesis
            )

        req_results = [r for r in match_results if r.priority.lower() == "required"]
        pref_results = [r for r in match_results if r.priority.lower() == "preferred"]

        # Transferable non-equivalent skills: recognized but NOT direct essential matches
        transferable_skills = [
            {"required": r.requirement_text, "candidate_has": r.matched_term, "explanation": r.explanation}
            for r in match_results
            if r.classification == MatchClassification.RELATED_SKILL_NOT_EQUIVALENT
        ]
        has_transferable = len(transferable_skills) > 0

        # ---------------------------------------------------------------
        # 1. ESSENTIAL SKILLS COVERAGE
        #    Strict: transferable skills get 0.0 credit
        # ---------------------------------------------------------------
        total_essential = len(req_results)
        req_earned = sum(ESSENTIAL_CLASSIFICATION_CREDITS.get(r.classification, 0.0) for r in req_results)
        essential_pct = (req_earned / total_essential * 100.0) if total_essential > 0 else 100.0

        # Confirmed matches (exact / semantic / fuzzy — skill is genuinely in the resume)
        confirmed_essential = [r for r in req_results if r.classification in CONFIRMED_CLASSIFICATIONS]
        matched_essential_count = len(confirmed_essential)

        # Confirmed essential skills demonstrated through real work/projects
        demonstrated_essential = [r for r in confirmed_essential if r.evidence_strength in DEMONSTRATED_EVIDENCE]
        # Confirmed essential skills that are ONLY listed (no work/project backing)
        listed_only_essential = [r for r in confirmed_essential if r.evidence_strength not in DEMONSTRATED_EVIDENCE]

        # Gap categorization
        missing_critical = [r for r in req_results if r.classification == MatchClassification.NOT_FOUND]
        transferable_essential = [r for r in req_results if r.classification == MatchClassification.RELATED_SKILL_NOT_EQUIVALENT]
        uncertain_essential = [r for r in req_results if r.classification == MatchClassification.NEEDS_MANUAL_REVIEW]

        # ---------------------------------------------------------------
        # 2. PREFERRED SKILLS COVERAGE
        # ---------------------------------------------------------------
        total_pref = len(pref_results)
        pref_earned = sum(PREFERRED_CLASSIFICATION_CREDITS.get(r.classification, 0.0) for r in pref_results)
        matched_pref_count = sum(1 for r in pref_results if r.classification in CONFIRMED_CLASSIFICATIONS)
        preferred_pct = (pref_earned / total_pref * 100.0) if total_pref > 0 else 100.0

        # ---------------------------------------------------------------
        # 3. PRACTICAL EVIDENCE SCORE
        #    Distinguishes actual demonstrated skills from merely listed ones.
        #    Skills only in a skills/list section receive a strong discount.
        #    The score reflects how much JD-relevant evidence the candidate
        #    has actually backed through real work or projects.
        # ---------------------------------------------------------------
        demonstrated_matches = [
            r for r in match_results
            if r.classification in CONFIRMED_CLASSIFICATIONS
            and r.evidence_strength in DEMONSTRATED_EVIDENCE
        ]
        listed_confirmed_matches = [
            r for r in match_results
            if r.classification in CONFIRMED_CLASSIFICATIONS
            and r.evidence_strength not in DEMONSTRATED_EVIDENCE
            and r.evidence_strength != EvidenceStrength.NONE
        ]

        if demonstrated_matches or listed_confirmed_matches:
            demo_score_sum = sum(PRACTICAL_EVIDENCE_WEIGHTS[r.evidence_strength] for r in demonstrated_matches)
            # Listed-only skills receive an additional 0.5x discount on top of PRACTICAL_EVIDENCE_WEIGHTS
            listed_score_sum = sum(
                PRACTICAL_EVIDENCE_WEIGHTS.get(r.evidence_strength, 0.0) * 0.5
                for r in listed_confirmed_matches
            )
            total_count = len(demonstrated_matches) + len(listed_confirmed_matches)
            avg_evidence = (demo_score_sum + listed_score_sum) / total_count

            if total_essential > 0:
                # Grounding ratio: fraction of mandatory requirements backed by demonstrated work
                grounding_ratio = len(demonstrated_essential) / total_essential
                # 70% quality of evidence, 30% essential grounding
                practical_score = ((avg_evidence * 0.70) + (grounding_ratio * 0.30)) * 100.0
            else:
                practical_score = avg_evidence * 100.0
        else:
            practical_score = 0.0

        # ---------------------------------------------------------------
        # 4. EXPERIENCE DURATION SCORE
        # ---------------------------------------------------------------
        verified_tenure = getattr(resume, "verified_experience_years", 0.0) if resume else 0.0

        # Active confirmed matches (for fallback exp_duration)
        active_matches = [r for r in match_results if r.classification in CONFIRMED_CLASSIFICATIONS]

        min_years_req = None
        if jd and hasattr(jd, "requirements"):
            exp_reqs = [r.min_years for r in jd.requirements if getattr(r, "min_years", None) is not None]
            if exp_reqs:
                min_years_req = max(exp_reqs)

        if min_years_req and min_years_req > 0:
            exp_duration_score = min(100.0, (verified_tenure / min_years_req) * 100.0)
        elif active_matches:
            exp_matches = sum(1 for r in active_matches if r.resume_section == SectionType.EXPERIENCE)
            exp_duration_score = (exp_matches / len(active_matches)) * 100.0
        else:
            exp_duration_score = 0.0

        # ---------------------------------------------------------------
        # 5. PROJECT QUALITY SCORE
        #    Requires actual project evidence (not listed-only).
        #    Scores both implementation depth AND relevance to JD mandatory skills.
        #    Unrelated projects or keyword-listed skills do NOT score here.
        # ---------------------------------------------------------------
        project_matches = [
            r for r in active_matches
            if (r.resume_section == SectionType.PROJECTS or r.evidence_strength == EvidenceStrength.PROJECT_EVIDENCE)
            and r.evidence_strength in DEMONSTRATED_EVIDENCE  # actual project depth required
        ]
        if project_matches:
            depth_scores = [PRACTICAL_EVIDENCE_WEIGHTS.get(r.evidence_strength, 0.5) for r in project_matches]
            avg_proj_depth = sum(depth_scores) / len(depth_scores)

            if total_essential > 0:
                mandatory_in_proj = sum(1 for r in project_matches if r.priority.lower() == "required")
                # Full relevance if ≥2 mandatory skills are demonstrated through projects
                proj_relevance = min(1.0, mandatory_in_proj / max(1, min(total_essential, 2)))
                project_score = min(100.0, ((avg_proj_depth * 0.60) + (proj_relevance * 0.40)) * 100.0)
            else:
                project_score = min(100.0, avg_proj_depth * 100.0)
        else:
            project_score = 0.0

        # ---------------------------------------------------------------
        # 6. UNSUPPORTED KEYWORD PENALTY
        #    Fired when ≥60% of essential skills are listed-only (no practical
        #    work/project backing) and practical evidence is still low.
        # ---------------------------------------------------------------
        unsupported_penalty = 0.0
        if req_results:
            essential_listed_only_count = sum(
                1 for r in req_results
                if r.classification in CONFIRMED_CLASSIFICATIONS
                and r.evidence_strength not in DEMONSTRATED_EVIDENCE
                and r.evidence_strength != EvidenceStrength.NONE
            )
            if (essential_listed_only_count / len(req_results)) >= 0.60 and practical_score < 35.0:
                unsupported_penalty = 8.0

        # ---------------------------------------------------------------
        # 7. COMPOSITE OVERALL SCORE
        # ---------------------------------------------------------------
        has_llm = (llm_score is not None and getattr(self.weights, "llm_evaluation_weight", 0.0) > 0)
        if has_llm:
            overall = (
                (essential_pct * self.weights.essential_skills_weight)
                + (preferred_pct * self.weights.preferred_skills_weight)
                + (practical_score * self.weights.practical_evidence_weight)
                + (exp_duration_score * self.weights.relevant_experience_weight)
                + (project_score * self.weights.project_depth_weight)
                + (llm_score * self.weights.llm_evaluation_weight)
                - unsupported_penalty
            )
        else:
            det_sum = (
                self.weights.essential_skills_weight
                + self.weights.preferred_skills_weight
                + self.weights.practical_evidence_weight
                + self.weights.relevant_experience_weight
                + self.weights.project_depth_weight
            )
            scale = (1.0 / det_sum) if det_sum > 0 else 1.0
            overall = (
                (essential_pct * self.weights.essential_skills_weight * scale)
                + (preferred_pct * self.weights.preferred_skills_weight * scale)
                + (practical_score * self.weights.practical_evidence_weight * scale)
                + (exp_duration_score * self.weights.relevant_experience_weight * scale)
                + (project_score * self.weights.project_depth_weight * scale)
                - unsupported_penalty
            )

        n_missing = len(missing_critical)
        n_transferable_req = len(transferable_essential)
        n_uncertain_req = len(uncertain_essential)
        extraction_clean = not is_low_conf

        # ---------------------------------------------------------------
        # MANDATORY COMPETENCY GATING
        # No amount of unrelated skills, tenure, or preferred coverage can
        # push a candidate past the mandatory coverage ceiling.
        # ---------------------------------------------------------------
        if total_essential > 0:
            if essential_pct < 50.0:
                # Hard ceiling: cannot exceed essential coverage or 44
                overall = min(overall, essential_pct, 44.0)
            elif essential_pct < 65.0:
                # Soft ceiling: cannot cross into Selection band without strong coverage
                overall = min(overall, essential_pct + 12.0, 64.0)

            # Enforce ceiling if critical mandatory requirements are missing without transferable alternative
            if n_missing >= 2 and n_transferable_req == 0:
                overall = min(overall, 44.0)
            elif n_missing >= 1 and n_transferable_req == 0:
                overall = min(overall, 58.0)

        overall = min(100.0, max(0.0, overall))

        # ---------------------------------------------------------------
        # 8. DETERMINISTIC DECISION RULES — STRICT PRIORITY ORDER
        #
        #  Rule A (highest priority): Extraction failure / low confidence
        #  Rule B: Clear mandatory gap / low qualification → Not Recommended
        #  Rule C: Uncertain evidence on mandatory items (genuinely ambiguous)
        #  Rule D: Strong comprehensive JD match → Recommended for Selection
        #  Rule E: Borderline / transferable / tenure gap → Manual Review
        #  Fallback else: → Not Recommended
        # ---------------------------------------------------------------
        is_tenure_exempt = (self.weights.relevant_experience_weight <= 0.01)
        tenure_satisfied = (
            is_tenure_exempt
            or min_years_req is None
            or verified_tenure >= (min_years_req - 0.25)
        )

        # --- A. Extraction failure ---
        if is_low_conf:
            rec = Recommendation.MANUAL_REVIEW_REQUIRED
            decision_caveat = (
                "Manual Review Required: Low extraction confidence or unreadable PDF text. "
                "Human verification is required before a hiring decision can be made."
            )

        # --- B. Clear mandatory gap → Not Recommended ---
        elif total_essential > 0 and extraction_clean and (
            # Majority of mandatory requirements completely missing
            n_missing > (total_essential / 2)
            # Two or more mandatory requirements missing without transferable alternatives
            or (n_missing >= 2 and n_transferable_req == 0 and n_uncertain_req == 0)
            # Zero confirmed essential matches, no transferable alternatives, and no uncertain items
            or (matched_essential_count == 0 and n_transferable_req == 0 and n_uncertain_req == 0)
            # Below 50% coverage, real gaps exist, no transferable or uncertain escape hatch
            or (essential_pct < 50.0 and n_missing > 0
                and n_transferable_req == 0 and n_uncertain_req == 0)
            # Low overall score with missing mandatory competencies
            or (overall < 45.0 and n_transferable_req == 0 and n_uncertain_req == 0)
            # Skills listed but NONE demonstrated in real work/projects, low practical evidence
            or (matched_essential_count > 0
                and len(demonstrated_essential) == 0
                and practical_score < 20.0
                and essential_pct < 65.0
                and n_transferable_req == 0)
        ):
            rec = Recommendation.NOT_RECOMMENDED
            missing_names = [r.requirement_text for r in missing_critical]
            decision_caveat = (
                f"Not Recommended: Missing critical mandatory requirements "
                f"({', '.join(missing_names[:3]) if missing_names else f'{matched_essential_count}/{total_essential} essential skills matched'}). "
                "General technical experience or unrelated skills cannot compensate for missing core job competencies."
            )

        # --- C. Uncertain evidence on mandatory items (genuinely ambiguous) ---
        elif uncertain_essential and extraction_clean and n_missing <= 1:
            rec = Recommendation.MANUAL_REVIEW_REQUIRED
            uncertain_names = [r.requirement_text for r in uncertain_essential[:2]]
            decision_caveat = (
                f"Manual Review Required: Evidence for mandatory requirement(s) "
                f"({', '.join(uncertain_names)}) is ambiguous or unverifiable. "
                "Human review of the original resume is recommended."
            )

        # --- D. Recommended for Selection ---
        elif (
            essential_pct >= 60.0
            and n_missing == 0
            and n_transferable_req == 0
            and practical_score >= 35.0
            and overall >= 68.0
            and tenure_satisfied
        ):
            rec = Recommendation.RECOMMENDED_FOR_SELECTION
            if has_llm:
                decision_caveat = (
                    f"Recommended for Selection: Strong JD match — "
                    f"{matched_essential_count}/{total_essential} essential skills ({essential_pct:.0f}%), "
                    f"practical evidence {practical_score:.0f}%, Groq AI assessment {llm_score:.0f}/100."
                )
            elif is_tenure_exempt:
                decision_caveat = (
                    f"Recommended for Selection: Strong match on essential skills ({essential_pct:.0f}%) "
                    f"and demonstrated project depth (Intern Mode; experience tenure not weighted)."
                )
            else:
                decision_caveat = (
                    f"Recommended for Selection: Comprehensive match on mandatory skills ({essential_pct:.0f}%), "
                    f"verified tenure ({verified_tenure} yrs), and grounded practical implementation ({practical_score:.0f}%)."
                )

        # --- E. Borderline / Transferable / Marginal Tenure gap → Manual Review ---
        elif (
            (has_transferable and (n_transferable_req > 0 or essential_pct >= 55.0))
            or (not is_tenure_exempt and min_years_req and verified_tenure < (min_years_req - 0.25) and n_missing == 0 and essential_pct >= 60.0)
            or (50.0 <= overall < 68.0 and n_missing <= 1 and (n_transferable_req > 0 or n_uncertain_req > 0 or unsupported_penalty > 0 or (len(listed_only_essential) > 0 and len(demonstrated_essential) == 0)))
        ):
            rec = Recommendation.MANUAL_REVIEW_REQUIRED
            reasons = []
            if has_transferable:
                trans_details = [
                    f"{t['required']} (candidate has {t['candidate_has']})"
                    for t in transferable_skills[:2]
                ]
                reasons.append(f"Transferable skills present: {'; '.join(trans_details)}")
            if not is_tenure_exempt and min_years_req and verified_tenure < (min_years_req - 0.25):
                reasons.append(
                    f"Tenure ({verified_tenure} yrs) marginally below required ({min_years_req} yrs)"
                )
            if unsupported_penalty > 0:
                reasons.append("Essential skills listed without practical work or project demonstration")
            if len(listed_only_essential) > 0 and len(demonstrated_essential) == 0:
                reasons.append(
                    f"{len(listed_only_essential)} essential skill(s) listed but not demonstrated "
                    "in work experience or projects"
                )
            if 50.0 <= overall < 68.0:
                reasons.append(f"Borderline overall competency match ({overall:.1f}/100)")
            decision_caveat = (
                f"Manual Review Required: {'; '.join(reasons) if reasons else 'Requires human assessment'}."
            )

        # --- Fallback: Not Recommended ---
        else:
            rec = Recommendation.NOT_RECOMMENDED
            decision_caveat = (
                f"Not Recommended: Candidate evaluation ({overall:.1f}/100, {essential_pct:.0f}% essential coverage) "
                f"did not satisfy minimum criteria for recommendation or borderline review."
            )


        # ---------------------------------------------------------------
        # 9. FORMULA & RATIONALE (Explainability)
        # ---------------------------------------------------------------
        if has_llm:
            formula = (
                f"Overall = ({essential_pct:.1f}% × {self.weights.essential_skills_weight:.2f} Essential) "
                f"+ ({practical_score:.1f}% × {self.weights.practical_evidence_weight:.2f} Practical) "
                f"+ ({project_score:.1f}% × {self.weights.project_depth_weight:.2f} Projects) "
                f"+ ({preferred_pct:.1f}% × {self.weights.preferred_skills_weight:.2f} Preferred) "
                f"+ ({llm_score:.1f} × {self.weights.llm_evaluation_weight:.2f} Groq AI)"
            )
        else:
            formula = (
                f"Overall = ({essential_pct:.1f}% × {self.weights.essential_skills_weight:.2f}) "
                f"+ ({preferred_pct:.1f}% × {self.weights.preferred_skills_weight:.2f}) "
                f"+ ({practical_score:.1f}% × {self.weights.practical_evidence_weight:.2f}) "
                f"+ ({exp_duration_score:.1f}% × {self.weights.relevant_experience_weight:.2f}) "
                f"+ ({project_score:.1f}% × {self.weights.project_depth_weight:.2f})"
            )
        if unsupported_penalty > 0:
            formula += f" - {unsupported_penalty:.1f} (Keyword Stuffing Penalty)"
        formula += f" = {overall:.1f}/100"

        missing_names = [r.requirement_text for r in missing_critical]
        tenure_desc = (
            f"Verified tenure: {verified_tenure} yrs."
            if not is_tenure_exempt
            else "Experience tenure: 0% weight (Intern Mode)."
        )
        llm_note = (
            f" Groq AI Qualitative Score: {llm_score:.1f}/100 ({self.weights.llm_evaluation_weight*100:.0f}% weight)."
            if has_llm else ""
        )

        rationale_parts = [
            decision_caveat,
            (
                f"Matched {matched_essential_count}/{total_essential} essential requirements "
                f"({essential_pct:.1f}% coverage; {len(demonstrated_essential)} demonstrated in work/projects, "
                f"{len(listed_only_essential)} listed-only)."
            ),
            f"Practical evidence: {practical_score:.1f}%. Project quality: {project_score:.1f}%.",
            f"{tenure_desc}{llm_note}",
        ]
        if missing_names:
            rationale_parts.append(f"Missing mandatory requirements: {', '.join(missing_names[:3])}.")
        if transferable_skills:
            trans_summary = [
                f"{t['required']} → candidate has {t['candidate_has']}"
                for t in transferable_skills[:2]
            ]
            rationale_parts.append(
                f"Transferable skills (not counted as direct match): {'; '.join(trans_summary)}."
            )
        if listed_only_essential:
            listed_names = [r.requirement_text for r in listed_only_essential[:2]]
            rationale_parts.append(
                f"Essential skills listed but not demonstrated in work/projects: {', '.join(listed_names)}."
            )

        rationale = " ".join(rationale_parts)

        return CandidateEvaluation(
            overall_score=overall,
            essential_coverage_pct=essential_pct,
            preferred_coverage_pct=preferred_pct,
            practical_evidence_score=practical_score,
            experience_duration_score=exp_duration_score,
            project_depth_score=project_score,
            unsupported_keyword_penalty=unsupported_penalty,
            verified_tenure_years=verified_tenure,
            min_years_required=min_years_req,
            recommendation=rec,
            total_essential_count=total_essential,
            matched_essential_count=matched_essential_count,
            total_preferred_count=total_pref,
            matched_preferred_count=matched_pref_count,
            transferable_skills=transferable_skills,
            is_low_confidence_extraction=is_low_conf,
            score_breakdown_formula=formula,
            decision_rationale=rationale,
            llm_score=llm_score,
            llm_recommendation=llm_recommendation,
            llm_synthesis=llm_synthesis
        )
