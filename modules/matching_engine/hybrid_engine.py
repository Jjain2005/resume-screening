import re
from typing import List, Optional, Dict, Any
from .models import (
    MatchClassification,
    RequirementMatchResult,
    EvidenceStrength
)
from .exact_matcher import ExactMatcher
from .fuzzy_matcher import FuzzyMatcher
from .semantic_matcher import SemanticMatcher
from .contextual_matcher import ContextualMatcher
from ..knowledge_base.taxonomy import SkillTaxonomy, normalize_text
from ..knowledge_base.models import RelationshipType
from ..jd_processing.models import (
    JobDescription,
    RequirementItem,
    RequirementType,
    RequirementPriority,
    MatchingLogic
)
from ..resume_processing.models import ProcessedResume, EvidenceSentence, SectionType

class HybridMatchingEngine:
    """
    Central matching intelligence combining Exact, Fuzzy, Semantic,
    Knowledge Base Relationship, Contextual Evidence matching, and Compound Logic.
    """
    def __init__(
        self,
        taxonomy: SkillTaxonomy,
        semantic_matcher: Optional[SemanticMatcher] = None,
        fuzzy_threshold: float = 85.0,
        semantic_threshold: float = 0.76,
        semantic_partial_threshold: float = 0.65
    ):
        self.taxonomy = taxonomy
        self.exact_matcher = ExactMatcher(taxonomy)
        self.fuzzy_matcher = FuzzyMatcher(taxonomy, threshold=fuzzy_threshold)
        self.semantic_matcher = semantic_matcher or SemanticMatcher(
            similarity_threshold=semantic_threshold,
            partial_threshold=semantic_partial_threshold
        )
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.semantic_partial_threshold = semantic_partial_threshold
        self._cached_jd_id: Optional[str] = None
        self._cached_jd_embeddings: dict = {}

    def cache_jd(self, jd: JobDescription):
        """Precomputes and caches normalized embeddings for all requirements in a JD once."""
        self._cached_jd_embeddings = self.semantic_matcher.cache_jd_embeddings(jd.requirements)
        self._cached_jd_id = jd.id

    def match_resume_to_jd(
        self,
        resume: ProcessedResume,
        jd: JobDescription
    ) -> List[RequirementMatchResult]:
        """Evaluates every requirement in the confirmed JD against the candidate resume."""
        if jd.id != self._cached_jd_id:
            self.cache_jd(jd)

        # Precompute vectorized semantic similarity across all requirements in one NumPy operation
        precomputed_hits = self.semantic_matcher.batch_match_resume_against_jd(
            resume, self._cached_jd_embeddings
        )

        results: List[RequirementMatchResult] = []
        for req in jd.requirements:
            match_res = self.match_single_requirement(
                req, resume, precomputed_semantic_hit=precomputed_hits.get(req.id)
            )
            results.append(match_res)

        return results

    def match_single_requirement(
        self,
        req: RequirementItem,
        resume: ProcessedResume,
        precomputed_semantic_hit: Optional[tuple] = None
    ) -> RequirementMatchResult:
        """
        Multi-stage matching pipeline for a single requirement:
        1. Compound ALL_OF Evaluation (if multiple components required)
        2. Exact Canonical & Known Alias Matching (including ANY_OF options)
        3. Accepted Equivalents Matching
        4. Fuzzy Matching for Typos
        5. Related Skill (Not Equivalent) Taxonomy Check
        6. Semantic Sentence Similarity
        7. Contextual Evidence Strength Assessment
        8. Deterministic Classification & Calibration
        """
        # Collect top-k evidence passages for inspection & explainability
        top_k_hits = self.semantic_matcher.find_top_k_semantic_matches(req.text, resume, k=3, threshold=0.45)
        evidence_passages = [
            {
                "sentence": s.text,
                "section": s.section.value,
                "similarity": round(sc, 3),
                "strength": ContextualMatcher.evaluate_evidence_strength(s).value
            }
            for s, sc in top_k_hits
        ]

        # -------------------------------------------------------------
        # Stage 0a: Education Verification
        # -------------------------------------------------------------
        if req.req_type == RequirementType.EDUCATION or any(dk in req.text.lower() for dk in ["bachelor", "master", "ph.d", "phd", "degree in"]):
            edu_hit = self._match_education_requirement(req, resume, evidence_passages)
            if edu_hit:
                return edu_hit

        # -------------------------------------------------------------
        # Stage 0b: Compound ALL_OF Logic
        # -------------------------------------------------------------
        if req.matching_logic == MatchingLogic.ALL_OF and len(req.compound_components) > 1:
            component_results = {}
            matched_components = []
            missing_components = []
            component_sentences = []
            comp_strengths = []

            for comp in req.compound_components:
                # Detect core skill term if component was extracted as a phrase
                core_comp = comp
                if len(comp.split()) > 3:
                    for t, c in self.taxonomy.get_all_lookup_terms().items():
                        if len(t) >= 3 and re.search(rf'\b{re.escape(t)}\b', comp.lower()):
                            core_comp = c
                            break

                canon = self.taxonomy.resolve_canonical(core_comp) or core_comp
                hit = self.exact_matcher.match_skill(core_comp, resume) or self.exact_matcher.match_skill(canon, resume)
                if not hit and core_comp != comp:
                    hit = self.exact_matcher.match_skill(comp, resume)
                if not hit:
                    fhit = self.fuzzy_matcher.match_skill(core_comp, resume, threshold=self.fuzzy_threshold)
                    if fhit:
                        hit = (fhit[0], "Fuzzy", fhit[2])
                if not hit:
                    sem_hits = self.semantic_matcher.find_top_k_semantic_matches(core_comp, resume, k=1, threshold=self.semantic_threshold)
                    if sem_hits:
                        hit = (sem_hits[0][0].text[:30], "Semantic", sem_hits[0][0])

                if hit:
                    term, mtype, sent = hit
                    st = ContextualMatcher.evaluate_evidence_strength(sent)
                    component_results[comp] = {"matched": True, "term": term, "method": mtype, "sentence": sent.text}
                    matched_components.append(comp)
                    component_sentences.append(sent)
                    comp_strengths.append(st)
                else:
                    component_results[comp] = {"matched": False}
                    missing_components.append(comp)

            fulfillment = len(matched_components) / len(req.compound_components)
            best_sentence = component_sentences[0] if component_sentences else (top_k_hits[0][0] if top_k_hits else None)
            best_strength = comp_strengths[0] if comp_strengths else EvidenceStrength.NONE

            if fulfillment == 1.0:
                classification = MatchClassification.EXACT_MATCH
                explanation = f"All {len(req.compound_components)} required components matched: {', '.join(matched_components)}."
            elif fulfillment > 0.0:
                classification = MatchClassification.PARTIAL_EVIDENCE
                explanation = f"Partial fulfillment ({len(matched_components)}/{len(req.compound_components)}): Matched {', '.join(matched_components)}; missing: {', '.join(missing_components)}."
            else:
                classification = MatchClassification.NOT_FOUND
                explanation = f"None of the {len(req.compound_components)} required components were found."

            return RequirementMatchResult(
                requirement_id=req.id,
                requirement_text=req.text,
                priority=req.priority.value,
                classification=classification,
                matched_term=", ".join(matched_components) if matched_components else None,
                canonical_skill=req.canonical_skill,
                evidence_sentence=best_sentence.text if best_sentence else None,
                resume_section=best_sentence.section if best_sentence else None,
                similarity_score=round(fulfillment, 3),
                evidence_strength=best_strength,
                matching_method=f"Compound ALL_OF ({len(matched_components)}/{len(req.compound_components)})",
                explanation=explanation,
                confidence=round(fulfillment * 0.95, 2),
                weight=req.weight,
                req_type=req.req_type.value,
                matching_logic=req.matching_logic.value,
                is_eligibility_condition=req.is_eligibility_condition,
                fulfillment_ratio=fulfillment,
                evidence_passages=evidence_passages,
                component_matches=component_results
            )

        target_skill = req.canonical_skill or (req.text if req.req_type == RequirementType.SKILL else None)

        # -------------------------------------------------------------
        # Stage 1: Exact Match (Canonical & Alias & ANY_OF Alternatives)
        # -------------------------------------------------------------
        candidate_target_skills = []
        if target_skill:
            candidate_target_skills.append(target_skill)

        # Include compound components for ANY_OF logic
        if req.matching_logic == MatchingLogic.ANY_OF and req.compound_components:
            for comp in req.compound_components:
                canon = self.taxonomy.resolve_canonical(comp) or comp
                if canon not in candidate_target_skills:
                    candidate_target_skills.append(canon)

        # Detect any additional canonical skills mentioned in req.text
        if req.text:
            all_terms = self.taxonomy.get_all_lookup_terms()
            for term, canon in all_terms.items():
                if len(term) >= 3 and term in req.text.lower():
                    if canon not in candidate_target_skills:
                        candidate_target_skills.append(canon)

        for skill_to_check in candidate_target_skills:
            exact_hit = self.exact_matcher.match_skill(skill_to_check, resume)
            if exact_hit:
                matched_term, match_type, sentence = exact_hit
                strength = ContextualMatcher.evaluate_evidence_strength(sentence)
                canonical = self.taxonomy.resolve_canonical(skill_to_check) or skill_to_check

                explanation = (
                    f"Direct {match_type.lower()} of '{matched_term}' located in {sentence.section.value} section."
                    if match_type == "Exact Canonical" else
                    f"Known alias '{matched_term}' of canonical skill '{canonical}' matched in {sentence.section.value} section."
                )

                return RequirementMatchResult(
                    requirement_id=req.id,
                    requirement_text=req.text,
                    priority=req.priority.value,
                    classification=MatchClassification.EXACT_MATCH,
                    matched_term=matched_term,
                    canonical_skill=canonical,
                    evidence_sentence=sentence.text,
                    resume_section=sentence.section,
                    similarity_score=1.0,
                    evidence_strength=strength,
                    matching_method=match_type,
                    explanation=explanation,
                    confidence=0.98 if strength in (EvidenceStrength.EXPERIENCE_EVIDENCE, EvidenceStrength.STRONG_IMPLEMENTATION) else 0.90,
                    weight=req.weight,
                    req_type=req.req_type.value,
                    matching_logic=req.matching_logic.value,
                    is_eligibility_condition=req.is_eligibility_condition,
                    fulfillment_ratio=1.0,
                    evidence_passages=evidence_passages
                )

        # -------------------------------------------------------------
        # Stage 1b: Accepted Equivalents Matching
        # -------------------------------------------------------------
        if req.accepted_equivalents:
            for eq in req.accepted_equivalents:
                eq_hit = self.exact_matcher.match_skill(eq, resume)
                if not eq_hit:
                    eq_canon = self.taxonomy.resolve_canonical(eq) or eq
                    eq_hit = self.exact_matcher.match_skill(eq_canon, resume)
                if eq_hit:
                    matched_term, match_type, sentence = eq_hit
                    strength = ContextualMatcher.evaluate_evidence_strength(sentence)
                    return RequirementMatchResult(
                        requirement_id=req.id,
                        requirement_text=req.text,
                        priority=req.priority.value,
                        classification=MatchClassification.EXACT_MATCH,
                        matched_term=matched_term,
                        canonical_skill=eq,
                        evidence_sentence=sentence.text,
                        resume_section=sentence.section,
                        similarity_score=0.95,
                        evidence_strength=strength,
                        matching_method="Accepted Equivalent",
                        explanation=f"Candidate demonstrates accepted equivalent technology '{matched_term}' for '{req.text}'.",
                        confidence=0.90,
                        weight=req.weight,
                        req_type=req.req_type.value,
                        matching_logic=req.matching_logic.value,
                        is_eligibility_condition=req.is_eligibility_condition,
                        fulfillment_ratio=1.0,
                        evidence_passages=evidence_passages
                    )

        # -------------------------------------------------------------
        # Stage 2: Fuzzy Match (Spelling variations, typos)
        # -------------------------------------------------------------
        skills_for_fuzzy = candidate_target_skills if candidate_target_skills else ([target_skill] if target_skill else [])
        for skill_to_fuzzy in skills_for_fuzzy:
            fuzzy_hit = self.fuzzy_matcher.match_skill(skill_to_fuzzy, resume, threshold=self.fuzzy_threshold)
            if fuzzy_hit:
                matched_term, score, sentence = fuzzy_hit
                strength = ContextualMatcher.evaluate_evidence_strength(sentence)
                canonical = self.taxonomy.resolve_canonical(skill_to_fuzzy) or skill_to_fuzzy

                if score >= 90.0:
                    classification = MatchClassification.FUZZY_MATCH
                    conf = 0.85
                    explanation = f"Fuzzy match '{matched_term}' ({score:.1f}% similarity) for '{skill_to_fuzzy}' in {sentence.section.value}."
                    fulfillment = 1.0
                else:
                    classification = MatchClassification.NEEDS_MANUAL_REVIEW
                    conf = 0.65
                    explanation = f"Borderline fuzzy match '{matched_term}' ({score:.1f}% similarity). Recommended for manual verification."
                    fulfillment = 0.5

                return RequirementMatchResult(
                    requirement_id=req.id,
                    requirement_text=req.text,
                    priority=req.priority.value,
                    classification=classification,
                    matched_term=matched_term,
                    canonical_skill=canonical,
                    evidence_sentence=sentence.text,
                    resume_section=sentence.section,
                    similarity_score=round(score / 100.0, 3),
                    evidence_strength=strength,
                    matching_method="Fuzzy Matching",
                    explanation=explanation,
                    confidence=conf,
                    weight=req.weight,
                    req_type=req.req_type.value,
                    matching_logic=req.matching_logic.value,
                    is_eligibility_condition=req.is_eligibility_condition,
                    fulfillment_ratio=fulfillment,
                    evidence_passages=evidence_passages
                )

        # -------------------------------------------------------------
        # Stage 3: Semantic Matching against Resume Sentences
        # -------------------------------------------------------------
        candidates = []
        if precomputed_semantic_hit is not None:
            candidates.append(precomputed_semantic_hit)

        all_sem_hits = self.semantic_matcher.find_all_matches_above_threshold(
            req.text, resume, threshold=self.semantic_partial_threshold
        )
        for s_sent, s_score in all_sem_hits:
            if not any(c[0].text == s_sent.text for c in candidates):
                candidates.append((s_sent, s_score))

        # Check for strong semantic match with verified domain relevance
        strong_hit = None
        for s_sent, s_score in candidates:
            if s_score >= self.semantic_threshold:
                if self._verify_domain_relevance(req, s_sent):
                    strong_hit = (s_sent, s_score)
                    break

        if strong_hit:
            sem_sentence, sem_score = strong_hit
            # Check if this sentence actually describes a related but non-equivalent alternative skill
            if target_skill:
                alt_skill = self._find_competing_related_skill_in_sentence(target_skill, sem_sentence, req_text=req.text)
                if alt_skill:
                    strength = ContextualMatcher.evaluate_evidence_strength(sem_sentence)
                    return RequirementMatchResult(
                        requirement_id=req.id,
                        requirement_text=req.text,
                        priority=req.priority.value,
                        classification=MatchClassification.RELATED_SKILL_NOT_EQUIVALENT,
                        matched_term=alt_skill,
                        canonical_skill=target_skill,
                        evidence_sentence=sem_sentence.text,
                        resume_section=sem_sentence.section,
                        similarity_score=round(sem_score, 3),
                        evidence_strength=strength,
                        matching_method="Taxonomy Relationship (Alternative Skill Detected)",
                        explanation=f"High semantic similarity was triggered by '{alt_skill}', which is related to '{target_skill}' but not fully equivalent.",
                        confidence=0.80,
                        weight=req.weight,
                        req_type=req.req_type.value,
                        matching_logic=req.matching_logic.value,
                        is_eligibility_condition=req.is_eligibility_condition,
                        fulfillment_ratio=0.0,
                        evidence_passages=evidence_passages
                    )

            strength = ContextualMatcher.evaluate_evidence_strength(sem_sentence)
            return RequirementMatchResult(
                requirement_id=req.id,
                requirement_text=req.text,
                priority=req.priority.value,
                classification=MatchClassification.STRONG_SEMANTIC_MATCH,
                matched_term=sem_sentence.text[:40] + "...",
                canonical_skill=target_skill,
                evidence_sentence=sem_sentence.text,
                resume_section=sem_sentence.section,
                similarity_score=round(sem_score, 3),
                evidence_strength=strength,
                matching_method="Semantic Matching",
                explanation=f"Strong semantic correspondence ({sem_score:.2f} cosine similarity) found in {sem_sentence.section.value}.",
                confidence=min(0.92, round(sem_score * 0.95, 2)),
                weight=req.weight,
                req_type=req.req_type.value,
                matching_logic=req.matching_logic.value,
                is_eligibility_condition=req.is_eligibility_condition,
                fulfillment_ratio=1.0,
                evidence_passages=evidence_passages
            )

        # -------------------------------------------------------------
        # Stage 4: Check for Known Related Skills (Not Equivalent)
        # -------------------------------------------------------------
        if target_skill:
            related_match = self._check_related_skills(target_skill, resume)
            if related_match:
                rel_skill, rel_type, rel_sentence = related_match
                return RequirementMatchResult(
                    requirement_id=req.id,
                    requirement_text=req.text,
                    priority=req.priority.value,
                    classification=MatchClassification.RELATED_SKILL_NOT_EQUIVALENT,
                    matched_term=rel_skill,
                    canonical_skill=target_skill,
                    evidence_sentence=rel_sentence.text,
                    resume_section=rel_sentence.section,
                    similarity_score=0.60,
                    evidence_strength=ContextualMatcher.evaluate_evidence_strength(rel_sentence),
                    matching_method=f"Taxonomy Relationship ({rel_type})",
                    explanation=f"Candidate possesses '{rel_skill}' which is {rel_type} to '{target_skill}', but not fully equivalent.",
                    confidence=0.70,
                    weight=req.weight,
                    req_type=req.req_type.value,
                    matching_logic=req.matching_logic.value,
                    is_eligibility_condition=req.is_eligibility_condition,
                    fulfillment_ratio=0.0,
                    evidence_passages=evidence_passages
                )

        # -------------------------------------------------------------
        # Stage 5: Partial Evidence / Borderline Semantic
        # -------------------------------------------------------------
        partial_hit = None
        for s_sent, s_score in candidates:
            if s_score >= self.semantic_partial_threshold:
                if self._verify_domain_relevance(req, s_sent):
                    partial_hit = (s_sent, s_score)
                    break

        if partial_hit:
            sem_sentence, sem_score = partial_hit
            strength = ContextualMatcher.evaluate_evidence_strength(sem_sentence)
            return RequirementMatchResult(
                requirement_id=req.id,
                requirement_text=req.text,
                priority=req.priority.value,
                classification=MatchClassification.PARTIAL_EVIDENCE,
                matched_term=sem_sentence.text[:40] + "...",
                canonical_skill=target_skill,
                evidence_sentence=sem_sentence.text,
                resume_section=sem_sentence.section,
                similarity_score=round(sem_score, 3),
                evidence_strength=strength,
                matching_method="Semantic (Partial)",
                explanation=f"Moderate semantic similarity ({sem_score:.2f}) indicates partial evidence, but lacks specific confirmation.",
                confidence=round(sem_score * 0.8, 2),
                weight=req.weight,
                req_type=req.req_type.value,
                matching_logic=req.matching_logic.value,
                is_eligibility_condition=req.is_eligibility_condition,
                fulfillment_ratio=0.3,
                evidence_passages=evidence_passages
            )

        # -------------------------------------------------------------
        # Stage 6: Not Found
        # -------------------------------------------------------------
        return RequirementMatchResult(
            requirement_id=req.id,
            requirement_text=req.text,
            priority=req.priority.value,
            classification=MatchClassification.NOT_FOUND,
            matched_term=None,
            canonical_skill=target_skill,
            evidence_sentence=None,
            resume_section=None,
            similarity_score=0.0,
            evidence_strength=EvidenceStrength.NONE,
            matching_method="None",
            explanation="No direct, alias, fuzzy, semantic, compound, or related evidence found in resume.",
            confidence=0.95,
            weight=req.weight,
            req_type=req.req_type.value,
            matching_logic=req.matching_logic.value,
            is_eligibility_condition=req.is_eligibility_condition,
            fulfillment_ratio=0.0,
            evidence_passages=evidence_passages
        )

    def _find_competing_related_skill_in_sentence(
        self,
        target_skill: str,
        sentence: EvidenceSentence,
        req_text: str = ""
    ) -> Optional[str]:
        """Detects if a high-scoring semantic sentence actually mentions an alternative competing skill."""
        canonical = self.taxonomy.resolve_canonical(target_skill) or target_skill
        entry = self.taxonomy.get_skill(canonical)
        if not entry:
            return None

        req_text_lower = req_text.lower() if req_text else ""

        competing_types = (RelationshipType.RELATED_NOT_EQUIVALENT, RelationshipType.RELATED)
        for rel in entry.relationships:
            if rel.type in competing_types:
                rel_canonical = self.taxonomy.resolve_canonical(rel.target) or rel.target
                if rel_canonical.lower() in req_text_lower:
                    continue

                rel_entry = self.taxonomy.get_skill(rel_canonical)
                terms = [rel_canonical]
                if rel_entry:
                    terms.extend(rel_entry.aliases)

                for t in terms:
                    if t.lower() in req_text_lower:
                        continue
                    from .exact_matcher import word_boundary_regex
                    if word_boundary_regex(t).search(sentence.text):
                        return t

        return None

    def _check_related_skills(
        self,
        target_skill: str,
        resume: ProcessedResume
    ) -> Optional[tuple]:
        """Checks if resume contains any skill related (non-equivalent) to target_skill in the taxonomy."""
        canonical = self.taxonomy.resolve_canonical(target_skill) or target_skill
        entry = self.taxonomy.get_skill(canonical)
        if not entry:
            return None

        competing_types = (RelationshipType.RELATED_NOT_EQUIVALENT, RelationshipType.RELATED)
        for rel in entry.relationships:
            if rel.type in competing_types:
                related_name = rel.target
                exact_hit = self.exact_matcher.match_skill(related_name, resume)
                if exact_hit:
                    matched_term, _, sentence = exact_hit
                    return (matched_term, rel.type.value, sentence)

        return None

    def _verify_domain_relevance(
        self,
        req: RequirementItem,
        sentence: EvidenceSentence
    ) -> bool:
        """
        Validates that a candidate evidence sentence shares topical/domain relevance
        with the target requirement to prevent generic linguistic false positives.
        """
        if req.req_type == RequirementType.RESPONSIBILITY:
            return True

        req_text_clean = normalize_text(req.text)
        sent_text_clean = normalize_text(sentence.text)

        stopwords = {
            "strong", "experience", "proficiency", "proficient", "hands", "on", "working",
            "knowledge", "expertise", "years", "year", "professional", "understanding",
            "skills", "skill", "ability", "background", "good", "plus", "bonus",
            "familiarity", "frameworks", "framework", "tools", "tool", "technologies",
            "technology", "platforms", "platform", "environments", "environment",
            "develop", "development", "building", "built", "using", "with", "and", "or",
            "for", "the", "in", "of", "to", "a", "an", "at", "by", "from", "on",
            "demonstrated", "proven", "minimum", "required", "requirements"
        }

        req_words = [w for w in re.findall(r'\b[a-z0-9+#.]+\b', req_text_clean) if len(w) >= 2 and w not in stopwords]
        sent_words = set(w for w in re.findall(r'\b[a-z0-9+#.]+\b', sent_text_clean) if len(w) >= 2 and w not in stopwords)

        if set(req_words) & sent_words:
            return True

        target_skill = req.canonical_skill
        if target_skill:
            entry = self.taxonomy.get_skill(target_skill)
            if entry:
                for alias in entry.aliases:
                    alias_norm = normalize_text(alias)
                    if alias_norm and re.search(rf'\b{re.escape(alias_norm)}\b', sent_text_clean):
                        return True
                for rel in entry.relationships:
                    rel_target = normalize_text(rel.target)
                    if rel_target and re.search(rf'\b{re.escape(rel_target)}\b', sent_text_clean):
                        return True
                    rel_entry = self.taxonomy.get_skill(rel.target)
                    if rel_entry:
                        for rel_alias in rel_entry.aliases:
                            rel_alias_norm = normalize_text(rel_alias)
                            if rel_alias_norm and re.search(rf'\b{re.escape(rel_alias_norm)}\b', sent_text_clean):
                                return True

        if req.req_type == RequirementType.EDUCATION or any(kw in req_text_clean for kw in ["degree", "bachelor", "master", "phd", "computer science"]):
            edu_keywords = {"bachelor", "master", "phd", "degree", "bs", "ms", "computer", "science", "engineering", "technology", "university", "college"}
            if (set(req_words) & edu_keywords) and (sent_words & edu_keywords):
                return True

        if len(req_words) <= 1 and not target_skill:
            return True

        return False

    def _match_education_requirement(
        self,
        req: RequirementItem,
        resume: ProcessedResume,
        evidence_passages: List[Dict[str, Any]]
    ) -> Optional[RequirementMatchResult]:
        """
        Directly checks candidate's education and degrees against educational requirements.
        """
        edu_sentences = resume.get_sentences_by_section(SectionType.EDUCATION) or []
        sentences_to_search = edu_sentences if edu_sentences else resume.all_sentences
        all_text = " ".join(s.text.lower() for s in sentences_to_search)

        degree_keywords = [
            "bachelor", "master", "ph.d", "phd", "doctorate", "bs", "ms", "b.s", "m.s",
            "b.tech", "m.tech", "btech", "mtech", "b.e", "m.e", "degree", "diploma", "graduate"
        ]
        field_keywords = [
            "computer science", "software engineering", "computer engineering",
            "data science", "information technology", "engineering", "math", "physics", "related field"
        ]

        has_degree = any(re.search(rf'\b{re.escape(dk)}\b', all_text) for dk in degree_keywords)
        has_field = any(fk in all_text for fk in field_keywords)

        if has_degree:
            best_sent = None
            for s in sentences_to_search:
                s_low = s.text.lower()
                if any(re.search(rf'\b{re.escape(dk)}\b', s_low) for dk in degree_keywords):
                    if any(fk in s_low for fk in field_keywords):
                        best_sent = s
                        break
                    elif not best_sent:
                        best_sent = s

            matched_text = best_sent.text if best_sent else "Verified Degree"
            return RequirementMatchResult(
                requirement_id=req.id,
                requirement_text=req.text,
                priority=req.priority.value,
                classification=MatchClassification.EXACT_MATCH,
                matched_term=matched_text,
                canonical_skill="Education",
                evidence_sentence=best_sent.text if best_sent else None,
                resume_section=best_sent.section if best_sent else SectionType.EDUCATION,
                similarity_score=1.0,
                evidence_strength=EvidenceStrength.EXPERIENCE_EVIDENCE,
                matching_method="Education Verification",
                explanation=f"Candidate satisfies education requirement with: '{matched_text}'.",
                confidence=0.98,
                weight=req.weight,
                req_type="Education",
                matching_logic=req.matching_logic.value,
                is_eligibility_condition=req.is_eligibility_condition,
                fulfillment_ratio=1.0,
                evidence_passages=evidence_passages
            )
        return None

