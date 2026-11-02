import re
import json
import logging
import hashlib
from typing import List, Optional, Dict, Any
from .models import (
    JobDescription,
    RequirementItem,
    RequirementType,
    RequirementPriority,
    MatchingLogic,
    JDEvaluationStrategy
)
from ..knowledge_base.taxonomy import SkillTaxonomy, normalize_text

logger = logging.getLogger(__name__)

class JobDescriptionParser:
    """
    Parses unstructured Job Description text into structured requirements.
    Supports LLM-assisted deep understanding with deterministic regex/taxonomy fallback.
    Identifies mandatory, preferred, and optional requirements, compound AND/OR logic,
    accepted equivalents, and dynamically assigned weights without fixed role presets.
    """
    def __init__(
        self,
        taxonomy: Optional[SkillTaxonomy] = None,
        groq_manager: Optional[Any] = None
    ):
        self.taxonomy = taxonomy
        self.groq_manager = groq_manager

    def parse(
        self,
        text: str,
        jd_id: str = "jd_1",
        default_title: str = "Job Position",
        explicit_title: Optional[str] = None,
        use_llm: bool = True,
        **kwargs
    ) -> JobDescription:
        """
        Parses a Job Description. Uses LLM if available and requested, otherwise
        falls back to deterministic parsing.
        """
        if use_llm and self.groq_manager and getattr(self.groq_manager, "is_available", False):
            try:
                llm_jd = self.parse_with_llm(
                    text,
                    jd_id=jd_id,
                    default_title=default_title,
                    explicit_title=explicit_title
                )
                if llm_jd and len(llm_jd.requirements) > 0:
                    return llm_jd
            except Exception as e:
                logger.warning(f"LLM Job Description parsing failed: {e}. Falling back to deterministic parser.")

        return self.parse_deterministic(
            text,
            jd_id=jd_id,
            default_title=default_title,
            explicit_title=explicit_title,
            **kwargs
        )

    def parse_with_llm(
        self,
        text: str,
        jd_id: str = "jd_1",
        default_title: str = "Job Position",
        explicit_title: Optional[str] = None
    ) -> Optional[JobDescription]:
        """
        Uses Groq LLM to deeply understand the Job Description, extract criteria,
        and assign priority (Mandatory, Preferred, Optional), compound logic, and weights.
        """
        client = self.groq_manager.get_client()
        if client is None:
            return None

        system_prompt = (
            "You are an expert Job Description Analyzer and Technical Recruiter. "
            "Analyze the provided Job Description text and extract a structured evaluation strategy and evaluation criteria.\n\n"
            "Produce a JSON object with:\n"
            "- 'title': Job position title\n"
            "- 'job_domain': Primary domain/field (e.g., 'Cloud Infrastructure & DevOps', 'Machine Learning', 'Full-Stack Development')\n"
            "- 'role_summary': 1-2 sentence overview of core mission and responsibilities\n"
            "- 'essential_capabilities': List of 3-6 core practical abilities required to succeed\n"
            "- 'relevant_skills': Key technical tools, languages, and frameworks\n"
            "- 'expected_practical_knowledge': Concrete implementations expected (e.g., CI/CD pipelines, API design)\n"
            "- 'desirable_qualifications': Nice-to-have skills or experiences\n"
            "- 'requirements': List of specific requirements with:\n"
            "   - 'text': Exact atomic requirement statement (break compound multi-skill bullets into atomic items while preserving AND/OR compound logic)\n"
            "   - 'priority': 'Mandatory' (strict eligibility prerequisites: e.g. required degree, work authorization, hard minimum years) | 'Core' (primary essential job competencies, day-to-day technical skills) | 'Preferred' (nice-to-have supplementary skills) | 'Optional' (minor bonus)\n"
            "   - 'req_type': 'Skill' | 'Experience' | 'Education' | 'Certification' | 'Responsibility'\n"
            "   - 'matching_logic': 'Single' | 'ALL_OF' | 'ANY_OF'\n"
            "   - 'compound_components': List of components if ALL_OF (AND) or ANY_OF (OR)\n"
            "   - 'accepted_equivalents': List of interchangeable technologies\n"
            "   - 'min_years': Number or null\n"
            "   - 'relative_weight': Float (e.g. Mandatory 3.0, Core 2.5, Preferred 1.2, Optional 0.5)\n\n"
            "CRITICAL RULES:\n"
            "1. Extract ATOMIC requirements: do not lump distinct tools/skills together unless connected by explicit AND/OR.\n"
            "2. Do NOT confuse responsibilities ('Responsible for...', 'Lead the team...') with mandatory skills. Responsibilities must have req_type 'Responsibility' and priority 'Core' or 'Preferred', NEVER 'Mandatory'.\n"
            "3. Preserve AND/OR relationships: Use matching_logic 'ALL_OF' for AND, 'ANY_OF' for OR, and list component skills in 'compound_components'.\n"
            "4. Classify Priority strictly into ONE of: Mandatory, Core, Preferred, Optional.\n"
            "5. DO NOT ASSUME requirements not in the text. If experience, degree, projects, or certifications are not mentioned, DO NOT invent them.\n"
            "6. Return strictly valid JSON."
        )

        user_prompt = f"Analyze this Job Description:\n\n{text}"

        response = client.chat.completions.create(
            model=self.groq_manager.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2500,
            timeout=25.0
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        raw_title = explicit_title or data.get("title") or default_title
        title = raw_title.strip() if raw_title and raw_title.strip() else default_title
        role_summary = data.get("role_summary") or data.get("summary") or ""

        raw_reqs = data.get("requirements") or data.get("criteria") or []
        requirements: List[RequirementItem] = []

        for idx, item_data in enumerate(raw_reqs):
            req_text = item_data.get("text") or item_data.get("description") or ""
            if not req_text.strip():
                continue

            prio_str = item_data.get("priority", "Mandatory")
            priority = RequirementPriority.from_str(prio_str)

            type_str = item_data.get("req_type", item_data.get("type", "Skill"))
            try:
                req_type = RequirementType(type_str)
            except ValueError:
                req_type = self._detect_requirement_type(req_text)

            logic_str = item_data.get("matching_logic", "Single")
            try:
                matching_logic = MatchingLogic(logic_str)
            except ValueError:
                matching_logic = self._detect_matching_logic(req_text)

            components = item_data.get("compound_components") or []
            if not components and matching_logic != MatchingLogic.SINGLE:
                components = self._extract_compound_components(req_text, matching_logic)

            equivalents = item_data.get("accepted_equivalents") or []
            min_years = item_data.get("min_years")
            if min_years is not None:
                try:
                    min_years = float(min_years)
                except (ValueError, TypeError):
                    min_years = self._extract_min_years(req_text)
            else:
                if req_type == RequirementType.EXPERIENCE:
                    min_years = self._extract_min_years(req_text)

            canonical = self._detect_canonical_skill(req_text)

            # Initial relative weight
            rel_weight = float(item_data.get("relative_weight", item_data.get("weight", 3.0 if priority.is_mandatory else 1.5)))

            # Responsibilities must not be confused with mandatory skills
            if req_type == RequirementType.RESPONSIBILITY:
                if priority.is_mandatory:
                    priority = RequirementPriority.CORE
                rel_weight = min(rel_weight, 1.0)

            # Detect behavioural requirements and treat them as optional/low-weight
            # even if they appear under a "Mandatory" JD heading.
            # Their absence in a resume does NOT prove a candidate lacks these qualities.
            if req_type not in (RequirementType.EDUCATION, RequirementType.CERTIFICATION,
                                RequirementType.EXPERIENCE) and self._is_behavioural_requirement(req_text):
                req_type = RequirementType.BEHAVIOURAL
                # Behavioural reqs contribute to score but never gate on 'missing'
                if priority.is_mandatory:
                    priority = RequirementPriority.PREFERRED
                rel_weight = min(rel_weight, 0.5)  # Cap weight

            req_item = RequirementItem(
                id=f"req_{idx+1}",
                text=req_text.strip(),
                req_type=req_type,
                priority=priority,
                canonical_skill=canonical,
                min_years=min_years,
                weight=rel_weight,
                matching_logic=matching_logic,
                compound_components=[c.strip() for c in components if isinstance(c, str) and c.strip()],
                accepted_equivalents=[e.strip() for e in equivalents if isinstance(e, str) and e.strip()],
                expected_evidence=str(item_data.get("expected_evidence", "")),
                is_eligibility_condition=bool(item_data.get("is_eligibility_condition", False)),
                importance_rationale=str(item_data.get("importance_rationale", ""))
            )
            requirements.append(req_item)

        jd_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
        essential_caps = list(data.get("essential_capabilities") or [r.text for r in requirements if r.priority.is_mandatory][:5])
        relevant_sk = list(data.get("relevant_skills") or [r.canonical_skill or r.text for r in requirements if r.canonical_skill][:10])
        expected_pract = list(data.get("expected_practical_knowledge") or [r.text for r in requirements if r.req_type == RequirementType.EXPERIENCE][:5])
        desirable_quals = list(data.get("desirable_qualifications") or [r.text for r in requirements if not r.priority.is_mandatory][:5])
        domain = str(data.get("job_domain") or self._infer_domain(title, text))

        strategy = JDEvaluationStrategy(
            job_domain=domain,
            essential_capabilities=essential_caps,
            relevant_skills=relevant_sk,
            expected_practical_knowledge=expected_pract,
            desirable_qualifications=desirable_quals,
            role_summary=role_summary,
            strategy_hash=jd_hash
        )

        jd = JobDescription(
            id=jd_id,
            title=title,
            raw_text=text,
            requirements=requirements,
            role_summary=role_summary,
            evaluation_strategy=strategy
        )
        jd.normalize_weights()
        return jd

    def parse_deterministic(
        self,
        text: str,
        jd_id: str = "jd_1",
        default_title: str = "Job Position",
        explicit_title: Optional[str] = None,
        **kwargs
    ) -> JobDescription:
        """
        Deterministic regex and taxonomy-based JD parser.
        Detects headings, extracts requirements, identifies compound logic,
        and normalizes dynamic weights.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        raw_title = explicit_title or kwargs.get("explicit_title") or default_title
        title = raw_title.strip() if raw_title and raw_title.strip() else "Job Position"
        title_inferred_from_first_line = False

        if not raw_title or raw_title == "Job Position":
            if lines:
                first_line = lines[0]
                if len(first_line) < 60 and not any(kw in first_line.lower() for kw in [
                    "requirement", "qualification", "responsibilit", "description",
                    "overview", "about", "looking for", "we are"
                ]):
                    title = first_line
                    title_inferred_from_first_line = True

        requirements: List[RequirementItem] = []
        req_counter = 1

        section_heading_keywords = [
            "requirements", "required", "minimum qualifications", "must have", "what you need",
            "preferred", "nice to have", "bonus", "good to have", "desired", "plus", "optional",
            "responsibilities", "what you will do", "duties", "role overview"
        ]
        has_section_headers = any(
            any(kw in line.lower() for kw in section_heading_keywords) and len(line.strip()) < 50
            for line in lines
        )

        in_preamble = has_section_headers
        current_priority = RequirementPriority.MANDATORY
        current_section_type = "default"

        for idx, line in enumerate(lines):
            if idx == 0 and title_inferred_from_first_line:
                continue

            line_lower = line.lower().strip()

            if any(kw in line_lower for kw in ["preferred", "nice to have", "bonus", "good to have", "desired", "plus", "optional"]) and len(line_lower) < 50:
                if any(kw in line_lower for kw in ["bonus", "optional"]):
                    current_priority = RequirementPriority.OPTIONAL
                else:
                    current_priority = RequirementPriority.PREFERRED
                current_section_type = "preferred"
                in_preamble = False
                continue
            elif any(kw in line_lower for kw in ["requirements", "required", "minimum qualifications", "must have", "basic qualifications", "what you need", "what we're looking for"]) and len(line_lower) < 50:
                current_priority = RequirementPriority.MANDATORY
                current_section_type = "required"
                in_preamble = False
                continue
            elif any(kw in line_lower for kw in ["responsibilities", "what you will do", "duties", "role overview", "day to day", "key responsibilities"]) and len(line_lower) < 50:
                current_priority = RequirementPriority.PREFERRED
                current_section_type = "responsibilities"
                in_preamble = False
                continue
            elif any(kw in line_lower for kw in ["about us", "company overview", "about the company", "who we are", "about the role"]) and len(line_lower) < 50:
                in_preamble = True
                continue

            if in_preamble:
                if not any(qkw in line_lower for qkw in ["must have", "minimum of", "required:", "qualifications:"]):
                    continue

            cleaned_line = re.sub(r'^[\s•\-\*\u2022\u25E6\u25AA\u25CF]+\s*', '', line)
            cleaned_line = re.sub(r'^\(?\d+[\.\)]\s*', '', cleaned_line).strip()
            if not cleaned_line or len(cleaned_line) < 3:
                continue

            if cleaned_line.endswith(":") and len(cleaned_line) < 30:
                continue

            list_prefix_match = re.match(r'^(?:skills|technologies|tools|competencies)[\s:]+(.+)$', cleaned_line, re.IGNORECASE)
            if list_prefix_match:
                sub_items = [s.strip() for s in re.split(r'[,;•|]', list_prefix_match.group(1)) if s.strip()]
                for sub in sub_items:
                    if len(sub) >= 2:
                        canon = self._detect_canonical_skill(sub)
                        w = 3.0 if current_priority.is_mandatory else 1.5
                        requirements.append(RequirementItem(
                            id=f"req_{req_counter}",
                            text=sub,
                            req_type=RequirementType.SKILL,
                            priority=current_priority,
                            canonical_skill=canon,
                            min_years=None,
                            weight=w,
                            matching_logic=MatchingLogic.SINGLE
                        ))
                        req_counter += 1
                continue

            sub_sentences = [cleaned_line]
            if len(cleaned_line) > 140 and ("." in cleaned_line or ";" in cleaned_line):
                split_parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', cleaned_line) if len(p.strip()) >= 10]
                if len(split_parts) > 1:
                    sub_sentences = split_parts

            for sentence_text in sub_sentences:
                req_type = self._detect_requirement_type(sentence_text)
                if current_section_type == "responsibilities" or req_type == RequirementType.RESPONSIBILITY:
                    req_type = RequirementType.RESPONSIBILITY
                    item_priority = RequirementPriority.CORE if current_priority.is_mandatory else current_priority
                    weight = 0.5
                elif req_type == RequirementType.BEHAVIOURAL:
                    # Behavioural qualities: real but unverifiable from resume text alone.
                    # Downgrade to Preferred even if they appear under a Mandatory heading.
                    # Their absence cannot be used to reject a candidate.
                    item_priority = RequirementPriority.PREFERRED
                    weight = 0.5
                else:
                    item_priority = current_priority
                    if item_priority.is_mandatory:
                        weight = 3.0
                    elif item_priority == RequirementPriority.CORE:
                        weight = 2.5
                    elif item_priority == RequirementPriority.PREFERRED:
                        weight = 1.2
                    else:
                        weight = 0.5

                min_years = self._extract_min_years(sentence_text) if req_type == RequirementType.EXPERIENCE else None
                canonical_skill = self._detect_canonical_skill(sentence_text)

                if canonical_skill and req_type not in (RequirementType.EDUCATION, RequirementType.CERTIFICATION, RequirementType.RESPONSIBILITY):
                    req_type = RequirementType.SKILL

                matching_logic = self._detect_matching_logic(sentence_text)
                components = self._extract_compound_components(sentence_text, matching_logic)

                item = RequirementItem(
                    id=f"req_{req_counter}",
                    text=sentence_text,
                    req_type=req_type,
                    priority=item_priority,
                    canonical_skill=canonical_skill,
                    min_years=min_years,
                    weight=weight,
                    matching_logic=matching_logic,
                    compound_components=components
                )
                requirements.append(item)
                req_counter += 1

        jd_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
        essential_caps = [r.text for r in requirements if r.priority.is_mandatory]
        relevant_sk = [r.canonical_skill or r.text for r in requirements if r.req_type == RequirementType.SKILL]
        expected_pract = [r.text for r in requirements if r.req_type == RequirementType.EXPERIENCE or "project" in r.text.lower()]
        desirable_quals = [r.text for r in requirements if not r.priority.is_mandatory]

        strategy = JDEvaluationStrategy(
            job_domain=self._infer_domain(title, text),
            essential_capabilities=essential_caps[:5],
            relevant_skills=relevant_sk[:10],
            expected_practical_knowledge=expected_pract[:5],
            desirable_qualifications=desirable_quals[:5],
            role_summary=f"Role for {title}",
            strategy_hash=jd_hash
        )

        jd = JobDescription(
            id=jd_id,
            title=title,
            raw_text=text,
            requirements=requirements,
            role_summary=f"Role for {title}",
            evaluation_strategy=strategy
        )
        jd.normalize_weights()
        return jd

    def _infer_domain(self, title: str, text: str) -> str:
        t = f"{title} {text[:400]}".lower()
        if any(w in t for w in ["ai", "machine learning", "data science", "nlp", "deep learning", "llm"]):
            return "Artificial Intelligence & Data Science"
        if any(w in t for w in ["devops", "cloud", "kubernetes", "docker", "infra", "sre", "ci/cd"]):
            return "Cloud Infrastructure & DevOps"
        if any(w in t for w in ["frontend", "react", "ui", "ux", "web developer", "typescript"]):
            return "Frontend Web Development"
        if any(w in t for w in ["backend", "api", "microservice", "python", "golang", "java", "django", "fastapi"]):
            return "Backend Software Engineering"
        if any(w in t for w in ["marketing", "seo", "sales", "growth", "lead generation"]):
            return "Growth Marketing & Sales"
        return f"{title} Engineering"

    def _detect_matching_logic(self, text: str) -> MatchingLogic:
        """Detects whether requirement implies compound ALL_OF (AND) or ANY_OF (OR)."""
        t_lower = text.lower()
        if re.search(r'\b(?:or|either\s+.+\s+or|alternatively)\b', t_lower) or " / " in text:
            return MatchingLogic.ANY_OF
        elif re.search(r'\b(?:and|both|as\s+well\s+as)\b', t_lower) or " & " in text:
            return MatchingLogic.ALL_OF
        return MatchingLogic.SINGLE

    def _extract_compound_components(self, text: str, logic: MatchingLogic) -> List[str]:
        """Extracts component skills from a compound requirement."""
        if logic == MatchingLogic.ANY_OF:
            parts = re.split(r'\b(?:or)\b|/|;', text, flags=re.IGNORECASE)
            return [p.strip() for p in parts if len(p.strip()) >= 2]
        elif logic == MatchingLogic.ALL_OF:
            parts = re.split(r'\b(?:and|as\s+well\s+as)\b|&|;', text, flags=re.IGNORECASE)
            return [p.strip() for p in parts if len(p.strip()) >= 2]
        return []

    def _detect_requirement_type(self, text: str) -> RequirementType:
        text_lower = text.lower()
        if any(kw in text_lower for kw in [
            "bachelor", "master", "ph.d", "phd", "degree in", "b.s.", "m.s.",
            "b.tech", "m.tech", "diploma", "computer science or related"
        ]):
            return RequirementType.EDUCATION

        if any(kw in text_lower for kw in [
            "certified", "certification", "license", "pmp", "aws certified",
            "gcp certified", "scrum master certified", "cpa"
        ]):
            return RequirementType.CERTIFICATION

        if re.search(r'\b(?:\d+\+?|\d+\s*-\s*\d+)\s*(?:years?|yrs?)\b', text_lower):
            return RequirementType.EXPERIENCE

        if any(text_lower.startswith(prefix) for prefix in [
            "responsible for", "develop and maintain", "collaborate with",
            "design and implement", "lead the", "participate in", "work closely with"
        ]):
            return RequirementType.RESPONSIBILITY

        if self._is_behavioural_requirement(text):
            return RequirementType.BEHAVIOURAL

        return RequirementType.SKILL

    def _is_behavioural_requirement(self, text: str) -> bool:
        """
        Returns True if the requirement is a generic behavioural quality or soft skill
        that cannot be directly verified from a resume (e.g. communication, teamwork,
        problem-solving, attention to detail).

        These qualities are real requirements but:
        - Cannot be confirmed or denied from typical resume text
        - Their absence in a resume does NOT mean the candidate lacks them
        - They must NOT trigger automatic rejection on their own
        """
        text_lower = text.lower().strip()

        # Generic soft skill / behavioural keyword patterns
        BEHAVIOURAL_KEYWORDS = [
            "communication", "communicat",
            "teamwork", "team player", "team-player", "collaborate", "collaboration",
            "interpersonal", "cross-functional", "cross functional",
            "problem.solv", "problem solving", "analytical thinking",
            "critical thinking", "logical thinking",
            "attention to detail", "detail.oriented", "detail oriented",
            "self.starter", "self starter", "self-motivated", "motivated",
            "adaptab", "flexible", "flexibility",
            "time management", "prioritiz", "organisational", "organizational",
            "leadership", "initiative", "proactiv",
            "positive attitude", "work ethic", "enthusiasm",
            "fast learner", "quick learner", "eager to learn",
            "result.oriented", "results driven", "goal.oriented",
            "creativity", "creative thinking", "innovative thinking",
            "multitask", "multi-task", "work under pressure",
            "verbal", "written communication", "presentation skills",
        ]

        # If any behavioural keyword found AND no concrete technical skill keyword found
        has_behavioural = any(re.search(rf'\b{re.escape(kw)}', text_lower) for kw in BEHAVIOURAL_KEYWORDS)
        if not has_behavioural:
            return False

        # Check if there's also a concrete technical skill (in that case, it's a technical req with a soft component)
        TECHNICAL_INDICATORS = [
            r'\b(?:python|java|javascript|typescript|go|rust|c\+\+|c#|ruby|scala|kotlin|swift)\b',
            r'\b(?:sql|nosql|postgresql|mysql|mongodb|redis|elasticsearch|cassandra)\b',
            r'\b(?:aws|gcp|azure|kubernetes|docker|terraform|ci/cd|devops|mlops)\b',
            r'\b(?:react|angular|vue|node\.js|django|flask|fastapi|spring|rails)\b',
            r'\b(?:machine learning|deep learning|nlp|neural network|pytorch|tensorflow)\b',
            r'\b(?:git|linux|bash|shell|api|rest|graphql|microservice)\b',
        ]
        has_technical = any(re.search(pat, text_lower) for pat in TECHNICAL_INDICATORS)
        if has_technical:
            return False  # Technical requirement that mentions soft component

        return True

    def _extract_min_years(self, text: str) -> Optional[float]:
        match = re.search(r'\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b', text.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _detect_canonical_skill(self, text: str) -> Optional[str]:
        if not self.taxonomy:
            return None

        resolved = self.taxonomy.resolve_canonical(text)
        if resolved:
            return resolved

        all_terms = sorted(self.taxonomy.get_all_lookup_terms().keys(), key=len, reverse=True)
        norm_text = normalize_text(text)

        for term in all_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, norm_text):
                canonical = self.taxonomy.resolve_canonical(term)
                if canonical:
                    return canonical

        return None
