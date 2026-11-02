import re
from typing import List, Optional
from .models import JobDescription, RequirementItem, RequirementType, RequirementPriority
from ..knowledge_base.taxonomy import SkillTaxonomy, normalize_text

class JobDescriptionParser:
    """
    Parses unstructured Job Description text into structured requirements,
    categorizing into Required/Preferred and detecting Skills, Experience,
    Education, Certifications, and Responsibilities.
    """
    def __init__(self, taxonomy: Optional[SkillTaxonomy] = None):
        self.taxonomy = taxonomy

    def parse(
        self,
        text: str,
        jd_id: str = "jd_1",
        default_title: str = "Job Position",
        explicit_title: Optional[str] = None,
        **kwargs
    ) -> JobDescription:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        raw_title = explicit_title or kwargs.get("explicit_title") or default_title
        title = raw_title.strip() if raw_title and raw_title.strip() else "Job Position"
        title_inferred_from_first_line = False

        # Try to infer title from first line if not explicitly provided
        if not raw_title or raw_title == "Job Position":
            if lines:
                first_line = lines[0]
                if len(first_line) < 60 and not any(kw in first_line.lower() for kw in ["requirement", "qualification", "responsibilit", "description", "overview", "about", "looking for", "we are"]):
                    title = first_line
                    title_inferred_from_first_line = True

        requirements: List[RequirementItem] = []
        req_counter = 1

        # Check if the JD has recognizable section headers
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
        current_priority = RequirementPriority.REQUIRED
        current_section_type = "default"

        for idx, line in enumerate(lines):
            # Skip the first line if it was used as the title
            if idx == 0 and title_inferred_from_first_line:
                continue

            line_lower = line.lower().strip()

            # Detect section heading transitions
            if any(kw in line_lower for kw in ["preferred", "nice to have", "bonus", "good to have", "desired", "plus", "optional"]) and len(line_lower) < 50:
                current_priority = RequirementPriority.PREFERRED
                current_section_type = "preferred"
                in_preamble = False
                continue
            elif any(kw in line_lower for kw in ["requirements", "required", "minimum qualifications", "must have", "basic qualifications", "what you need", "what we're looking for"]) and len(line_lower) < 50:
                current_priority = RequirementPriority.REQUIRED
                current_section_type = "required"
                in_preamble = False
                continue
            elif any(kw in line_lower for kw in ["responsibilities", "what you will do", "duties", "role overview", "day to day", "key responsibilities"]) and len(line_lower) < 50:
                # Responsibilities provide context without acting as mandatory blockers
                current_priority = RequirementPriority.PREFERRED
                current_section_type = "responsibilities"
                in_preamble = False
                continue
            elif any(kw in line_lower for kw in ["about us", "company overview", "about the company", "who we are", "about the role"]) and len(line_lower) < 50:
                in_preamble = True
                continue

            # If still in preamble before first section header, do not treat as requirements
            if in_preamble:
                # Only keep line if it has explicit qualification keywords
                if not any(qkw in line_lower for qkw in ["must have", "minimum of", "required:", "qualifications:"]):
                    continue

            # Strip leading bullets, numbers, hyphens without removing digits from e.g. "4+ years"
            cleaned_line = re.sub(r'^[\s•\-\*\u2022\u25E6\u25AA\u25CF]+\s*', '', line)
            cleaned_line = re.sub(r'^\(?\d+[\.\)]\s*', '', cleaned_line).strip()
            if not cleaned_line or len(cleaned_line) < 3:
                continue

            # Skip lines that are just section headers like "Qualifications:"
            if cleaned_line.endswith(":") and len(cleaned_line) < 30:
                continue

            # Check if this line is a comma-separated skill list e.g. "Skills: Python, Docker, SQL, Git"
            list_prefix_match = re.match(r'^(?:skills|technologies|tools|competencies)[\s:]+(.+)$', cleaned_line, re.IGNORECASE)
            if list_prefix_match:
                sub_items = [s.strip() for s in re.split(r'[,;•|]', list_prefix_match.group(1)) if s.strip()]
                for sub in sub_items:
                    if len(sub) >= 2:
                        canon = self._detect_canonical_skill(sub)
                        requirements.append(RequirementItem(
                            id=f"req_{req_counter}",
                            text=sub,
                            req_type=RequirementType.SKILL,
                            priority=current_priority,
                            canonical_skill=canon,
                            min_years=None,
                            weight=1.5 if current_priority == RequirementPriority.REQUIRED else 1.0
                        ))
                        req_counter += 1
                continue

            # If the line is a long paragraph (> 140 chars) without bullets, split into sentences
            sub_sentences = [cleaned_line]
            if len(cleaned_line) > 140 and ("." in cleaned_line or ";" in cleaned_line):
                split_parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', cleaned_line) if len(p.strip()) >= 10]
                if len(split_parts) > 1:
                    sub_sentences = split_parts

            for sentence_text in sub_sentences:
                req_type = self._detect_requirement_type(sentence_text)
                if current_section_type == "responsibilities":
                    req_type = RequirementType.RESPONSIBILITY
                    item_priority = RequirementPriority.PREFERRED
                    weight = 0.5
                else:
                    item_priority = current_priority
                    weight = 1.5 if item_priority == RequirementPriority.REQUIRED else 1.0

                min_years = self._extract_min_years(sentence_text) if req_type == RequirementType.EXPERIENCE else None
                canonical_skill = self._detect_canonical_skill(sentence_text)

                if canonical_skill and req_type not in (RequirementType.EDUCATION, RequirementType.CERTIFICATION, RequirementType.RESPONSIBILITY):
                    req_type = RequirementType.SKILL

                item = RequirementItem(
                    id=f"req_{req_counter}",
                    text=sentence_text,
                    req_type=req_type,
                    priority=item_priority,
                    canonical_skill=canonical_skill,
                    min_years=min_years,
                    weight=weight
                )
                requirements.append(item)
                req_counter += 1


        return JobDescription(
            id=jd_id,
            title=title,
            raw_text=text,
            requirements=requirements
        )

    def _detect_requirement_type(self, text: str) -> RequirementType:
        text_lower = text.lower()
        # Education patterns
        if any(kw in text_lower for kw in [
            "bachelor", "master", "ph.d", "phd", "degree in", "b.s.", "m.s.",
            "b.tech", "m.tech", "diploma", "computer science or related"
        ]):
            return RequirementType.EDUCATION

        # Certification patterns
        if any(kw in text_lower for kw in [
            "certified", "certification", "license", "pmp", "aws certified",
            "gcp certified", "scrum master certified", "cpa"
        ]):
            return RequirementType.CERTIFICATION

        # Experience patterns
        if re.search(r'\b(?:\d+\+?|\d+\s*-\s*\d+)\s*(?:years?|yrs?)\b', text_lower):
            return RequirementType.EXPERIENCE

        # Responsibilities patterns
        if any(text_lower.startswith(prefix) for prefix in [
            "responsible for", "develop and maintain", "collaborate with",
            "design and implement", "lead the", "participate in", "work closely with"
        ]):
            return RequirementType.RESPONSIBILITY

        return RequirementType.SKILL

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

        # Direct canonical resolution if text is short
        resolved = self.taxonomy.resolve_canonical(text)
        if resolved:
            return resolved

        # Look for taxonomy skills within the requirement text (longest match first)
        all_terms = sorted(self.taxonomy.get_all_lookup_terms().keys(), key=len, reverse=True)
        norm_text = normalize_text(text)

        for term in all_terms:
            # Match whole words only
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, norm_text):
                canonical = self.taxonomy.resolve_canonical(term)
                if canonical:
                    return canonical

        return None
