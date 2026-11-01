import re
from typing import Optional, Tuple, List
from ..knowledge_base.taxonomy import SkillTaxonomy, normalize_text
from ..resume_processing.models import ProcessedResume, EvidenceSentence

def word_boundary_regex(term: str) -> re.Pattern:
    """Creates a word-boundary regex that safely handles symbols like C++, C#, .NET."""
    escaped = re.escape(term.strip())
    # If term starts or ends with alphanumeric, enforce word boundary
    prefix = r'(?<![A-Za-z0-9])' if term[0].isalnum() else ''
    suffix = r'(?![A-Za-z0-9])' if term[-1].isalnum() else ''
    return re.compile(f'{prefix}{escaped}{suffix}', re.IGNORECASE)

class ExactMatcher:
    """
    Identifies exact canonical and alias matches within candidate resumes
    with strict word-boundary protection.
    """
    def __init__(self, taxonomy: SkillTaxonomy):
        self.taxonomy = taxonomy

    def match_skill(
        self,
        target_skill: str,
        resume: ProcessedResume
    ) -> Optional[Tuple[str, str, EvidenceSentence]]:
        """
        Looks for exact match of target_skill (canonical or any of its aliases) in resume sentences.
        Returns (matched_term, match_type, evidence_sentence) or None.
        match_type is 'Exact Canonical' or 'Exact Alias'.
        """
        canonical = self.taxonomy.resolve_canonical(target_skill) or target_skill
        skill_entry = self.taxonomy.get_skill(canonical)

        terms_to_check = []
        # Canonical first
        terms_to_check.append((canonical, "Exact Canonical"))

        # Aliases
        if skill_entry:
            for alias in skill_entry.aliases:
                if alias.lower() != canonical.lower():
                    terms_to_check.append((alias, "Exact Alias"))
        else:
            # If target_skill itself wasn't canonical, add it
            if target_skill.lower() != canonical.lower():
                terms_to_check.append((target_skill, "Exact Alias"))

        # Search prioritized sections first: EXPERIENCE, PROJECTS, SKILLS, then others
        ordered_sentences = self._prioritize_sentences(resume.all_sentences)

        for term, match_type in terms_to_check:
            pattern = word_boundary_regex(term)
            for sentence in ordered_sentences:
                if pattern.search(sentence.text):
                    return (term, match_type, sentence)

        return None

    def _prioritize_sentences(self, sentences: List[EvidenceSentence]) -> List[EvidenceSentence]:
        # Experience and Projects contain higher quality evidence than raw lists
        priority_order = {"Experience": 1, "Projects": 2, "Summary": 3, "Achievements": 4, "Skills": 5, "General": 6, "Education": 7, "Certifications": 8}
        return sorted(sentences, key=lambda s: priority_order.get(s.section.value, 99))
