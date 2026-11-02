import re
from typing import Optional
from .models import EvidenceStrength
from ..resume_processing.models import SectionType, EvidenceSentence

ACTION_VERBS = {
    "developed", "built", "designed", "implemented", "deployed", "architected",
    "engineered", "optimized", "created", "led", "managed", "trained", "delivered",
    "automated", "reduced", "increased", "integrated", "spearheaded", "fine-tuned",
    "scaled", "analyzed", "authored", "orchestrated", "refactored"
}

QUANTIFIABLE_PATTERN = re.compile(
    r'(\b\d+%\b|\b\d+x\b|\$\d+|\b\d+\s*(?:users|clients|requests|qps|million|billion|models|records|nodes)\b)',
    re.IGNORECASE
)

class ContextualMatcher:
    """
    Evaluates the contextual depth and implementation strength of candidate evidence sentences.
    Differentiates between a bare skill listing vs. proven execution in work/projects.
    """
    @staticmethod
    def evaluate_evidence_strength(sentence: Optional[EvidenceSentence]) -> EvidenceStrength:
        if not sentence or not sentence.text:
            return EvidenceStrength.NONE

        sec = sentence.section
        text = sentence.text.lower()
        words = set(re.findall(r'\b[a-z]+\b', text))

        has_action_verb = bool(words & ACTION_VERBS)
        has_quantifiable = bool(QUANTIFIABLE_PATTERN.search(text))
        is_detailed_sentence = len(sentence.text.split()) >= 8

        # Check if the sentence is purely a listed skill entry (e.g. comma/colon separated skills)
        is_list_format = bool(
            re.match(r'^(?:technologies|tech\s+stack|tools|skills|languages|frameworks|environment)[\s:]+', text)
            or (not has_action_verb and ("," in text or "|" in text or "/" in text or "•" in text))
            or (not has_action_verb and len(sentence.text.split()) <= 4)
        )

        # Section-specific evaluation
        if sec in (SectionType.EXPERIENCE, SectionType.PROJECTS):
            if is_list_format and not has_action_verb:
                return EvidenceStrength.LISTED_ONLY
            if has_action_verb and (has_quantifiable or is_detailed_sentence):
                return EvidenceStrength.STRONG_IMPLEMENTATION
            elif sec == SectionType.EXPERIENCE:
                return EvidenceStrength.EXPERIENCE_EVIDENCE
            else:
                return EvidenceStrength.PROJECT_EVIDENCE

        elif sec == SectionType.SKILLS:
            # If a sentence filed under Skills actually describes demonstrated project execution
            if has_action_verb and (has_quantifiable or is_detailed_sentence):
                return EvidenceStrength.STRONG_IMPLEMENTATION
            elif has_action_verb:
                return EvidenceStrength.PROJECT_EVIDENCE
            return EvidenceStrength.LISTED_ONLY

        elif sec in (SectionType.SUMMARY, SectionType.ACHIEVEMENTS):
            if has_action_verb and is_detailed_sentence:
                return EvidenceStrength.EXPERIENCE_EVIDENCE
            return EvidenceStrength.MENTIONED

        # General / Unstructured sections
        if has_action_verb and (has_quantifiable or is_detailed_sentence):
            return EvidenceStrength.STRONG_IMPLEMENTATION
        elif has_action_verb:
            return EvidenceStrength.PROJECT_EVIDENCE

        return EvidenceStrength.MENTIONED if not is_list_format else EvidenceStrength.LISTED_ONLY

