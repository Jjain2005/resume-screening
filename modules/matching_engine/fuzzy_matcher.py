from typing import Optional, Tuple, List
from rapidfuzz import fuzz
from ..knowledge_base.taxonomy import SkillTaxonomy, normalize_text
from ..resume_processing.models import ProcessedResume, EvidenceSentence

class FuzzyMatcher:
    """
    Identifies minor typographical and formatting variations in skills using RapidFuzz
    with strict length safeguards and configurable thresholds.
    """
    def __init__(self, taxonomy: SkillTaxonomy, threshold: float = 85.0):
        self.taxonomy = taxonomy
        self.threshold = threshold

    def match_skill(
        self,
        target_skill: str,
        resume: ProcessedResume,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[str, float, EvidenceSentence]]:
        """
        Scans resume sentences for fuzzy token matches with target_skill.
        Returns (matched_phrase, score, evidence_sentence) or None.
        """
        th = threshold if threshold is not None else self.threshold
        canonical = self.taxonomy.resolve_canonical(target_skill) or target_skill
        target_norm = normalize_text(target_skill)

        # Do not fuzzy match extremely short acronyms (e.g. 'C', 'R', 'Go', 'AI', 'UI')
        # because fuzzy distance on length <= 3 causes massive false positive rates!
        if len(target_norm) < 4:
            return None

        best_score = 0.0
        best_match = None
        target_word_count = len(target_norm.split())

        for sentence in resume.all_sentences:
            words = sentence.text.split()
            # Sliding window of length ~ target word count (+/- 1)
            for window_size in {max(1, target_word_count - 1), target_word_count, target_word_count + 1}:
                for i in range(len(words) - window_size + 1):
                    candidate_phrase = " ".join(words[i:i + window_size])
                    cand_norm = normalize_text(candidate_phrase)
                    
                    # Prevent matching substring that is identical to a DIFFERENT skill (e.g. Java vs JavaScript)
                    resolved_cand = self.taxonomy.resolve_canonical(cand_norm)
                    if resolved_cand and resolved_cand.lower() != canonical.lower():
                        continue

                    # RapidFuzz token_sort_ratio or ratio
                    score = fuzz.ratio(target_norm, cand_norm)
                    if score >= th and score > best_score:
                        # Extra guard: difference in length should not be > 30%
                        len_ratio = min(len(target_norm), len(cand_norm)) / max(len(target_norm), len(cand_norm))
                        if len_ratio >= 0.7:
                            best_score = score
                            best_match = (candidate_phrase, score, sentence)

        return best_match
