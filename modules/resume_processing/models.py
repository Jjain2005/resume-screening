from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class SectionType(str, Enum):
    SUMMARY = "Summary"
    SKILLS = "Skills"
    EXPERIENCE = "Experience"
    PROJECTS = "Projects"
    EDUCATION = "Education"
    CERTIFICATIONS = "Certifications"
    ACHIEVEMENTS = "Achievements"
    GENERAL = "General"

@dataclass
class EvidenceSentence:
    text: str
    section: SectionType
    page_number: int = 1
    char_start: int = 0
    char_end: int = 0

@dataclass
class ResumeSection:
    section_type: SectionType
    heading: str
    raw_text: str
    sentences: List[EvidenceSentence] = field(default_factory=list)

@dataclass
class ProcessedResume:
    filename: str
    candidate_name: Optional[str] = None
    raw_text: str = ""
    cleaned_text: str = ""
    page_count: int = 0
    is_scanned: bool = False
    ocr_applied: bool = False
    extraction_success: bool = True
    extraction_warnings: List[str] = field(default_factory=list)
    extraction_confidence: float = 1.0
    is_low_confidence: bool = False
    confidence_reasons: List[str] = field(default_factory=list)
    verified_experience_years: float = 0.0
    employment_periods: List[Dict] = field(default_factory=list)
    timeline_summary: str = ""
    sections: Dict[SectionType, ResumeSection] = field(default_factory=dict)
    all_sentences: List[EvidenceSentence] = field(default_factory=list)
    extracted_skills: List[str] = field(default_factory=list)

    def get_sentences_by_section(self, section_type: SectionType) -> List[EvidenceSentence]:
        if section_type in self.sections:
            return self.sections[section_type].sentences
        return [s for s in self.all_sentences if s.section == section_type]
