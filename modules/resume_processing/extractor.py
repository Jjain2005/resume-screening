import re
import os
import io
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import fitz  # PyMuPDF
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from .models import (
    ProcessedResume,
    ResumeSection,
    EvidenceSentence,
    SectionType
)

COMPOSITE_EXP_PROJ_PATTERN = re.compile(
    r'^(?:(?:work\s+|professional\s+)?experience\s+(?:&|and|/)\s+(?:key\s+|selected\s+|academic\s+|personal\s+|technical\s+)?projects|'
    r'(?:key\s+|selected\s+|academic\s+|personal\s+|technical\s+)?projects\s+(?:&|and|/)\s+(?:work\s+|professional\s+)?experience)[\s:]*$',
    re.IGNORECASE
)

SECTION_PATTERNS = [
    (SectionType.SUMMARY, re.compile(
        r'^(?:professional\s+summary|summary|profile|professional\s+profile|about\s+me|career\s+objective|objective|executive\s+summary|personal\s+statement|overview)[\s:]*$',
        re.IGNORECASE
    )),
    (SectionType.EXPERIENCE, re.compile(
        r'^(?:professional\s+|work\s+|relevant\s+|industry\s+|practical\s+)?(?:experience|history|employment|work\s+record|experience\s+history|career\s+history|professional\s+background|employment\s+history)[\s:]*$',
        re.IGNORECASE
    )),
    (SectionType.PROJECTS, re.compile(
        r'^(?:key\s+|selected\s+|academic\s+|recent\s+|personal\s+|technical\s+|notable\s+|major\s+|capstone\s+|open\s+source\s+)?(?:projects|project\s+experience|project\s+work|projects\s+undertaken)[\s:]*$',
        re.IGNORECASE
    )),
    (SectionType.SKILLS, re.compile(
        r'^(?:technical\s+|core\s+|key\s+|professional\s+|relevant\s+|it\s+)?(?:skills|competencies|technologies|tools|proficiencies|tech\s+stack|expertise|technical\s+expertise|skills\s+&\s+tools|skills\s+and\s+tools|skills\s+&\s+technologies|skills\s+and\s+technologies|tools\s+&\s+technologies|tools\s+and\s+technologies|programming\s+languages|core\s+strengths|core\s+competencies|areas\s+of\s+expertise)[\s:]*$',
        re.IGNORECASE
    )),
    (SectionType.EDUCATION, re.compile(
        r'^(?:education|academic\s+qualifications|academic\s+background|academics|educational\s+background|educational\s+qualifications|degrees\s+&\s+education)[\s:]*$',
        re.IGNORECASE
    )),
    (SectionType.CERTIFICATIONS, re.compile(
        r'^(?:certifications|certificates|licenses|credentials|courses\s+&\s+certifications|licenses\s+&\s+certifications|professional\s+certifications)[\s:]*$',
        re.IGNORECASE
    )),
    (SectionType.ACHIEVEMENTS, re.compile(
        r'^(?:achievements|awards|honors|accomplishments|awards\s+&\s+honors|honors\s+&\s+awards|key\s+achievements|publications|patents)[\s:]*$',
        re.IGNORECASE
    )),
]


def clean_text_preserving_structure(text: str) -> str:
    """Clean text while preserving sentence punctuation and bullet boundaries."""
    if not text:
        return ""
    # Normalize bullet points to a standard marker
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\u25CF\u25CB\u25A0\u25A1]', '\n• ', text)
    # Normalize unicode whitespaces
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    # Normalize varied dashes/hyphens
    text = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015]', '-', text)
    # Replace carriage returns
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with a single space (except newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def split_into_sentences(text: str, default_section: SectionType = SectionType.GENERAL, page_num: int = 1) -> List[EvidenceSentence]:
    """Split text into distinct sentences without truncating evidence context."""
    if not text:
        return []

    sentences: List[EvidenceSentence] = []
    # Split by explicit newlines or bullet points first
    raw_blocks = re.split(r'\n+|•', text)
    
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        
        # Avoid splitting common acronyms e.g. e.g., i.e., vs., Dr., Ph.D.
        # Regex looks for punctuation followed by space and uppercase letter
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', block)
        for part in parts:
            cleaned_part = part.strip()
            # Filter out very short noise fragments (e.g. single symbols, page numbers)
            if len(cleaned_part) >= 4 and not re.match(r'^\d+$', cleaned_part):
                sentences.append(EvidenceSentence(
                    text=cleaned_part,
                    section=default_section,
                    page_number=page_num
                ))
    return sentences

class ResumeExtractor:
    """
    Extracts text from PDF documents using PyMuPDF, with automatic scanned document
    detection, optional OCR fallback, section identification, and evidence preservation.
    """
    def __init__(self, ocr_min_chars: int = 60):
        self.ocr_min_chars = ocr_min_chars

    def extract_from_pdf(self, pdf_file_path_or_bytes, filename: str = "resume.pdf") -> ProcessedResume:
        processed = ProcessedResume(filename=filename)
        doc = None

        try:
            if isinstance(pdf_file_path_or_bytes, (str, os.PathLike)):
                doc = fitz.open(pdf_file_path_or_bytes)
            elif isinstance(pdf_file_path_or_bytes, bytes):
                doc = fitz.open(stream=pdf_file_path_or_bytes, filetype="pdf")
            elif hasattr(pdf_file_path_or_bytes, "read"):
                # File-like object (e.g. from Streamlit)
                file_bytes = pdf_file_path_or_bytes.read()
                doc = fitz.open(stream=file_bytes, filetype="pdf")
            else:
                raise ValueError("Unsupported input type for PDF extraction")

            processed.page_count = len(doc)
            raw_pages_text = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text("text").strip()

                # Check if page has insufficient extractable text (potential scanned page)
                if len(page_text) < self.ocr_min_chars:
                    ocr_text = self._attempt_ocr(page, page_idx + 1, processed)
                    if ocr_text:
                        page_text = ocr_text
                    else:
                        processed.extraction_warnings.append(
                            f"Page {page_idx + 1} has minimal extractable text ({len(page_text)} chars). Scanned document suspected."
                        )

                raw_pages_text.append(page_text)

            combined_raw = "\n\n".join(raw_pages_text)
            processed.raw_text = combined_raw
            processed.cleaned_text = clean_text_preserving_structure(combined_raw)

            if len(processed.cleaned_text) < 50:
                processed.extraction_success = False
                processed.extraction_warnings.append("Document text is nearly empty or unreadable.")

            # Extract candidate name candidate
            processed.candidate_name = self._extract_candidate_name(doc, processed.cleaned_text)

            # Segment sections and extract evidence sentences
            self._segment_sections(processed)

        except Exception as e:
            processed.extraction_success = False
            processed.extraction_warnings.append(f"Failed to extract PDF: {str(e)}")
        finally:
            if doc:
                doc.close()

        return processed

    def _attempt_ocr(self, page, page_num: int, processed: ProcessedResume) -> Optional[str]:
        """Attempts OCR on a page using PyMuPDF rendering + pytesseract if available."""
        processed.is_scanned = True
        if not PYTESSERACT_AVAILABLE:
            processed.extraction_warnings.append(
                f"Page {page_num} appears scanned, but pytesseract is not installed or configured."
            )
            return None

        try:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                processed.ocr_applied = True
                return ocr_text.strip()
        except Exception as e:
            processed.extraction_warnings.append(
                f"OCR attempt on page {page_num} failed: {str(e)}"
            )
        return None

    def _extract_candidate_name(self, doc, cleaned_text: str) -> Optional[str]:
        """Extract candidate name from first page lines or metadata."""
        # 1. Check PDF metadata
        meta_title = doc.metadata.get("author") or doc.metadata.get("title")
        if meta_title and len(meta_title.split()) in (2, 3, 4) and not any(kw in meta_title.lower() for kw in ["resume", "cv", "curriculum", "untitled"]):
            return meta_title.strip()

        # 2. Check top non-empty lines on the first page
        lines = [line.strip() for line in cleaned_text.split("\n") if line.strip()]
        for line in lines[:5]:
            # Avoid obvious section headers, emails, phones, or URLs
            if "@" in line or "http" in line or "+" in line or any(c.isdigit() for c in line):
                continue
            if any(h in line.lower() for h in ["resume", "curriculum vitae", "profile", "summary"]):
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                return line.strip()

        return "Candidate"

    def _segment_sections(self, processed: ProcessedResume):
        """Segment raw text into sections and attach evidence sentences with section tags."""
        lines = processed.cleaned_text.split("\n")
        current_section = SectionType.GENERAL
        current_heading = "General"
        is_composite_exp_proj = False
        section_lines: Dict[SectionType, List[str]] = {st: [] for st in SectionType}
        composite_lines: List[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line is a section heading
            matched_section = None
            is_composite = False
            if len(line_str) < 55:
                if COMPOSITE_EXP_PROJ_PATTERN.match(line_str):
                    is_composite = True
                else:
                    for sec_type, pattern in SECTION_PATTERNS:
                        if pattern.match(line_str):
                            matched_section = sec_type
                            break

            if is_composite:
                is_composite_exp_proj = True
                current_heading = line_str
            elif matched_section:
                is_composite_exp_proj = False
                current_section = matched_section
                current_heading = line_str
            else:
                if is_composite_exp_proj:
                    composite_lines.append(line_str)
                else:
                    section_lines[current_section].append(line_str)

        # Process composite lines if any
        if composite_lines:
            # Route lines within composite section to both Experience and Projects
            # to preserve evidence provenance for both verified tenure and project depth
            for cline in composite_lines:
                section_lines[SectionType.EXPERIENCE].append(cline)
                section_lines[SectionType.PROJECTS].append(cline)

        # Build ResumeSection objects and sentences
        all_sentences = []
        for sec_type, s_lines in section_lines.items():
            raw_sec_text = "\n".join(s_lines)
            if not raw_sec_text.strip():
                continue

            sentences = split_into_sentences(raw_sec_text, default_section=sec_type)
            all_sentences.extend(sentences)
            processed.sections[sec_type] = ResumeSection(
                section_type=sec_type,
                heading=sec_type.value,
                raw_text=raw_sec_text,
                sentences=sentences
            )

        processed.all_sentences = all_sentences

        # 3. Check extraction integrity and confidence
        self.check_extraction_confidence(processed)

        # 4. Calculate verified non-overlapping experience timeline
        self.calculate_verified_experience_years(processed)


    def check_extraction_confidence(self, processed: ProcessedResume):
        """
        Validates extraction integrity to detect scanned, empty, or broken PDFs.
        Marks low-confidence resumes so they are routed to manual review rather
        than falsely rejected for missing skills.
        """
        reasons = []
        confidence = 1.0

        if not processed.extraction_success:
            confidence = 0.1
            reasons.append("PDF extraction reported an unhandled error.")

        cleaned_len = len(processed.cleaned_text)
        word_count = len(processed.cleaned_text.split())

        if cleaned_len < 180:
            confidence = min(confidence, 0.25)
            reasons.append(f"Minimal text extracted ({cleaned_len} characters). Potential scanned image or vector icon layout.")
        elif cleaned_len < 350:
            confidence = min(confidence, 0.50)
            reasons.append(f"Low character volume ({cleaned_len} characters). Resume content may be clipped.")

        if word_count < 30:
            confidence = min(confidence, 0.25)
            reasons.append(f"Fewer than 30 readable words detected ({word_count} words).")

        if processed.is_scanned and not processed.ocr_applied:
            confidence = min(confidence, 0.20)
            reasons.append("Document appears to be a scanned image, but OCR could not be applied.")

        # Check if structural sections were identified
        detected_sections = [st for st, sec in processed.sections.items() if sec.sentences and st != SectionType.GENERAL]
        if not detected_sections and word_count >= 30:
            confidence = min(confidence, 0.65)
            reasons.append("No standard resume section headings detected (unstructured layout).")

        processed.extraction_confidence = round(confidence, 2)
        processed.confidence_reasons = reasons
        processed.is_low_confidence = (confidence < 0.60)
        return processed.extraction_confidence, processed.is_low_confidence, processed.confidence_reasons

    def calculate_verified_experience_years(
        self,
        processed: Union[ProcessedResume, str],
        relevant_keywords: Optional[List[str]] = None
    ) -> float:
        """
        Extracts employment date intervals, merges overlapping employment periods
        to prevent double-counting, filters out unrelated experience, and handles
        missing or unclear dates as unverifiable.
        """
        if isinstance(processed, str):
            text_input = processed
            dummy_resume = ProcessedResume(filename="raw_input.txt", raw_text=text_input)
            dummy_resume.all_sentences = [
                EvidenceSentence(text=line.strip(), section=SectionType.EXPERIENCE)
                for line in text_input.splitlines() if line.strip()
            ]
            dummy_resume.sections[SectionType.EXPERIENCE].sentences = list(dummy_resume.all_sentences)
            processed = dummy_resume

        month_map = {
            "jan": 0, "january": 0, "feb": 1, "february": 1, "mar": 2, "march": 2,
            "apr": 3, "april": 3, "may": 4, "jun": 5, "june": 5, "jul": 6, "july": 6,
            "aug": 7, "august": 7, "sep": 8, "sept": 8, "september": 8,
            "oct": 9, "october": 9, "nov": 10, "november": 10, "dec": 11, "december": 11
        }
        now = datetime.now()
        current_year_float = now.year + (now.month - 1) / 12.0  # Dynamic current date baseline for "Present / Current"

        # Academic / study indicators to strictly ignore
        academic_filter_regex = re.compile(
            r'\b(?:bachelor|master|phd|doctorate|b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?s|m\.?s|b\.?sc|m\.?sc|bca|mca|bba|mba|diploma|degree|college|university|school|high\s+school|matriculation|secondary|hsc|ssc|cbse|icse|cgpa|gpa|percentage|undergraduate|postgraduate|graduat(?:ed|ion)|curriculum|syllabus)\b',
            re.IGNORECASE
        )

        # Search ONLY in Experience and explicit work/internship project sections.
        # NEVER fall back to Education or general sentences to avoid capturing study/college years.
        exp_sentences = processed.get_sentences_by_section(SectionType.EXPERIENCE)
        proj_sentences = processed.get_sentences_by_section(SectionType.PROJECTS)

        work_intern_regex = re.compile(
            r'\b(?:intern|internship|co-?op|trainee|apprentice|employment|client\s+work|freelance)\b',
            re.IGNORECASE
        )
        filtered_proj = [
            s for s in proj_sentences
            if work_intern_regex.search(s.text) and not academic_filter_regex.search(s.text)
        ]

        search_sentences = exp_sentences + filtered_proj

        def _parse_single_date(date_str: str, is_end: bool) -> Optional[float]:
            s = date_str.strip()
            if not s:
                return None
            if is_end and any(kw in s.lower() for kw in ["present", "current", "now", "ongoing"]):
                return current_year_float

            # 1. ISO format: YYYY-MM or YYYY-MM-DD or YYYY/MM or YYYY.MM
            iso_m = re.search(r'\b(19\d{2}|20\d{2})[\-\/\.](0?[1-9]|1[0-2])(?:\b|[\-\/\.])', s)
            if iso_m:
                y = int(iso_m.group(1))
                m = int(iso_m.group(2))
                return float(y + (m - 1) / 12.0)

            # 2. Month name + Year (e.g. Jan 2021, January 2021, Jan '21)
            for m_name, m_val in month_map.items():
                if re.search(r'\b' + re.escape(m_name) + r'[a-z]*\b', s.lower()):
                    y_m = re.search(r'(?:19|20)?(\d{2})\b', s)
                    if y_m:
                        digits = int(y_m.group(1))
                        y = 2000 + digits if digits < 50 else 1900 + digits
                        return float(y + m_val / 12.0)

            # 3. Numeric Month/Year: MM/YYYY or MM-YYYY or MM.YYYY
            num_m = re.search(r'\b(0?[1-9]|1[0-2])[\-\/\.](19\d{2}|20\d{2})\b', s)
            if num_m:
                m = int(num_m.group(1))
                y = int(num_m.group(2))
                return float(y + (m - 1) / 12.0)

            # 4. Year only: YYYY
            y_m = re.search(r'\b(19\d{2}|20\d{2})\b', s)
            if y_m:
                y = int(y_m.group(1))
                return float(y)

            return None

        date_range_regex = re.compile(
            r'((?:(?:19|20)\d{2}[\-\/\.](?:0?[1-9]|1[0-2])(?:\-[\d]{1,2})?)|'
            r'(?:(?:0?[1-9]|1[0-2])[\-\/\.](?:19|20)\d{2})|'
            r'(?:[a-zA-Z]{3,9}\.?\s*[\']?(?:19|20)?\d{2})|'
            r'(?:(?:19|20)\d{2}))'
            r'\s*(?:–|—|to|\s+-\s+|-\s*|\s*-\s*|\s+\/\s+)\s*'
            r'((?:(?:19|20)\d{2}[\-\/\.](?:0?[1-9]|1[0-2])(?:\-[\d]{1,2})?)|'
            r'(?:(?:0?[1-9]|1[0-2])[\-\/\.](?:19|20)\d{2})|'
            r'(?:[a-zA-Z]{3,9}\.?\s*[\']?(?:19|20)?\d{2})|'
            r'(?:(?:19|20)\d{2})|'
            r'(?:Present|Current|Now|Ongoing))',
            re.IGNORECASE
        )

        raw_intervals: List[Tuple[float, float, str, bool]] = []
        clean_kws = [k.lower().strip() for k in relevant_keywords] if relevant_keywords else []

        for idx, s in enumerate(search_sentences):
            if s.section == SectionType.EDUCATION or academic_filter_regex.search(s.text):
                continue
            matches = date_range_regex.finditer(s.text)
            for m in matches:
                start_str, end_str = m.group(1).strip(), m.group(2).strip()
                start_float = _parse_single_date(start_str, is_end=False)
                end_float = _parse_single_date(end_str, is_end=True)

                if start_float is None or end_float is None:
                    continue
                if end_float >= start_float and (end_float - start_float) <= 35.0:
                    is_rel = True
                    if clean_kws:
                        s_text_lower = s.text.lower()
                        is_rel = any(re.search(r'\b' + re.escape(kw) + r'\b', s_text_lower) for kw in clean_kws)
                        if not is_rel:
                            # Check subsequent bullet points for this position, stopping at the next job/date boundary
                            for next_s in search_sentences[idx + 1: min(len(search_sentences), idx + 6)]:
                                if date_range_regex.search(next_s.text):
                                    break
                                if any(re.search(r'\b' + re.escape(kw) + r'\b', next_s.text.lower()) for kw in clean_kws):
                                    is_rel = True
                                    break
                    raw_intervals.append((start_float, end_float, m.group(0), is_rel))

        if not raw_intervals:
            processed.verified_experience_years = 0.0
            processed.relevant_experience_years = 0.0
            processed.employment_periods = []
            processed.timeline_summary = "No verifiable employment date intervals found."
            return 0.0

        # Sort intervals by start date for interval merging
        raw_intervals.sort(key=lambda x: (x[0], x[1]))

        # Merge overlapping intervals (Interval Scheduling Algorithm)
        merged_intervals: List[Tuple[float, float]] = []
        for start, end, _, _ in raw_intervals:
            if not merged_intervals:
                merged_intervals.append((start, end))
            else:
                prev_start, prev_end = merged_intervals[-1]
                # If current interval overlaps or directly abuts the previous one (within 2 months buffer)
                if start <= prev_end + 0.16:
                    merged_intervals[-1] = (prev_start, max(prev_end, end))
                else:
                    merged_intervals.append((start, end))

        # Sum total non-overlapping duration
        total_duration = sum(end - start for start, end in merged_intervals)
        verified_years = round(total_duration, 1)

        # Merge relevant intervals if keywords provided
        if clean_kws:
            rel_raw = [(start, end) for start, end, _, is_rel in raw_intervals if is_rel]
            rel_raw.sort(key=lambda x: (x[0], x[1]))
            merged_rel: List[Tuple[float, float]] = []
            for start, end in rel_raw:
                if not merged_rel:
                    merged_rel.append((start, end))
                else:
                    p_start, p_end = merged_rel[-1]
                    if start <= p_end + 0.16:
                        merged_rel[-1] = (p_start, max(p_end, end))
                    else:
                        merged_rel.append((start, end))
            relevant_years = round(sum(end - start for start, end in merged_rel), 1)
        else:
            relevant_years = verified_years

        processed.verified_experience_years = verified_years
        processed.relevant_experience_years = relevant_years
        processed.employment_periods = [
            {"start": round(s, 2), "end": round(e, 2), "duration_years": round(e - s, 1)}
            for s, e in merged_intervals
        ]
        if clean_kws and relevant_years != verified_years:
            processed.timeline_summary = f"{verified_years} verified years ({relevant_years} relevant) across {len(merged_intervals)} non-overlapping period(s)."
        else:
            processed.timeline_summary = f"{verified_years} verified years across {len(merged_intervals)} non-overlapping period(s)."
        return verified_years

    def calculate_relevant_experience_years(
        self,
        processed: Union[ProcessedResume, str],
        relevant_keywords: List[str]
    ) -> float:
        """Calculates non-overlapping years of experience relevant to specific keywords/skills."""
        self.calculate_verified_experience_years(processed, relevant_keywords=relevant_keywords)
        if isinstance(processed, ProcessedResume):
            return processed.relevant_experience_years
        return 0.0
