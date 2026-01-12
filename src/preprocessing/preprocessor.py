"""
Clinical Text Preprocessing Pipeline

Handles the unique challenges of clinical text:
- Section segmentation (HPI, ROS, Assessment, Plan)
- Abbreviation expansion
- Negation detection
- Temporal expression normalization
- Sentence boundary detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ClinicalSection(str, Enum):
    """Standard clinical note sections."""
    
    CHIEF_COMPLAINT = "chief_complaint"
    HISTORY_PRESENT_ILLNESS = "hpi"
    PAST_MEDICAL_HISTORY = "pmh"
    MEDICATIONS = "medications"
    ALLERGIES = "allergies"
    FAMILY_HISTORY = "family_history"
    SOCIAL_HISTORY = "social_history"
    REVIEW_OF_SYSTEMS = "ros"
    PHYSICAL_EXAM = "physical_exam"
    ASSESSMENT = "assessment"
    PLAN = "plan"
    LABS = "labs"
    IMAGING = "imaging"
    UNKNOWN = "unknown"


@dataclass
class Section:
    """A segmented section of clinical text."""
    
    section_type: ClinicalSection
    header: str
    content: str
    start_pos: int
    end_pos: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "section_type": self.section_type.value,
            "header": self.header,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "length": len(self.content),
        }


@dataclass
class Sentence:
    """A sentence from clinical text."""
    
    text: str
    start: int
    end: int
    section: ClinicalSection | None = None
    negated: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "section": self.section.value if self.section else None,
            "negated": self.negated,
        }


@dataclass
class PreprocessedDocument:
    """Result of clinical text preprocessing."""
    
    document_id: str
    original_text: str
    cleaned_text: str
    sections: list[Section]
    sentences: list[Sentence]
    abbreviations_expanded: int
    processing_time_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "original_length": len(self.original_text),
            "cleaned_length": len(self.cleaned_text),
            "num_sections": len(self.sections),
            "num_sentences": len(self.sentences),
            "abbreviations_expanded": self.abbreviations_expanded,
            "processing_time_ms": self.processing_time_ms,
            "sections": [s.to_dict() for s in self.sections],
        }


class ClinicalPreprocessorConfig(BaseModel):
    """Configuration for clinical text preprocessing."""
    
    expand_abbreviations: bool = True
    detect_negation: bool = True
    segment_sections: bool = True
    normalize_whitespace: bool = True
    lowercase_output: bool = False
    remove_phi_markers: bool = True
    sentence_max_length: int = Field(default=1000, ge=10)


class ClinicalAbbreviations:
    """Common clinical abbreviations with expansions."""
    
    ABBREVIATIONS = {
        # Common abbreviations
        "pt": "patient",
        "pts": "patients",
        "hx": "history",
        "dx": "diagnosis",
        "tx": "treatment",
        "rx": "prescription",
        "sx": "symptoms",
        "fx": "fracture",
        "bx": "biopsy",
        "cx": "culture",
        
        # Vital signs
        "bp": "blood pressure",
        "hr": "heart rate",
        "rr": "respiratory rate",
        "t": "temperature",
        "o2": "oxygen",
        "sat": "saturation",
        
        # Medical history
        "pmh": "past medical history",
        "psh": "past surgical history",
        "fh": "family history",
        "sh": "social history",
        "ros": "review of systems",
        "hpi": "history of present illness",
        "cc": "chief complaint",
        
        # Physical exam
        "pe": "physical exam",
        "heent": "head eyes ears nose throat",
        "cv": "cardiovascular",
        "resp": "respiratory",
        "gi": "gastrointestinal",
        "gu": "genitourinary",
        "neuro": "neurological",
        "msk": "musculoskeletal",
        "ext": "extremities",
        
        # Common conditions
        "dm": "diabetes mellitus",
        "htn": "hypertension",
        "cad": "coronary artery disease",
        "chf": "congestive heart failure",
        "copd": "chronic obstructive pulmonary disease",
        "ckd": "chronic kidney disease",
        "aki": "acute kidney injury",
        "mi": "myocardial infarction",
        "cva": "cerebrovascular accident",
        "dvt": "deep vein thrombosis",
        "pe": "pulmonary embolism",
        "uti": "urinary tract infection",
        "uri": "upper respiratory infection",
        
        # Timing/frequency
        "qd": "once daily",
        "bid": "twice daily",
        "tid": "three times daily",
        "qid": "four times daily",
        "prn": "as needed",
        "po": "by mouth",
        "iv": "intravenous",
        "im": "intramuscular",
        "sq": "subcutaneous",
        "ac": "before meals",
        "pc": "after meals",
        "hs": "at bedtime",
        
        # Lab values
        "wbc": "white blood cell",
        "rbc": "red blood cell",
        "hgb": "hemoglobin",
        "hct": "hematocrit",
        "plt": "platelet",
        "bmp": "basic metabolic panel",
        "cmp": "comprehensive metabolic panel",
        "cbc": "complete blood count",
        "lfts": "liver function tests",
        "ua": "urinalysis",
        "abg": "arterial blood gas",
        
        # Other common
        "w/": "with",
        "w/o": "without",
        "b/l": "bilateral",
        "c/o": "complaining of",
        "s/p": "status post",
        "r/o": "rule out",
        "f/u": "follow up",
        "d/c": "discharge",
        "yo": "year old",
        "y/o": "year old",
    }
    
    @classmethod
    def expand(cls, text: str) -> tuple[str, int]:
        """
        Expand abbreviations in text.
        
        Returns (expanded_text, num_expansions)
        """
        count = 0
        result = text
        
        for abbr, expansion in cls.ABBREVIATIONS.items():
            # Word boundary matching
            pattern = rf'\b{re.escape(abbr)}\b'
            matches = re.findall(pattern, result, re.IGNORECASE)
            if matches:
                count += len(matches)
                result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
        
        return result, count


class NegationDetector:
    """Detect negation in clinical text."""
    
    # Negation cues
    NEGATION_CUES = [
        "no", "not", "none", "neither", "never", "nobody",
        "denies", "denied", "negative", "absent", "without",
        "rule out", "ruled out", "r/o", "fails to", "failed to",
        "free of", "lack of", "unremarkable", "normal",
    ]
    
    # Scope terminators (words that end negation scope)
    TERMINATORS = [
        "but", "however", "although", "except", "apart from",
        "nonetheless", "yet", "still", "instead",
    ]
    
    @classmethod
    def detect(cls, sentence: str) -> bool:
        """Check if sentence contains negation."""
        sentence_lower = sentence.lower()
        
        for cue in cls.NEGATION_CUES:
            if cue in sentence_lower:
                return True
        
        return False
    
    @classmethod
    def get_negation_scope(cls, sentence: str) -> list[tuple[int, int]]:
        """
        Get character ranges that are under negation scope.
        
        Returns list of (start, end) tuples.
        """
        scopes = []
        sentence_lower = sentence.lower()
        
        for cue in cls.NEGATION_CUES:
            for match in re.finditer(rf'\b{re.escape(cue)}\b', sentence_lower):
                start = match.end()
                
                # Find terminator or end of sentence
                end = len(sentence)
                for term in cls.TERMINATORS:
                    term_match = re.search(rf'\b{re.escape(term)}\b', sentence_lower[start:])
                    if term_match:
                        end = min(end, start + term_match.start())
                
                # Also terminate at punctuation
                punct_match = re.search(r'[.;:]', sentence[start:])
                if punct_match:
                    end = min(end, start + punct_match.start())
                
                if start < end:
                    scopes.append((start, end))
        
        return scopes


class SectionSegmenter:
    """Segment clinical notes into standard sections."""
    
    # Section header patterns
    SECTION_PATTERNS = {
        ClinicalSection.CHIEF_COMPLAINT: [
            r"chief\s+complaint[s]?:?",
            r"cc:?",
            r"reason\s+for\s+visit:?",
        ],
        ClinicalSection.HISTORY_PRESENT_ILLNESS: [
            r"history\s+of\s+present\s+illness:?",
            r"hpi:?",
            r"present\s+illness:?",
        ],
        ClinicalSection.PAST_MEDICAL_HISTORY: [
            r"past\s+medical\s+history:?",
            r"pmh:?",
            r"medical\s+history:?",
        ],
        ClinicalSection.MEDICATIONS: [
            r"medications?:?",
            r"current\s+medications?:?",
            r"meds:?",
        ],
        ClinicalSection.ALLERGIES: [
            r"allergies:?",
            r"drug\s+allergies:?",
            r"nkda",  # No known drug allergies
        ],
        ClinicalSection.FAMILY_HISTORY: [
            r"family\s+history:?",
            r"fh:?",
            r"fam\s+hx:?",
        ],
        ClinicalSection.SOCIAL_HISTORY: [
            r"social\s+history:?",
            r"sh:?",
            r"soc\s+hx:?",
        ],
        ClinicalSection.REVIEW_OF_SYSTEMS: [
            r"review\s+of\s+systems:?",
            r"ros:?",
            r"systems\s+review:?",
        ],
        ClinicalSection.PHYSICAL_EXAM: [
            r"physical\s+exam(?:ination)?:?",
            r"pe:?",
            r"exam:?",
        ],
        ClinicalSection.ASSESSMENT: [
            r"assessment:?",
            r"impression:?",
            r"diagnosis:?",
            r"dx:?",
        ],
        ClinicalSection.PLAN: [
            r"plan:?",
            r"treatment\s+plan:?",
            r"recommendations?:?",
        ],
        ClinicalSection.LABS: [
            r"lab(?:oratory)?\s+(?:results|values):?",
            r"labs:?",
        ],
        ClinicalSection.IMAGING: [
            r"imaging:?",
            r"radiology:?",
            r"x-?ray:?",
            r"ct\s+scan:?",
            r"mri:?",
        ],
    }
    
    def __init__(self):
        # Compile patterns
        self._compiled_patterns: dict[ClinicalSection, re.Pattern] = {}
        for section, patterns in self.SECTION_PATTERNS.items():
            combined = "|".join(f"({p})" for p in patterns)
            self._compiled_patterns[section] = re.compile(
                rf"^\s*({combined})\s*$",
                re.IGNORECASE | re.MULTILINE
            )
    
    def segment(self, text: str) -> list[Section]:
        """Segment text into clinical sections."""
        sections = []
        
        # Find all section headers
        header_positions: list[tuple[int, int, ClinicalSection, str]] = []
        
        for section, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                header_positions.append((
                    match.start(),
                    match.end(),
                    section,
                    match.group(0).strip(),
                ))
        
        # Sort by position
        header_positions.sort(key=lambda x: x[0])
        
        # Extract sections
        for i, (start, header_end, section_type, header) in enumerate(header_positions):
            # Content ends at next header or end of text
            if i + 1 < len(header_positions):
                content_end = header_positions[i + 1][0]
            else:
                content_end = len(text)
            
            content = text[header_end:content_end].strip()
            
            sections.append(Section(
                section_type=section_type,
                header=header,
                content=content,
                start_pos=start,
                end_pos=content_end,
            ))
        
        # If no sections found, treat entire text as unknown section
        if not sections:
            sections.append(Section(
                section_type=ClinicalSection.UNKNOWN,
                header="",
                content=text.strip(),
                start_pos=0,
                end_pos=len(text),
            ))
        
        return sections


class ClinicalPreprocessor:
    """
    Production clinical text preprocessor.
    
    Features:
    - Section segmentation
    - Abbreviation expansion
    - Negation detection
    - Sentence boundary detection
    - Whitespace normalization
    """
    
    def __init__(self, config: ClinicalPreprocessorConfig | None = None):
        self.config = config or ClinicalPreprocessorConfig()
        self.section_segmenter = SectionSegmenter()
        self._processed_count = 0
    
    def preprocess(
        self,
        text: str,
        document_id: str | None = None,
    ) -> PreprocessedDocument:
        """
        Preprocess clinical text.
        
        Args:
            text: Raw clinical text
            document_id: Optional document identifier
        
        Returns:
            PreprocessedDocument with cleaned text and structure
        """
        import time
        import uuid
        
        start_time = time.perf_counter()
        doc_id = document_id or str(uuid.uuid4())
        
        # Normalize whitespace
        cleaned = text
        if self.config.normalize_whitespace:
            cleaned = self._normalize_whitespace(cleaned)
        
        # Expand abbreviations
        abbrev_count = 0
        if self.config.expand_abbreviations:
            cleaned, abbrev_count = ClinicalAbbreviations.expand(cleaned)
        
        # Remove PHI markers if present
        if self.config.remove_phi_markers:
            cleaned = re.sub(r'\[[\*]+\]', '', cleaned)
            cleaned = re.sub(r'\[REDACTED\]', '', cleaned)
        
        # Segment sections
        sections = []
        if self.config.segment_sections:
            sections = self.section_segmenter.segment(cleaned)
        
        # Split into sentences
        sentences = self._split_sentences(cleaned, sections)
        
        # Detect negation in sentences
        if self.config.detect_negation:
            for sentence in sentences:
                sentence.negated = NegationDetector.detect(sentence.text)
        
        # Optional lowercase
        if self.config.lowercase_output:
            cleaned = cleaned.lower()
        
        processing_time = (time.perf_counter() - start_time) * 1000
        self._processed_count += 1
        
        logger.info(
            "clinical_preprocessing_complete",
            document_id=doc_id,
            num_sections=len(sections),
            num_sentences=len(sentences),
            abbreviations_expanded=abbrev_count,
            processing_time_ms=processing_time,
        )
        
        return PreprocessedDocument(
            document_id=doc_id,
            original_text=text,
            cleaned_text=cleaned,
            sections=sections,
            sentences=sentences,
            abbreviations_expanded=abbrev_count,
            processing_time_ms=processing_time,
        )
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        return text.strip()
    
    def _split_sentences(
        self,
        text: str,
        sections: list[Section],
    ) -> list[Sentence]:
        """Split text into sentences."""
        sentences = []
        
        # Simple sentence boundary detection
        # Handles common clinical patterns like "Dr." "mg." etc.
        sentence_pattern = re.compile(
            r'(?<!\b(?:Dr|Mr|Mrs|Ms|Jr|Sr|vs|etc|mg|ml|kg|lb|cm|mm|hr|min|sec|pt|pts))'
            r'(?<!\b\d)'
            r'[.!?]+'
            r'(?=\s+[A-Z]|\s*$)',
            re.MULTILINE
        )
        
        # Split text
        last_end = 0
        for match in sentence_pattern.finditer(text):
            sent_end = match.end()
            sent_text = text[last_end:sent_end].strip()
            
            if sent_text and len(sent_text) <= self.config.sentence_max_length:
                # Find which section this sentence belongs to
                section_type = None
                for section in sections:
                    if section.start_pos <= last_end < section.end_pos:
                        section_type = section.section_type
                        break
                
                sentences.append(Sentence(
                    text=sent_text,
                    start=last_end,
                    end=sent_end,
                    section=section_type,
                ))
            
            last_end = sent_end
        
        # Handle remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append(Sentence(
                    text=remaining,
                    start=last_end,
                    end=len(text),
                    section=sections[-1].section_type if sections else None,
                ))
        
        return sentences
    
    @property
    def total_documents_processed(self) -> int:
        """Total documents processed."""
        return self._processed_count
