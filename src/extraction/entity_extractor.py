"""
Medical Entity Extraction

Extract medical entities from clinical text:
- Medications (drug name, dosage, frequency)
- Conditions (diseases, symptoms)
- Procedures (surgeries, tests)
- Anatomy (body parts, organs)
- Measurements (vitals, lab values)
- Temporal expressions
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class EntityType(str, Enum):
    """Types of medical entities."""
    
    MEDICATION = "medication"
    CONDITION = "condition"
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    TEMPORAL = "temporal"
    MEASUREMENT = "measurement"
    OBSERVATION = "observation"
    PERSON = "person"
    ORGANIZATION = "organization"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    
    TREATS = "treats"
    CAUSES = "causes"
    ADMINISTERED_TO = "administered_to"
    LOCATED_AT = "located_at"
    OCCURRED_AT = "occurred_at"
    MEASURED_AT = "measured_at"
    DIAGNOSED_WITH = "diagnosed_with"


@dataclass
class Entity:
    """An extracted medical entity."""
    
    entity_id: str
    entity_type: EntityType
    text: str
    start: int
    end: int
    confidence: float
    normalized_text: str | None = None
    umls_cui: str | None = None  # UMLS Concept Unique Identifier
    attributes: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 4),
            "normalized_text": self.normalized_text,
            "umls_cui": self.umls_cui,
            "attributes": self.attributes,
        }


@dataclass
class Relation:
    """A relationship between two entities."""
    
    relation_id: str
    relation_type: RelationType
    source_entity_id: str
    target_entity_id: str
    confidence: float
    evidence_text: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type.value,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "confidence": round(self.confidence, 4),
            "evidence_text": self.evidence_text,
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    
    document_id: str
    text_length: int
    entities: list[Entity]
    relations: list[Relation]
    entity_counts: dict[str, int]
    processing_time_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text_length": self.text_length,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entity_counts": self.entity_counts,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
        }


class EntityExtractorConfig(BaseModel):
    """Configuration for entity extraction."""
    
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    extract_relations: bool = True
    normalize_entities: bool = True
    max_entity_length: int = Field(default=100, ge=1)
    relation_window_chars: int = Field(default=200, ge=10)


class MedicalPatterns:
    """Regex patterns for medical entity extraction."""
    
    # Medication patterns
    MEDICATION_PATTERNS = [
        # Drug name + dosage + unit
        r'\b([A-Z][a-z]+(?:mab|nib|vir|ine|ide|ole|ate|il|ol|ium)?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|IU)\b',
        # Common drug names
        r'\b(aspirin|ibuprofen|acetaminophen|metformin|lisinopril|atorvastatin|omeprazole|amlodipine|metoprolol|losartan|gabapentin|hydrochlorothiazide|sertraline|simvastatin|montelukast|escitalopram|pantoprazole|rosuvastatin|bupropion|tramadol|furosemide|tamsulosin|meloxicam|carvedilol|trazodone|prednisone|duloxetine|clopidogrel|potassium|albuterol|fluticasone|insulin|levothyroxine|warfarin|amoxicillin|azithromycin|ciprofloxacin)\b',
    ]
    
    # Condition patterns
    CONDITION_PATTERNS = [
        r'\b(diabetes(?:\s+mellitus)?(?:\s+type\s*[12])?)\b',
        r'\b(hypertension|HTN|high\s+blood\s+pressure)\b',
        r'\b(coronary\s+artery\s+disease|CAD)\b',
        r'\b(chronic\s+kidney\s+disease|CKD(?:\s+stage\s*[1-5])?)\b',
        r'\b(congestive\s+heart\s+failure|CHF|heart\s+failure)\b',
        r'\b(chronic\s+obstructive\s+pulmonary\s+disease|COPD)\b',
        r'\b(atrial\s+fibrillation|AFib|a-?fib)\b',
        r'\b(depression|major\s+depressive\s+disorder|MDD)\b',
        r'\b(anxiety|generalized\s+anxiety\s+disorder|GAD)\b',
        r'\b(cancer|carcinoma|malignancy|neoplasm)\b',
        r'\b(stroke|CVA|cerebrovascular\s+accident)\b',
        r'\b(pneumonia|bronchitis|asthma)\b',
        r'\b(arthritis|osteoarthritis|rheumatoid\s+arthritis)\b',
    ]
    
    # Procedure patterns
    PROCEDURE_PATTERNS = [
        r'\b(surgery|surgical\s+procedure|operation)\b',
        r'\b(colonoscopy|endoscopy|biopsy)\b',
        r'\b(MRI|CT\s+scan|X-?ray|ultrasound|echocardiogram)\b',
        r'\b(blood\s+test|lab\s+work|CBC|BMP|CMP)\b',
        r'\b(EKG|ECG|electrocardiogram)\b',
        r'\b(catheterization|angiogram|angioplasty)\b',
        r'\b(transplant|implant|replacement)\b',
    ]
    
    # Anatomy patterns
    ANATOMY_PATTERNS = [
        r'\b(heart|cardiac|cardiovascular)\b',
        r'\b(lung|pulmonary|respiratory)\b',
        r'\b(liver|hepatic)\b',
        r'\b(kidney|renal)\b',
        r'\b(brain|cerebral|neurological)\b',
        r'\b(stomach|gastric|GI|gastrointestinal)\b',
        r'\b(chest|thoracic)\b',
        r'\b(abdomen|abdominal)\b',
        r'\b(spine|spinal|vertebral)\b',
        r'\b(joint|knee|hip|shoulder|ankle|wrist|elbow)\b',
    ]
    
    # Measurement patterns
    MEASUREMENT_PATTERNS = [
        # Blood pressure
        r'\b(?:BP|blood\s+pressure)[:\s]+(\d{2,3})/(\d{2,3})\s*(?:mmHg)?\b',
        # Heart rate
        r'\b(?:HR|heart\s+rate|pulse)[:\s]+(\d{2,3})\s*(?:bpm|/min)?\b',
        # Temperature
        r'\b(?:temp|temperature)[:\s]+(\d{2,3}(?:\.\d)?)\s*(?:°?[FC])?\b',
        # Weight
        r'\b(?:weight|wt)[:\s]+(\d{2,3}(?:\.\d)?)\s*(?:kg|lbs?|pounds?)?\b',
        # Height
        r'\b(?:height|ht)[:\s]+(\d{1,3}(?:\.\d)?)\s*(?:cm|in|inches|feet|ft)?\b',
        # HbA1c
        r'\b(?:HbA1c|A1c|hemoglobin\s+A1c)[:\s]+(\d{1,2}(?:\.\d)?)\s*%?\b',
        # Blood glucose
        r'\b(?:glucose|blood\s+sugar|BG)[:\s]+(\d{2,3})\s*(?:mg/dL)?\b',
        # Creatinine
        r'\b(?:creatinine|Cr)[:\s]+(\d{1,2}(?:\.\d{1,2})?)\s*(?:mg/dL)?\b',
        # eGFR
        r'\b(?:eGFR|GFR)[:\s]+(\d{1,3})\s*(?:mL/min)?\b',
    ]


class MedicalEntityExtractor:
    """
    Production medical entity extractor.
    
    Features:
    - Pattern-based entity recognition
    - Confidence scoring
    - UMLS concept normalization (placeholder)
    - Relation extraction
    - Entity deduplication
    """
    
    def __init__(self, config: EntityExtractorConfig | None = None):
        self.config = config or EntityExtractorConfig()
        self._compile_patterns()
        self._processed_count = 0
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._patterns = {
            EntityType.MEDICATION: [
                re.compile(p, re.IGNORECASE)
                for p in MedicalPatterns.MEDICATION_PATTERNS
            ],
            EntityType.CONDITION: [
                re.compile(p, re.IGNORECASE)
                for p in MedicalPatterns.CONDITION_PATTERNS
            ],
            EntityType.PROCEDURE: [
                re.compile(p, re.IGNORECASE)
                for p in MedicalPatterns.PROCEDURE_PATTERNS
            ],
            EntityType.ANATOMY: [
                re.compile(p, re.IGNORECASE)
                for p in MedicalPatterns.ANATOMY_PATTERNS
            ],
            EntityType.MEASUREMENT: [
                re.compile(p, re.IGNORECASE)
                for p in MedicalPatterns.MEASUREMENT_PATTERNS
            ],
        }
    
    def extract(
        self,
        text: str,
        document_id: str | None = None,
    ) -> ExtractionResult:
        """
        Extract medical entities from text.
        
        Args:
            text: Clinical text to process
            document_id: Optional document identifier
        
        Returns:
            ExtractionResult with entities and relations
        """
        import time
        
        start_time = time.perf_counter()
        doc_id = document_id or str(uuid.uuid4())
        
        # Extract entities by type
        all_entities: list[Entity] = []
        
        for entity_type, patterns in self._patterns.items():
            entities = self._extract_entity_type(text, entity_type, patterns)
            all_entities.extend(entities)
        
        # Remove overlapping entities
        all_entities = self._remove_overlaps(all_entities)
        
        # Filter by confidence
        all_entities = [
            e for e in all_entities
            if e.confidence >= self.config.min_confidence
        ]
        
        # Extract relations
        relations = []
        if self.config.extract_relations:
            relations = self._extract_relations(text, all_entities)
        
        # Count entities by type
        entity_counts = {}
        for entity in all_entities:
            key = entity.entity_type.value
            entity_counts[key] = entity_counts.get(key, 0) + 1
        
        processing_time = (time.perf_counter() - start_time) * 1000
        self._processed_count += 1
        
        logger.info(
            "entity_extraction_complete",
            document_id=doc_id,
            entity_count=len(all_entities),
            relation_count=len(relations),
            processing_time_ms=processing_time,
        )
        
        return ExtractionResult(
            document_id=doc_id,
            text_length=len(text),
            entities=all_entities,
            relations=relations,
            entity_counts=entity_counts,
            processing_time_ms=processing_time,
        )
    
    def _extract_entity_type(
        self,
        text: str,
        entity_type: EntityType,
        patterns: list[re.Pattern],
    ) -> list[Entity]:
        """Extract entities of a specific type."""
        entities = []
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                # Calculate confidence based on match quality
                confidence = self._calculate_confidence(match, entity_type)
                
                entity_text = match.group(0)
                if len(entity_text) > self.config.max_entity_length:
                    continue
                
                entity = Entity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=entity_type,
                    text=entity_text,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    normalized_text=self._normalize(entity_text) if self.config.normalize_entities else None,
                    attributes=self._extract_attributes(match, entity_type),
                )
                
                entities.append(entity)
        
        return entities
    
    def _calculate_confidence(self, match: re.Match, entity_type: EntityType) -> float:
        """Calculate confidence score for a match."""
        # Base confidence by entity type
        base_confidence = {
            EntityType.MEDICATION: 0.85,
            EntityType.CONDITION: 0.80,
            EntityType.PROCEDURE: 0.85,
            EntityType.ANATOMY: 0.90,
            EntityType.MEASUREMENT: 0.95,
        }.get(entity_type, 0.75)
        
        # Adjust based on match characteristics
        text = match.group(0)
        
        # Longer matches are generally more reliable
        if len(text) > 10:
            base_confidence += 0.05
        
        # Exact case matches are more reliable
        if text[0].isupper():
            base_confidence += 0.02
        
        return min(base_confidence, 1.0)
    
    def _normalize(self, text: str) -> str:
        """Normalize entity text."""
        # Basic normalization
        normalized = text.lower().strip()
        
        # Common abbreviation expansions
        expansions = {
            "dm": "diabetes mellitus",
            "htn": "hypertension",
            "cad": "coronary artery disease",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "ckd": "chronic kidney disease",
            "cva": "cerebrovascular accident",
            "afib": "atrial fibrillation",
            "a-fib": "atrial fibrillation",
        }
        
        return expansions.get(normalized, normalized)
    
    def _extract_attributes(self, match: re.Match, entity_type: EntityType) -> dict[str, Any]:
        """Extract additional attributes from match groups."""
        attributes = {}
        
        if entity_type == EntityType.MEDICATION:
            groups = match.groups()
            if len(groups) >= 3:
                attributes["drug_name"] = groups[0]
                attributes["dosage"] = groups[1]
                attributes["unit"] = groups[2]
        
        elif entity_type == EntityType.MEASUREMENT:
            groups = match.groups()
            if groups:
                attributes["values"] = [g for g in groups if g]
        
        return attributes
    
    def _remove_overlaps(self, entities: list[Entity]) -> list[Entity]:
        """Remove overlapping entities, keeping highest confidence."""
        if not entities:
            return []
        
        # Sort by confidence (descending)
        entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        
        kept = []
        used_spans = set()
        
        for entity in entities:
            span = (entity.start, entity.end)
            
            # Check for overlap with any kept entity
            overlaps = False
            for used_start, used_end in used_spans:
                if entity.start < used_end and entity.end > used_start:
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(entity)
                used_spans.add(span)
        
        return sorted(kept, key=lambda e: e.start)
    
    def _extract_relations(
        self,
        text: str,
        entities: list[Entity],
    ) -> list[Relation]:
        """Extract relationships between entities."""
        relations = []
        
        # Simple proximity-based relation extraction
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                # Check if entities are within window
                distance = e2.start - e1.end
                if 0 < distance <= self.config.relation_window_chars:
                    relation = self._infer_relation(e1, e2, text[e1.end:e2.start])
                    if relation:
                        relations.append(relation)
        
        return relations
    
    def _infer_relation(
        self,
        e1: Entity,
        e2: Entity,
        context: str,
    ) -> Relation | None:
        """Infer relationship type from entity types and context."""
        # Medication -> Condition: TREATS
        if e1.entity_type == EntityType.MEDICATION and e2.entity_type == EntityType.CONDITION:
            if any(w in context.lower() for w in ["for", "treats", "managing"]):
                return Relation(
                    relation_id=str(uuid.uuid4()),
                    relation_type=RelationType.TREATS,
                    source_entity_id=e1.entity_id,
                    target_entity_id=e2.entity_id,
                    confidence=0.75,
                    evidence_text=context.strip(),
                )
        
        # Condition -> Anatomy: LOCATED_AT
        if e1.entity_type == EntityType.CONDITION and e2.entity_type == EntityType.ANATOMY:
            return Relation(
                relation_id=str(uuid.uuid4()),
                relation_type=RelationType.LOCATED_AT,
                source_entity_id=e1.entity_id,
                target_entity_id=e2.entity_id,
                confidence=0.70,
                evidence_text=context.strip(),
            )
        
        # Measurement -> Temporal: MEASURED_AT
        if e1.entity_type == EntityType.MEASUREMENT and e2.entity_type == EntityType.TEMPORAL:
            return Relation(
                relation_id=str(uuid.uuid4()),
                relation_type=RelationType.MEASURED_AT,
                source_entity_id=e1.entity_id,
                target_entity_id=e2.entity_id,
                confidence=0.80,
                evidence_text=context.strip(),
            )
        
        return None
    
    @property
    def total_documents_processed(self) -> int:
        """Total documents processed."""
        return self._processed_count
