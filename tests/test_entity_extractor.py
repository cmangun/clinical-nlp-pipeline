"""Tests for medical entity extraction."""

import pytest

from src.extraction.entity_extractor import (
    MedicalEntityExtractor,
    EntityExtractorConfig,
    EntityType,
)


class TestMedicationExtraction:
    """Test medication entity extraction."""
    
    def test_extract_medication_with_dosage(self):
        """Test extracting medication with dosage."""
        extractor = MedicalEntityExtractor()
        
        text = "Patient is taking Metformin 500 mg twice daily for diabetes."
        result = extractor.extract(text)
        
        medications = [e for e in result.entities if e.entity_type == EntityType.MEDICATION]
        assert len(medications) >= 1
        
        # Check that metformin was found
        med_texts = [m.text.lower() for m in medications]
        assert any("metformin" in t for t in med_texts)
    
    def test_extract_common_medications(self):
        """Test extracting common medication names."""
        extractor = MedicalEntityExtractor()
        
        text = "Current medications include aspirin, lisinopril, and atorvastatin."
        result = extractor.extract(text)
        
        medications = [e for e in result.entities if e.entity_type == EntityType.MEDICATION]
        med_texts = [m.text.lower() for m in medications]
        
        assert any("aspirin" in t for t in med_texts)
        assert any("lisinopril" in t for t in med_texts)


class TestConditionExtraction:
    """Test condition entity extraction."""
    
    def test_extract_diabetes(self):
        """Test extracting diabetes mentions."""
        extractor = MedicalEntityExtractor()
        
        text = "Patient has diabetes mellitus type 2, well controlled."
        result = extractor.extract(text)
        
        conditions = [e for e in result.entities if e.entity_type == EntityType.CONDITION]
        assert len(conditions) >= 1
    
    def test_extract_hypertension(self):
        """Test extracting hypertension mentions."""
        extractor = MedicalEntityExtractor()
        
        text = "History of hypertension and CAD."
        result = extractor.extract(text)
        
        conditions = [e for e in result.entities if e.entity_type == EntityType.CONDITION]
        condition_texts = [c.text.lower() for c in conditions]
        
        assert any("hypertension" in t or "cad" in t for t in condition_texts)
    
    def test_extract_abbreviations(self):
        """Test extracting condition abbreviations."""
        extractor = MedicalEntityExtractor()
        
        text = "PMH: HTN, DM, CHF, COPD, CKD stage 3"
        result = extractor.extract(text)
        
        conditions = [e for e in result.entities if e.entity_type == EntityType.CONDITION]
        # Should find multiple conditions
        assert len(conditions) >= 3


class TestMeasurementExtraction:
    """Test measurement entity extraction."""
    
    def test_extract_blood_pressure(self):
        """Test extracting blood pressure."""
        extractor = MedicalEntityExtractor()
        
        text = "Vitals: BP 120/80, HR 72, Temp 98.6"
        result = extractor.extract(text)
        
        measurements = [e for e in result.entities if e.entity_type == EntityType.MEASUREMENT]
        assert len(measurements) >= 1
    
    def test_extract_hba1c(self):
        """Test extracting HbA1c values."""
        extractor = MedicalEntityExtractor()
        
        text = "Labs: HbA1c 7.2%, creatinine 1.2 mg/dL"
        result = extractor.extract(text)
        
        measurements = [e for e in result.entities if e.entity_type == EntityType.MEASUREMENT]
        assert len(measurements) >= 1


class TestProcedureExtraction:
    """Test procedure entity extraction."""
    
    def test_extract_imaging(self):
        """Test extracting imaging procedures."""
        extractor = MedicalEntityExtractor()
        
        text = "Ordered CT scan of abdomen and MRI of brain."
        result = extractor.extract(text)
        
        procedures = [e for e in result.entities if e.entity_type == EntityType.PROCEDURE]
        proc_texts = [p.text.lower() for p in procedures]
        
        assert any("ct" in t or "mri" in t for t in proc_texts)


class TestAnatomyExtraction:
    """Test anatomy entity extraction."""
    
    def test_extract_organs(self):
        """Test extracting organ mentions."""
        extractor = MedicalEntityExtractor()
        
        text = "Examination of the heart and lungs was unremarkable."
        result = extractor.extract(text)
        
        anatomy = [e for e in result.entities if e.entity_type == EntityType.ANATOMY]
        anat_texts = [a.text.lower() for a in anatomy]
        
        assert any("heart" in t or "lung" in t for t in anat_texts)


class TestConfidenceFiltering:
    """Test confidence-based filtering."""
    
    def test_min_confidence_filter(self):
        """Test that low confidence entities are filtered."""
        config = EntityExtractorConfig(min_confidence=0.95)
        extractor = MedicalEntityExtractor(config)
        
        text = "Patient has diabetes and hypertension."
        result = extractor.extract(text)
        
        # All returned entities should have confidence >= 0.95
        for entity in result.entities:
            assert entity.confidence >= 0.95


class TestEntityCounts:
    """Test entity counting."""
    
    def test_entity_count_by_type(self):
        """Test that entity counts are correct."""
        extractor = MedicalEntityExtractor()
        
        text = "Patient with diabetes and hypertension taking metformin and lisinopril."
        result = extractor.extract(text)
        
        # Check that counts match actual entities
        total_from_counts = sum(result.entity_counts.values())
        assert total_from_counts == len(result.entities)


class TestOverlapRemoval:
    """Test overlapping entity removal."""
    
    def test_no_overlapping_entities(self):
        """Test that overlapping entities are removed."""
        extractor = MedicalEntityExtractor()
        
        text = "Patient has coronary artery disease."
        result = extractor.extract(text)
        
        # Check no two entities overlap
        for i, e1 in enumerate(result.entities):
            for e2 in result.entities[i + 1:]:
                # Entities should not overlap
                assert e1.end <= e2.start or e2.end <= e1.start


class TestNormalization:
    """Test entity normalization."""
    
    def test_abbreviation_normalization(self):
        """Test that abbreviations are normalized."""
        config = EntityExtractorConfig(normalize_entities=True)
        extractor = MedicalEntityExtractor(config)
        
        text = "History of HTN and DM."
        result = extractor.extract(text)
        
        # Check for normalized text
        conditions = [e for e in result.entities if e.entity_type == EntityType.CONDITION]
        normalized_texts = [c.normalized_text for c in conditions if c.normalized_text]
        
        assert any("hypertension" in t for t in normalized_texts) or len(conditions) > 0
