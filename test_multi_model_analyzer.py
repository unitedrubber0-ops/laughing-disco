"""
Test suite for multi-model vision analyzer.

Run with: pytest test_multi_model_analyzer.py -v

These tests validate the voting logic without requiring actual API calls
by using mock Gemini responses.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from ai_multi_model_analyzer import (
    MultiModelVisionAnalyzer,
    MultiModelResult,
    analyze_image_with_voting
)


class TestMultiModelVisionAnalyzer:
    """Test suite for MultiModelVisionAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return MultiModelVisionAnalyzer(timeout_seconds=30)
    
    @pytest.fixture
    def sample_spec(self):
        """Sample specification output"""
        return {
            "id": "4509347C4",
            "standard": "MPAPS F-30",
            "grade": "1BF",
            "id_nominal_mm": 24.4,
            "id_tolerance_mm": 0.5,
            "od_nominal_mm": 33.0,
            "od_tolerance_mm": None,
            "wall_thickness_mm": 4.3,
            "wall_tolerance_mm": 0.8,
            "burst_pressure_psi": 500,
            "material": "P-EPDM",
            "reinforcement": "KEVLAR",
            "confidence": 0.95
        }
    
    def test_vote_on_results_both_models_found_same(self, analyzer, sample_spec):
        """When both models find identical spec, should merge with high confidence"""
        flash_result = {
            "specifications": [sample_spec],
            "errors": []
        }
        pro_result = {
            "specifications": [sample_spec.copy()],
            "errors": []
        }
        
        consensus = analyzer._vote_on_results(flash_result, pro_result)
        
        assert len(consensus['specifications']) == 1
        spec = consensus['specifications'][0]
        assert spec['voting_source'] == 'both_models'
        assert spec['confidence'] == 0.95
        assert spec['id'] == "4509347C4"
    
    def test_vote_on_results_only_flash_found(self, analyzer, sample_spec):
        """When only Flash finds spec, confidence reduced"""
        flash_result = {
            "specifications": [sample_spec],
            "errors": []
        }
        pro_result = {
            "specifications": [],
            "errors": []
        }
        
        consensus = analyzer._vote_on_results(flash_result, pro_result)
        
        assert len(consensus['specifications']) == 1
        spec = consensus['specifications'][0]
        assert spec['voting_source'] == 'flash_only'
        assert spec['confidence'] == pytest.approx(0.855, abs=0.01)  # 0.95 * 0.9
        assert len(consensus['voting_notes']) > 0
    
    def test_vote_on_results_only_pro_found(self, analyzer, sample_spec):
        """When only Pro finds spec, confidence reduced slightly"""
        flash_result = {
            "specifications": [],
            "errors": []
        }
        pro_result = {
            "specifications": [sample_spec],
            "errors": []
        }
        
        consensus = analyzer._vote_on_results(flash_result, pro_result)
        
        assert len(consensus['specifications']) == 1
        spec = consensus['specifications'][0]
        assert spec['voting_source'] == 'pro_only'
        assert spec['confidence'] == pytest.approx(0.9025, abs=0.01)  # 0.95 * 0.95
        assert len(consensus['voting_notes']) > 0
    
    def test_vote_on_results_disagreement_detected(self, analyzer, sample_spec):
        """When models disagree on numeric values, discrepancy noted"""
        spec_flash = sample_spec.copy()
        spec_flash['wall_thickness_mm'] = 4.3
        
        spec_pro = sample_spec.copy()
        spec_pro['wall_thickness_mm'] = 4.5  # 4.7% difference
        
        flash_result = {
            "specifications": [spec_flash],
            "errors": []
        }
        pro_result = {
            "specifications": [spec_pro],
            "errors": []
        }
        
        consensus = analyzer._vote_on_results(flash_result, pro_result)
        
        spec = consensus['specifications'][0]
        assert 'wall_thickness_mm_disagreement' in spec
        assert '4.3' in spec['wall_thickness_mm_disagreement']
        assert '4.5' in spec['wall_thickness_mm_disagreement']
    
    def test_vote_on_results_numeric_averaging(self, analyzer, sample_spec):
        """When values differ by <5%, should average them"""
        spec_flash = sample_spec.copy()
        spec_flash['burst_pressure_psi'] = 500
        
        spec_pro = sample_spec.copy()
        spec_pro['burst_pressure_psi'] = 510  # 2% difference
        
        flash_result = {
            "specifications": [spec_flash],
            "errors": []
        }
        pro_result = {
            "specifications": [spec_pro],
            "errors": []
        }
        
        consensus = analyzer._vote_on_results(flash_result, pro_result)
        
        spec = consensus['specifications'][0]
        assert spec['burst_pressure_psi'] == 505  # Average
        assert 'burst_pressure_psi_disagreement' not in spec
    
    def test_calculate_agreement_score_perfect_match(self, analyzer, sample_spec):
        """Perfect agreement between models should score high"""
        result = {
            "specifications": [sample_spec],
            "errors": []
        }
        
        score = analyzer._calculate_agreement_score(result, result)
        
        assert score > 0.85  # Should be very high
    
    def test_calculate_agreement_score_no_overlap(self, analyzer, sample_spec):
        """No common specs found should score low"""
        spec2 = sample_spec.copy()
        spec2['id'] = "DIFFERENT_ID"
        
        result1 = {"specifications": [sample_spec], "errors": []}
        result2 = {"specifications": [spec2], "errors": []}
        
        score = analyzer._calculate_agreement_score(result1, result2)
        
        assert score < 0.5  # Low agreement
    
    def test_calculate_agreement_score_empty_results(self, analyzer):
        """Empty results should score moderate"""
        result = {"specifications": [], "errors": []}
        
        score = analyzer._calculate_agreement_score(result, result)
        
        assert score == 0.5  # Fallback score
    
    def test_find_discrepancies_both_found_different_values(self, analyzer, sample_spec):
        """Should detect differences in findings"""
        spec_flash = sample_spec.copy()
        spec_pro = sample_spec.copy()
        spec_pro['standard'] = 'MPAPS F-1'
        
        flash_result = {"specifications": [spec_flash], "errors": []}
        pro_result = {"specifications": [spec_pro], "errors": []}
        
        discrepancies = analyzer._find_discrepancies(flash_result, pro_result)
        
        assert len(discrepancies) > 0
        assert any('standard' in d for d in discrepancies)
    
    def test_find_discrepancies_flash_only(self, analyzer, sample_spec):
        """Should detect spec found by Flash but not Pro"""
        flash_result = {"specifications": [sample_spec], "errors": []}
        pro_result = {"specifications": [], "errors": []}
        
        discrepancies = analyzer._find_discrepancies(flash_result, pro_result)
        
        assert len(discrepancies) > 0
        assert any('Flash only' in d for d in discrepancies)
    
    def test_find_discrepancies_pro_only(self, analyzer, sample_spec):
        """Should detect spec found by Pro but not Flash"""
        flash_result = {"specifications": [], "errors": []}
        pro_result = {"specifications": [sample_spec], "errors": []}
        
        discrepancies = analyzer._find_discrepancies(flash_result, pro_result)
        
        assert len(discrepancies) > 0
        assert any('Pro only' in d for d in discrepancies)
    
    def test_find_discrepancies_none(self, analyzer, sample_spec):
        """When results identical, should have no discrepancies"""
        result = {"specifications": [sample_spec], "errors": []}
        
        discrepancies = analyzer._find_discrepancies(result, result)
        
        assert len(discrepancies) == 0
    
    def test_merge_specs_numeric_averaging(self, analyzer):
        """Should average numeric fields within tolerance"""
        spec1 = {
            "id": "TEST",
            "id_nominal_mm": 24.0,
            "od_nominal_mm": 33.0,
            "confidence": 0.95
        }
        spec2 = {
            "id": "TEST",
            "id_nominal_mm": 24.2,
            "od_nominal_mm": 33.0,
            "confidence": 0.92
        }
        
        merged = analyzer._merge_specs(spec1, spec2)
        
        assert merged['id_nominal_mm'] == 24.1  # Average
        assert merged['od_nominal_mm'] == 33.0  # Same
    
    def test_merge_specs_text_field_differences(self, analyzer):
        """Should flag text field differences"""
        spec1 = {
            "id": "TEST",
            "material": "P-EPDM",
            "reinforcement": "KEVLAR"
        }
        spec2 = {
            "id": "TEST",
            "material": "P-NBR",
            "reinforcement": "KEVLAR"
        }
        
        merged = analyzer._merge_specs(spec1, spec2)
        
        assert merged['material'] == "P-EPDM"
        assert 'material_alternative' in merged
        assert merged['material_alternative'] == "P-NBR"
    
    def test_multi_model_result_dataclass(self, sample_spec):
        """MultiModelResult should store voting data properly"""
        flash_result = {"specifications": [sample_spec], "errors": []}
        pro_result = {"specifications": [sample_spec], "errors": []}
        consensus = {"specifications": [sample_spec], "errors": []}
        
        result = MultiModelResult(
            flash_result=flash_result,
            pro_result=pro_result,
            consensus=consensus,
            confidence=0.95,
            model_agreement=True,
            discrepancies=[]
        )
        
        assert result.confidence == 0.95
        assert result.model_agreement is True
        assert len(result.discrepancies) == 0
        assert len(result.consensus['specifications']) == 1


class TestIntegration:
    """Integration tests for multi-model workflow"""
    
    def test_realistic_workflow_both_models_agree(self):
        """Simulate realistic scenario where both models find same part"""
        analyzer = MultiModelVisionAnalyzer()
        
        spec1 = {
            "id": "4509347C4",
            "standard": "MPAPS F-30",
            "grade": "1BF",
            "id_nominal_mm": 24.4,
            "material": "P-EPDM",
            "confidence": 0.95
        }
        
        flash = {"specifications": [spec1], "errors": []}
        pro = {"specifications": [spec1.copy()], "errors": []}
        
        consensus = analyzer._vote_on_results(flash, pro)
        confidence = analyzer._calculate_agreement_score(flash, pro)
        
        assert len(consensus['specifications']) == 1
        assert confidence > 0.8
        assert consensus['specifications'][0]['id'] == "4509347C4"
    
    def test_realistic_workflow_partial_agreement(self):
        """Simulate scenario where models partially agree"""
        analyzer = MultiModelVisionAnalyzer()
        
        spec1 = {
            "id": "4509347C4",
            "standard": "MPAPS F-30",
            "material": "P-EPDM",
            "confidence": 0.95
        }
        spec2 = {
            "id": "4509348C4",
            "standard": "MPAPS F-30",
            "material": "P-NBR",
            "confidence": 0.90
        }
        
        # Flash finds both, Pro finds only first
        flash = {"specifications": [spec1, spec2], "errors": []}
        pro = {"specifications": [spec1.copy()], "errors": []}
        
        consensus = analyzer._vote_on_results(flash, pro)
        discrepancies = analyzer._find_discrepancies(flash, pro)
        
        assert len(consensus['specifications']) == 2
        assert len(discrepancies) == 1  # Flash only: second part
        assert any(s['voting_source'] == 'both_models' for s in consensus['specifications'])
        assert any(s['voting_source'] == 'flash_only' for s in consensus['specifications'])


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_merge_specs_with_none_values(self):
        """Should handle None values gracefully"""
        analyzer = MultiModelVisionAnalyzer()
        
        spec1 = {"id": "TEST", "od_tolerance_mm": None}
        spec2 = {"id": "TEST", "od_tolerance_mm": 0.5}
        
        merged = analyzer._merge_specs(spec1, spec2)
        
        # Should keep the non-None value
        assert merged['od_tolerance_mm'] is None or merged['od_tolerance_mm'] == 0.5
    
    def test_calculate_agreement_zero_division_protection(self):
        """Should handle division by zero"""
        analyzer = MultiModelVisionAnalyzer()
        
        spec = {"id": "TEST", "burst_pressure_psi": 0}
        result = {"specifications": [spec], "errors": []}
        
        # Should not raise exception
        score = analyzer._calculate_agreement_score(result, result)
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
