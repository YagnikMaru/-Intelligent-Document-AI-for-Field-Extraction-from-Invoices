import re
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentValidator:
    """
    Validates extracted document fields for consistency
    """
    
    def __init__(self):
        # Common tractor models and their expected HP ranges
        self.model_hp_map = {
            '475': (40, 50), '485': (45, 55), '575': (48, 58), '595': (52, 62),
            'DI': (45, 55), 'XP': (50, 60), 'XT': (55, 65), 'MX': (60, 70),
            'Swaraj': (30, 50), 'Mahindra': (40, 70), 'Eicher': (35, 55)
        }
        
        # Expected cost ranges for tractors (in INR)
        self.cost_ranges = {
            'small': (200000, 500000),    # Small tractors
            'medium': (500000, 1000000),  # Medium tractors
            'large': (1000000, 3000000)   # Large tractors
        }
    
    def validate_document(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all extracted fields
        
        Args:
            extracted_data: Dictionary with extracted fields
            
        Returns:
            Validated data with adjusted confidences
        """
        validated = extracted_data.copy()
        
        # Validate individual fields
        self._validate_dealer_name(validated)
        self._validate_model_name(validated)
        self._validate_horse_power(validated)
        self._validate_asset_cost(validated)
        self._validate_signature_stamp(validated)
        
        # Cross-field validation
        self._cross_validate_model_hp(validated)
        self._cross_validate_hp_cost(validated)
        
        # Recalculate overall confidence
        validated['overall_confidence'] = self._recalculate_confidence(validated)
        
        return validated
    
    def _validate_dealer_name(self, data: Dict[str, Any]):
        """Validate dealer name"""
        if 'dealer_name' not in data:
            return
        
        dealer_data = data['dealer_name']
        if dealer_data.get('value') is None:
            return
        
        dealer_name = str(dealer_data['value'])
        confidence = dealer_data.get('confidence', 0.5)
        
        # Check if dealer name looks reasonable
        issues = []
        
        # Too short
        if len(dealer_name) < 3:
            issues.append("too_short")
            confidence *= 0.7
        
        # Contains numbers (usually not in dealer names)
        if any(char.isdigit() for char in dealer_name):
            issues.append("contains_numbers")
            confidence *= 0.8
        
        # Check common dealer suffixes
        dealer_suffixes = ['tractors', 'motors', 'agency', 'enterprises', 'pvt', 'ltd']
        has_suffix = any(suffix in dealer_name.lower() for suffix in dealer_suffixes)
        
        if has_suffix:
            confidence *= 1.1  # Boost confidence if contains common suffix
            confidence = min(confidence, 0.95)
        
        if issues:
            logger.debug(f"Dealer validation issues: {issues}")
        
        dealer_data['confidence'] = confidence
        dealer_data['validation_issues'] = issues
    
    def _validate_model_name(self, data: Dict[str, Any]):
        """Validate model name"""
        if 'model_name' not in data:
            return
        
        model_data = data['model_name']
        if model_data.get('value') is None:
            return
        
        model_name = str(model_data['value'])
        confidence = model_data.get('confidence', 0.5)
        
        issues = []
        
        # Check if model contains common patterns
        model_pattern = re.search(r'(\d{3,4}\s*[A-Z]{1,3})', model_name)
        if model_pattern:
            confidence *= 1.2  # Boost for matching pattern
        else:
            issues.append("unusual_pattern")
            confidence *= 0.9
        
        # Check length
        if len(model_name) < 2:
            issues.append("too_short")
            confidence *= 0.7
        
        # Check for common tractor brands
        common_brands = ['MAHINDRA', 'SWARAJ', 'ESCORTS', 'EICHER', 'JOHN', 'DEERE', 'CASE']
        has_brand = any(brand in model_name.upper() for brand in common_brands)
        
        if has_brand:
            confidence *= 1.1
        
        model_data['confidence'] = min(confidence, 0.95)
        model_data['validation_issues'] = issues
    
    def _validate_horse_power(self, data: Dict[str, Any]):
        """Validate horse power"""
        if 'horse_power' not in data:
            return
        
        hp_data = data['horse_power']
        if hp_data.get('value') is None:
            return
        
        hp_value = hp_data['value']
        confidence = hp_data.get('confidence', 0.5)
        
        # Check if HP is within reasonable range for tractors
        if isinstance(hp_value, (int, float)):
            if 10 <= hp_value <= 200:  # Reasonable tractor HP range
                confidence *= 1.1
            else:
                confidence *= 0.7
                logger.warning(f"Horse power {hp_value} outside typical range")
        
        hp_data['confidence'] = min(confidence, 0.95)
    
    def _validate_asset_cost(self, data: Dict[str, Any]):
        """Validate asset cost"""
        if 'asset_cost' not in data:
            return
        
        cost_data = data['asset_cost']
        if cost_data.get('value') is None:
            return
        
        cost_value = cost_data['value']
        confidence = cost_data.get('confidence', 0.5)
        
        # Check if cost is within reasonable range
        if isinstance(cost_value, (int, float)):
            if cost_value > 0:
                if 100000 <= cost_value <= 5000000:  # 1L to 50L INR
                    confidence *= 1.1
                else:
                    confidence *= 0.8
                    logger.warning(f"Asset cost {cost_value} outside typical range")
            else:
                confidence *= 0.5
                logger.warning(f"Invalid asset cost: {cost_value}")
        
        cost_data['confidence'] = min(confidence, 0.95)
    
    def _validate_signature_stamp(self, data: Dict[str, Any]):
        """Validate signature and stamp data"""
        # Validate signature
        if 'signature' in data:
            sig_data = data['signature']
            if sig_data.get('present'):
                # Check if bounding box is provided
                if not sig_data.get('bbox'):
                    sig_data['confidence'] = sig_data.get('confidence', 0.7) * 0.9
        
        # Validate stamp
        if 'stamp' in data:
            stamp_data = data['stamp']
            if stamp_data.get('present'):
                # Check if bounding box is provided
                if not stamp_data.get('bbox'):
                    stamp_data['confidence'] = stamp_data.get('confidence', 0.7) * 0.9
    
    def _cross_validate_model_hp(self, data: Dict[str, Any]):
        """Cross-validate model name with horse power"""
        model_data = data.get('model_name', {})
        hp_data = data.get('horse_power', {})
        
        model_value = model_data.get('value')
        hp_value = hp_data.get('value')
        
        if model_value and hp_value and isinstance(hp_value, (int, float)):
            model_str = str(model_value).upper()
            
            # Check against known model-HP mappings
            for model_key, (min_hp, max_hp) in self.model_hp_map.items():
                if model_key in model_str:
                    if min_hp <= hp_value <= max_hp:
                        # Boost confidences for consistency
                        model_data['confidence'] = min(model_data.get('confidence', 0.5) * 1.15, 0.95)
                        hp_data['confidence'] = min(hp_data.get('confidence', 0.5) * 1.15, 0.95)
                    else:
                        # Reduce confidences for inconsistency
                        model_data['confidence'] = model_data.get('confidence', 0.5) * 0.85
                        hp_data['confidence'] = hp_data.get('confidence', 0.5) * 0.85
                    
                    logger.debug(f"Cross-validation: Model {model_key} expects HP {min_hp}-{max_hp}, got {hp_value}")
                    break
    
    def _cross_validate_hp_cost(self, data: Dict[str, Any]):
        """Cross-validate horse power with asset cost"""
        hp_data = data.get('horse_power', {})
        cost_data = data.get('asset_cost', {})
        
        hp_value = hp_data.get('value')
        cost_value = cost_data.get('value')
        
        if hp_value and cost_value and isinstance(hp_value, (int, float)) and isinstance(cost_value, (int, float)):
            # Very rough correlation: higher HP should generally cost more
            expected_min_cost = hp_value * 5000  # ~5000 INR per HP
            expected_max_cost = hp_value * 20000  # ~20000 INR per HP
            
            if expected_min_cost <= cost_value <= expected_max_cost:
                # Boost confidences
                hp_data['confidence'] = min(hp_data.get('confidence', 0.5) * 1.1, 0.95)
                cost_data['confidence'] = min(cost_data.get('confidence', 0.5) * 1.1, 0.95)
            else:
                # Still reasonable, just unusual
                hp_data['confidence'] = hp_data.get('confidence', 0.5) * 0.95
                cost_data['confidence'] = cost_data.get('confidence', 0.5) * 0.95
    
    def _recalculate_confidence(self, data: Dict[str, Any]) -> float:
        """
        Recalculate overall confidence based on validated field confidences
        
        Args:
            data: Validated data with field confidences
            
        Returns:
            Overall confidence score
        """
        # Field weights
        weights = {
            'dealer_name': 0.15,
            'model_name': 0.20,
            'horse_power': 0.20,
            'asset_cost': 0.25,
            'signature': 0.10,
            'stamp': 0.10
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for field, weight in weights.items():
            if field in data:
                field_data = data[field]
                
                if field in ['signature', 'stamp']:
                    # Binary fields
                    confidence = field_data.get('confidence', 0.5)
                    weighted_sum += confidence * weight
                    total_weight += weight
                else:
                    # Text/numeric fields
                    if field_data.get('value') is not None:
                        confidence = field_data.get('confidence', 0.5)
                        weighted_sum += confidence * weight
                        total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def check_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if all required fields are extracted
        
        Args:
            data: Extracted data
            
        Returns:
            Completeness report
        """
        required_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        
        report = {
            'missing_fields': [],
            'complete_fields': [],
            'completeness_score': 0.0
        }
        
        for field in required_fields:
            if field in data and data[field].get('value') is not None:
                report['complete_fields'].append(field)
            else:
                report['missing_fields'].append(field)
        
        report['completeness_score'] = len(report['complete_fields']) / len(required_fields)
        
        return report

# Singleton instance
_validator = None

def get_validator() -> DocumentValidator:
    """Get or create DocumentValidator instance"""
    global _validator
    if _validator is None:
        _validator = DocumentValidator()
    return _validator