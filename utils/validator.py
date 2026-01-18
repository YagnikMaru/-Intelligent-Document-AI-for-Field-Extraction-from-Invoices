import re
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Validation rule with condition and confidence modifier"""
    name: str
    condition: Callable[[Any], bool]
    confidence_multiplier: float
    is_boost: bool = True
    severity: str = 'medium'  # low, medium, high


@dataclass
class FieldConfig:
    """Configuration for field validation"""
    required: bool = True
    weight: float = 0.15
    min_confidence: float = 0.30
    max_confidence: float = 0.98
    rules: List[ValidationRule] = field(default_factory=list)


class DocumentValidator:
    """
    Advanced document validator with configurable rules and cross-validation.
    Designed for invoice/quotation document verification with ≥95% accuracy target.
    """
    
    _instance = None
    _initialized = False
    
    # Pre-compiled patterns (shared across instances)
    _MODEL_PATTERN = re.compile(r'\b([2-9]\d{2,3}\s*(?:[A-Z]{1,3})?)\b', re.I)
    _DIGIT_PATTERN = re.compile(r'\d')
    _SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z0-9\s&.,()-]')
    
    # Frozen sets for fast lookups
    _DEALER_SUFFIXES = frozenset([
        'tractors', 'motors', 'agency', 'enterprises', 'pvt', 'ltd', 
        'limited', 'company', 'corp', 'inc', 'dealer', 'distributor',
        'trading', 'sales', 'services', 'automotive', 'machinery'
    ])
    
    _COMMON_BRANDS = frozenset([
        'MAHINDRA', 'SWARAJ', 'ESCORTS', 'EICHER', 'JOHN DEERE', 
        'CASE', 'NEW HOLLAND', 'SONALIKA', 'KUBOTA', 'MASSEY FERGUSON',
        'ACE', 'PREET', 'CAPTAIN', 'POWERTRAC', 'VST', 'FARMTRAC',
        'TAFE', 'INDO FARM', 'STANDARD'
    ])
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize validator with comprehensive rules"""
        if self._initialized:
            return
        
        # Model-HP validation map (expanded)
        self.model_hp_map = {
            # Mahindra models
            '265': (28, 35), '275': (32, 38), '475': (40, 48), 
            '485': (43, 51), '575': (47, 55), '585': (48, 56),
            '595': (50, 58), '605': (53, 63), '275 DI': (32, 38),
            '475 DI': (42, 50), '575 DI': (48, 56), '595 DI': (52, 62),
            # Swaraj models
            '735': (38, 45), '744': (45, 52), '855': (52, 60), '963': (60, 68),
            # New Holland
            '3600': (48, 56), '4710': (52, 60), '5620': (60, 68),
            # Generic suffixes
            'DI': (40, 60), 'XP': (48, 62), 'XT': (53, 68), 
            'MX': (58, 72), 'GT': (65, 78), 'PRO': (55, 70)
        }
        
        # Validation ranges
        self.HP_MIN, self.HP_MAX = 10, 150
        self.COST_MIN, self.COST_MAX = 50000, 10000000
        self.COST_PER_HP_MIN, self.COST_PER_HP_MAX = 8000, 25000
        
        # Field configurations
        self.field_configs = {
            'dealer_name': FieldConfig(
                weight=0.20, 
                required=True,
                rules=self._get_dealer_rules()
            ),
            'model_name': FieldConfig(
                weight=0.25, 
                required=True,
                rules=self._get_model_rules()
            ),
            'horse_power': FieldConfig(
                weight=0.20, 
                required=True,
                rules=self._get_hp_rules()
            ),
            'asset_cost': FieldConfig(
                weight=0.25, 
                required=True,
                rules=self._get_cost_rules()
            ),
            'signature': FieldConfig(
                weight=0.05, 
                required=False,
                rules=self._get_signature_rules()
            ),
            'stamp': FieldConfig(
                weight=0.05, 
                required=False,
                rules=self._get_stamp_rules()
            )
        }
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'field_failures': {}
        }
        
        self._initialized = True
        logger.info("Document validator initialized")
    
    def _get_dealer_rules(self) -> List[ValidationRule]:
        """Validation rules for dealer name"""
        return [
            ValidationRule(
                'min_length',
                lambda x: len(str(x).strip()) >= 3,
                0.6,
                is_boost=False,
                severity='high'
            ),
            ValidationRule(
                'max_length',
                lambda x: len(str(x).strip()) <= 100,
                0.7,
                is_boost=False,
                severity='medium'
            ),
            ValidationRule(
                'no_excessive_digits',
                lambda x: len(self._DIGIT_PATTERN.findall(str(x))) <= 3,
                0.75,
                is_boost=False,
                severity='medium'
            ),
            ValidationRule(
                'has_suffix',
                lambda x: any(s in str(x).lower() for s in self._DEALER_SUFFIXES),
                1.15,
                is_boost=True,
                severity='low'
            ),
            ValidationRule(
                'capitalized',
                lambda x: any(c.isupper() for c in str(x)),
                1.05,
                is_boost=True,
                severity='low'
            ),
            ValidationRule(
                'no_special_chars',
                lambda x: not self._SPECIAL_CHAR_PATTERN.search(str(x)),
                1.08,
                is_boost=True,
                severity='low'
            )
        ]
    
    def _get_model_rules(self) -> List[ValidationRule]:
        """Validation rules for model name"""
        return [
            ValidationRule(
                'min_length',
                lambda x: len(str(x).strip()) >= 2,
                0.6,
                is_boost=False,
                severity='high'
            ),
            ValidationRule(
                'max_length',
                lambda x: len(str(x).strip()) <= 30,
                0.7,
                is_boost=False,
                severity='medium'
            ),
            ValidationRule(
                'pattern_match',
                lambda x: bool(self._MODEL_PATTERN.search(str(x))),
                1.20,
                is_boost=True,
                severity='low'
            ),
            ValidationRule(
                'has_brand',
                lambda x: any(b in str(x).upper() for b in self._COMMON_BRANDS),
                1.15,
                is_boost=True,
                severity='low'
            ),
            ValidationRule(
                'has_digits',
                lambda x: bool(self._DIGIT_PATTERN.search(str(x))),
                1.10,
                is_boost=True,
                severity='low'
            )
        ]
    
    def _get_hp_rules(self) -> List[ValidationRule]:
        """Validation rules for horse power"""
        return [
            ValidationRule(
                'is_numeric',
                lambda x: isinstance(x, (int, float)) and x > 0,
                0.5,
                is_boost=False,
                severity='high'
            ),
            ValidationRule(
                'in_valid_range',
                lambda x: self.HP_MIN <= float(x) <= self.HP_MAX if isinstance(x, (int, float)) else False,
                1.15,
                is_boost=True,
                severity='medium'
            ),
            ValidationRule(
                'reasonable_value',
                lambda x: 20 <= float(x) <= 100 if isinstance(x, (int, float)) else False,
                1.08,
                is_boost=True,
                severity='low'
            ),
            ValidationRule(
                'not_fractional',
                lambda x: float(x) == int(float(x)) if isinstance(x, (int, float)) else False,
                1.05,
                is_boost=True,
                severity='low'
            )
        ]
    
    def _get_cost_rules(self) -> List[ValidationRule]:
        """Validation rules for asset cost"""
        return [
            ValidationRule(
                'is_numeric',
                lambda x: isinstance(x, (int, float)) and x > 0,
                0.5,
                is_boost=False,
                severity='high'
            ),
            ValidationRule(
                'in_valid_range',
                lambda x: self.COST_MIN <= float(x) <= self.COST_MAX if isinstance(x, (int, float)) else False,
                1.15,
                is_boost=True,
                severity='medium'
            ),
            ValidationRule(
                'reasonable_rounding',
                lambda x: float(x) % 1000 == 0 if isinstance(x, (int, float)) else False,
                1.06,
                is_boost=True,
                severity='low'
            ),
            ValidationRule(
                'typical_range',
                lambda x: 200000 <= float(x) <= 2000000 if isinstance(x, (int, float)) else False,
                1.08,
                is_boost=True,
                severity='low'
            )
        ]
    
    def _get_signature_rules(self) -> List[ValidationRule]:
        """Validation rules for signature detection"""
        return [
            ValidationRule(
                'has_bbox',
                lambda x: x.get('bbox') is not None and len(x.get('bbox', [])) == 4,
                1.10,
                is_boost=True,
                severity='medium'
            ),
            ValidationRule(
                'reasonable_size',
                lambda x: self._check_bbox_size(x.get('bbox'), min_area=100, max_area=100000),
                1.08,
                is_boost=True,
                severity='low'
            )
        ]
    
    def _get_stamp_rules(self) -> List[ValidationRule]:
        """Validation rules for stamp detection"""
        return [
            ValidationRule(
                'has_bbox',
                lambda x: x.get('bbox') is not None and len(x.get('bbox', [])) == 4,
                1.10,
                is_boost=True,
                severity='medium'
            ),
            ValidationRule(
                'reasonable_size',
                lambda x: self._check_bbox_size(x.get('bbox'), min_area=400, max_area=200000),
                1.08,
                is_boost=True,
                severity='low'
            )
        ]
    
    @staticmethod
    def _check_bbox_size(bbox: Optional[List], min_area: int, max_area: int) -> bool:
        """Check if bounding box has reasonable size"""
        if not bbox or len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        return min_area <= area <= max_area
    
    def validate_document(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all extracted fields with comprehensive checks
        
        Args:
            extracted_data: Dictionary with extracted fields
            
        Returns:
            Validated data with adjusted confidences and issues
        """
        self.validation_stats['total_validations'] += 1
        validated = extracted_data.copy()
        
        # Apply individual field rules
        for field_name, config in self.field_configs.items():
            if field_name in validated:
                self._apply_validation_rules(validated[field_name], config, field_name)
        
        # Cross-field validations
        self._apply_cross_validations(validated)
        
        # Recalculate overall confidence
        validated['overall_confidence'] = self._calculate_overall_confidence(validated)
        
        # Check if validation passed
        if validated['overall_confidence'] >= 0.75:
            self.validation_stats['passed_validations'] += 1
        
        return validated
    
    def _apply_validation_rules(
        self, 
        field_data: Dict[str, Any], 
        config: FieldConfig,
        field_name: str
    ):
        """Apply validation rules to a single field"""
        value = field_data.get('value')
        
        # Special handling for signature/stamp
        if field_name in ['signature', 'stamp']:
            if not field_data.get('present'):
                return
            value = field_data  # Pass entire dict for bbox checks
        
        if value is None:
            return
        
        confidence = field_data.get('confidence', 0.5)
        issues = []
        boosts = []
        
        # Apply all rules
        for rule in config.rules:
            try:
                passes = rule.condition(value)
                
                if passes and rule.is_boost:
                    confidence *= rule.confidence_multiplier
                    boosts.append(rule.name)
                elif not passes and not rule.is_boost:
                    confidence *= rule.confidence_multiplier
                    issues.append({
                        'rule': rule.name,
                        'severity': rule.severity
                    })
                    
                    # Track field failures
                    if rule.severity == 'high':
                        if field_name not in self.validation_stats['field_failures']:
                            self.validation_stats['field_failures'][field_name] = 0
                        self.validation_stats['field_failures'][field_name] += 1
                    
            except Exception as e:
                logger.debug(f"Rule '{rule.name}' failed for {field_name}: {e}")
                continue
        
        # Clamp confidence
        field_data['confidence'] = np.clip(
            confidence, 
            config.min_confidence, 
            config.max_confidence
        )
        
        # Add metadata
        if issues:
            field_data['validation_issues'] = issues
        if boosts:
            field_data['validation_boosts'] = boosts
    
    def _apply_cross_validations(self, data: Dict[str, Any]):
        """Apply cross-field validations"""
        # Model-HP validation
        if 'model_name' in data and 'horse_power' in data:
            self._cross_validate_model_hp(data)
        
        # HP-Cost validation
        if 'horse_power' in data and 'asset_cost' in data:
            self._cross_validate_hp_cost(data)
        
        # Dealer-Model brand consistency
        if 'dealer_name' in data and 'model_name' in data:
            self._cross_validate_dealer_model(data)
    
    @lru_cache(maxsize=512)
    def _find_model_hp_range(self, model_str: str) -> Optional[Tuple[int, int]]:
        """Find expected HP range for model (cached)"""
        model_upper = str(model_str).upper().strip()
        
        # Exact match first
        if model_upper in self.model_hp_map:
            return self.model_hp_map[model_upper]
        
        # Partial match
        for model_key, hp_range in self.model_hp_map.items():
            if model_key in model_upper or model_upper in model_key:
                return hp_range
        
        return None
    
    def _cross_validate_model_hp(self, data: Dict[str, Any]):
        """Cross-validate model name with horse power"""
        model_data = data['model_name']
        hp_data = data['horse_power']
        
        model_value = model_data.get('value')
        hp_value = hp_data.get('value')
        
        if not (model_value and isinstance(hp_value, (int, float))):
            return
        
        hp_range = self._find_model_hp_range(str(model_value))
        
        if hp_range:
            min_hp, max_hp = hp_range
            
            # Check if HP is within expected range
            if min_hp <= hp_value <= max_hp:
                # Boost both confidences
                multiplier = 1.18
                logger.debug(f"✓ Model-HP match: {model_value} ({min_hp}-{max_hp}) ↔ {hp_value} HP")
            else:
                # Penalty for mismatch
                multiplier = 0.75
                logger.warning(
                    f"✗ Model-HP mismatch: {model_value} expects ({min_hp}-{max_hp}) "
                    f"but got {hp_value} HP"
                )
                
                # Add cross-validation issue
                if 'cross_validation_issues' not in model_data:
                    model_data['cross_validation_issues'] = []
                model_data['cross_validation_issues'].append('hp_mismatch')
            
            # Apply multiplier
            model_data['confidence'] = min(
                model_data.get('confidence', 0.5) * multiplier,
                self.field_configs['model_name'].max_confidence
            )
            hp_data['confidence'] = min(
                hp_data.get('confidence', 0.5) * multiplier,
                self.field_configs['horse_power'].max_confidence
            )
    
    def _cross_validate_hp_cost(self, data: Dict[str, Any]):
        """Cross-validate horse power with asset cost"""
        hp_value = data['horse_power'].get('value')
        cost_value = data['asset_cost'].get('value')
        
        if not (isinstance(hp_value, (int, float)) and isinstance(cost_value, (int, float))):
            return
        
        # Calculate expected cost range
        expected_min = hp_value * self.COST_PER_HP_MIN
        expected_max = hp_value * self.COST_PER_HP_MAX
        cost_per_hp = cost_value / hp_value
        
        if expected_min <= cost_value <= expected_max:
            multiplier = 1.12
            logger.debug(
                f"✓ HP-Cost match: {hp_value} HP @ ₹{cost_per_hp:.0f}/HP "
                f"(range: ₹{self.COST_PER_HP_MIN}-{self.COST_PER_HP_MAX})"
            )
        else:
            # Check severity of mismatch
            if cost_value < expected_min * 0.5 or cost_value > expected_max * 2:
                multiplier = 0.65  # Severe mismatch
                logger.warning(
                    f"✗ Severe HP-Cost mismatch: {hp_value} HP @ ₹{cost_per_hp:.0f}/HP"
                )
            else:
                multiplier = 0.85  # Moderate mismatch
                logger.warning(
                    f"⚠ Moderate HP-Cost mismatch: {hp_value} HP @ ₹{cost_per_hp:.0f}/HP"
                )
        
        # Apply multiplier
        data['horse_power']['confidence'] = min(
            data['horse_power'].get('confidence', 0.5) * multiplier,
            self.field_configs['horse_power'].max_confidence
        )
        data['asset_cost']['confidence'] = min(
            data['asset_cost'].get('confidence', 0.5) * multiplier,
            self.field_configs['asset_cost'].max_confidence
        )
    
    def _cross_validate_dealer_model(self, data: Dict[str, Any]):
        """Check dealer-model brand consistency"""
        dealer_name = str(data['dealer_name'].get('value', '')).upper()
        model_name = str(data['model_name'].get('value', '')).upper()
        
        # Check if dealer name contains brand matching model
        brand_match = False
        for brand in self._COMMON_BRANDS:
            if brand in dealer_name and brand in model_name:
                brand_match = True
                break
        
        if brand_match:
            # Slight boost for consistency
            multiplier = 1.06
            logger.debug("✓ Dealer-Model brand consistency")
        else:
            # No penalty (dealer might sell multiple brands)
            multiplier = 1.0
        
        data['dealer_name']['confidence'] = min(
            data['dealer_name'].get('confidence', 0.5) * multiplier,
            self.field_configs['dealer_name'].max_confidence
        )
    
    def _calculate_overall_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate weighted overall confidence"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field_name, config in self.field_configs.items():
            if field_name not in data:
                continue
            
            field_data = data[field_name]
            
            # Binary fields
            if field_name in ('signature', 'stamp'):
                if field_data.get('present'):
                    confidence = field_data.get('confidence', 0.0)
                    weighted_sum += confidence * config.weight
                    total_weight += config.weight
            # Value fields
            elif field_data.get('value') is not None:
                confidence = field_data.get('confidence', 0.0)
                weighted_sum += confidence * config.weight
                total_weight += config.weight
        
        return float(weighted_sum / total_weight) if total_weight > 0 else 0.0
    
    def check_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check document completeness"""
        required_fields = [
            name for name, config in self.field_configs.items() 
            if config.required
        ]
        
        complete = []
        missing = []
        
        for field in required_fields:
            if field in data:
                field_data = data[field]
                if field in ['signature', 'stamp']:
                    if field_data.get('present'):
                        complete.append(field)
                    else:
                        missing.append(field)
                elif field_data.get('value') is not None:
                    complete.append(field)
                else:
                    missing.append(field)
            else:
                missing.append(field)
        
        completeness_score = len(complete) / len(required_fields) if required_fields else 1.0
        
        return {
            'missing_fields': missing,
            'complete_fields': complete,
            'completeness_score': completeness_score,
            'total_required': len(required_fields),
            'total_complete': len(complete)
        }
    
    def get_validation_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        completeness = self.check_completeness(data)
        overall_conf = data.get('overall_confidence', 0.0)
        
        # Collect all issues
        all_issues = {}
        high_severity_issues = []
        
        for field_name, field_data in data.items():
            if not isinstance(field_data, dict):
                continue
            
            issues = field_data.get('validation_issues', [])
            if issues:
                all_issues[field_name] = issues
                
                # Track high severity
                for issue in issues:
                    if isinstance(issue, dict) and issue.get('severity') == 'high':
                        high_severity_issues.append(f"{field_name}: {issue['rule']}")
        
        # Determine quality level
        if overall_conf >= 0.85 and completeness['completeness_score'] >= 0.8:
            quality = 'high'
        elif overall_conf >= 0.65 and completeness['completeness_score'] >= 0.6:
            quality = 'medium'
        else:
            quality = 'low'
        
        # Generate recommendations
        recommendations = []
        
        if completeness['completeness_score'] < 1.0:
            recommendations.append(
                f"Missing required fields: {', '.join(completeness['missing_fields'])}"
            )
        
        if overall_conf < 0.75:
            recommendations.append("Review low-confidence fields manually")
        
        if high_severity_issues:
            recommendations.append(
                f"Critical validation failures: {', '.join(high_severity_issues)}"
            )
        
        if overall_conf < 0.50:
            recommendations.append("Consider re-scanning or manual data entry")
        
        # Calculate field confidence stats
        field_confidences = []
        for field_name in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            if field_name in data and data[field_name].get('value') is not None:
                field_confidences.append(data[field_name].get('confidence', 0.0))
        
        return {
            'overall_confidence': overall_conf,
            'completeness_score': completeness['completeness_score'],
            'quality_level': quality,
            'missing_fields': completeness['missing_fields'],
            'complete_fields': completeness['complete_fields'],
            'validation_issues': all_issues,
            'high_severity_issues': high_severity_issues,
            'recommendations': recommendations,
            'fields_validated': len(field_confidences),
            'min_field_confidence': min(field_confidences) if field_confidences else 0.0,
            'max_field_confidence': max(field_confidences) if field_confidences else 0.0,
            'avg_field_confidence': np.mean(field_confidences) if field_confidences else 0.0,
            'document_ready': quality == 'high' and not high_severity_issues
        }
    
    def batch_validate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate multiple documents"""
        return [self.validate_document(doc) for doc in documents]
    
    def add_custom_rule(self, field_name: str, rule: ValidationRule):
        """Add custom validation rule"""
        if field_name in self.field_configs:
            self.field_configs[field_name].rules.append(rule)
            logger.info(f"Added custom rule '{rule.name}' to {field_name}")
            
            # Clear cache if model-related
            if field_name == 'model_name':
                self._find_model_hp_range.cache_clear()
        else:
            logger.warning(f"Field '{field_name}' not found")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            'total_validations': self.validation_stats['total_validations'],
            'passed_validations': self.validation_stats['passed_validations'],
            'pass_rate': (
                self.validation_stats['passed_validations'] / 
                self.validation_stats['total_validations']
                if self.validation_stats['total_validations'] > 0 else 0.0
            ),
            'field_failures': self.validation_stats['field_failures']
        }
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'field_failures': {}
        }


# Singleton accessor
def get_validator() -> DocumentValidator:
    """Get validator instance (singleton)"""
    return DocumentValidator()


if __name__ == "__main__":
    print("=" * 70)
    print("Document Validator - Testing")
    print("=" * 70)
    
    validator = get_validator()
    
    # Test document 1: Good quality
    print("\n📄 Test 1: High Quality Document")
    test_doc1 = {
        'dealer_name': {'value': 'Mahindra Tractors Pvt Ltd', 'confidence': 0.88},
        'model_name': {'value': '575 DI', 'confidence': 0.85},
        'horse_power': {'value': 50.0, 'confidence': 0.82},
        'asset_cost': {'value': 650000.0, 'confidence': 0.86},
        'signature': {'present': True, 'confidence': 0.75, 'bbox': [100, 400, 200, 450]},
        'stamp': {'present': True, 'confidence': 0.72, 'bbox': [250, 400, 350, 480]}
    }
    
    validated1 = validator.validate_document(test_doc1)
    summary1 = validator.get_validation_summary(validated1)
    
    print(f"Quality: {summary1['quality_level'].upper()}")
    print(f"Overall Confidence: {summary1['overall_confidence']:.1%}")
    print(f"Completeness: {summary1['completeness_score']:.0%}")
    print(f"Document Ready: {'✓ YES' if summary1['document_ready'] else '✗ NO'}")
    
    # Test document 2: Issues
    print("\n\n📄 Test 2: Document with Issues")
    test_doc2 = {
        'dealer_name': {'value': 'XYZ123!@#', 'confidence': 0.65},
        'model_name': {'value': '999', 'confidence': 0.60},
        'horse_power': {'value': 30.0, 'confidence': 0.55},
        'asset_cost': {'value': 5000.0, 'confidence': 0.50}  # Way too low
    }
    
    validated2 = validator.validate_document(test_doc2)
    summary2 = validator.get_validation_summary(validated2)
    
    print(f"Quality: {summary2['quality_level'].upper()}")
    print(f"Overall Confidence: {summary2['overall_confidence']:.1%}")
    print(f"Completeness: {summary2['completeness_score']:.0%}")
    
    if summary2['recommendations']:
        print("\nRecommendations:")
        for rec in summary2['recommendations']:
            print(f"  • {rec}")
    
    # Show validation stats
    print("\n\n📊 Validation Statistics:")
    stats = validator.get_validation_stats()
    print(f"Total Validations: {stats['total_validations']}")
    print(f"Pass Rate: {stats['pass_rate']:.1%}")
    
    print("\n✅ Validation tests completed!")