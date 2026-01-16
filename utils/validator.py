import re
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Validation rule with condition and confidence modifier"""
    name: str
    condition: Callable[[Any], bool]
    confidence_multiplier: float
    is_boost: bool = True  # True for boost, False for penalty


@dataclass
class FieldConfig:
    """Configuration for field validation"""
    required: bool = True
    weight: float = 0.15
    min_confidence: float = 0.3
    max_confidence: float = 0.95
    rules: List[ValidationRule] = field(default_factory=list)


class EfficientDocumentValidator:
    """Optimized document validator with configurable rules and caching"""
    
    _instance = None
    _initialized = False
    
    # Pre-compiled regex patterns (class-level for sharing)
    _MODEL_PATTERN = re.compile(r'(\d{3,4}\s*[A-Z]{1,3})', re.IGNORECASE)
    _DIGIT_PATTERN = re.compile(r'\d')
    
    # Frozen sets for O(1) lookups
    _DEALER_SUFFIXES = frozenset(['tractors', 'motors', 'agency', 'enterprises', 
                                   'pvt', 'ltd', 'company', 'corp', 'inc'])
    _COMMON_BRANDS = frozenset(['MAHINDRA', 'SWARAJ', 'ESCORTS', 'EICHER', 
                                 'JOHN', 'DEERE', 'CASE', 'NEW HOLLAND', 'MASSEY'])
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize validator with efficient data structures"""
        if self._initialized:
            return
        
        # Use dict for O(1) model HP lookups
        self.model_hp_map = {
            '475': (40, 50), '485': (45, 55), '575': (48, 58), '595': (52, 62),
            'DI': (45, 55), 'XP': (50, 60), 'XT': (55, 65), 'MX': (60, 70),
            'SWARAJ': (30, 50), 'MAHINDRA': (40, 70), 'EICHER': (35, 55)
        }
        
        # HP range constants
        self.HP_MIN, self.HP_MAX = 10, 200
        
        # Cost range constants (in INR)
        self.COST_MIN, self.COST_MAX = 100000, 5000000
        self.COST_PER_HP_MIN, self.COST_PER_HP_MAX = 5000, 20000
        
        # Field configurations
        self.field_configs = {
            'dealer_name': FieldConfig(weight=0.15, rules=self._get_dealer_rules()),
            'model_name': FieldConfig(weight=0.20, rules=self._get_model_rules()),
            'horse_power': FieldConfig(weight=0.20, rules=self._get_hp_rules()),
            'asset_cost': FieldConfig(weight=0.25, rules=self._get_cost_rules()),
            'signature': FieldConfig(weight=0.10, required=False),
            'stamp': FieldConfig(weight=0.10, required=False)
        }
        
        self._initialized = True
    
    def _get_dealer_rules(self) -> List[ValidationRule]:
        """Get validation rules for dealer name"""
        return [
            ValidationRule('min_length', lambda x: len(str(x)) >= 3, 0.7, False),
            ValidationRule('no_digits', lambda x: not self._DIGIT_PATTERN.search(str(x)), 0.8, False),
            ValidationRule('has_suffix', lambda x: any(s in str(x).lower() for s in self._DEALER_SUFFIXES), 1.1, True)
        ]
    
    def _get_model_rules(self) -> List[ValidationRule]:
        """Get validation rules for model name"""
        return [
            ValidationRule('pattern_match', lambda x: bool(self._MODEL_PATTERN.search(str(x))), 1.2, True),
            ValidationRule('min_length', lambda x: len(str(x)) >= 2, 0.7, False),
            ValidationRule('has_brand', lambda x: any(b in str(x).upper() for b in self._COMMON_BRANDS), 1.1, True)
        ]
    
    def _get_hp_rules(self) -> List[ValidationRule]:
        """Get validation rules for horse power"""
        return [
            ValidationRule('in_range', lambda x: self.HP_MIN <= float(x) <= self.HP_MAX if isinstance(x, (int, float)) else False, 1.1, True)
        ]
    
    def _get_cost_rules(self) -> List[ValidationRule]:
        """Get validation rules for asset cost"""
        return [
            ValidationRule('positive', lambda x: float(x) > 0 if isinstance(x, (int, float)) else False, 0.5, False),
            ValidationRule('in_range', lambda x: self.COST_MIN <= float(x) <= self.COST_MAX if isinstance(x, (int, float)) else False, 1.1, True)
        ]
    
    def validate_document(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all extracted fields efficiently
        
        Args:
            extracted_data: Dictionary with extracted fields
            
        Returns:
            Validated data with adjusted confidences
        """
        validated = extracted_data.copy()
        
        # Validate individual fields with rules
        for field_name, config in self.field_configs.items():
            if field_name in validated:
                self._apply_validation_rules(validated[field_name], config)
        
        # Cross-field validations (only if both fields exist)
        if 'model_name' in validated and 'horse_power' in validated:
            self._cross_validate_model_hp(validated)
        
        if 'horse_power' in validated and 'asset_cost' in validated:
            self._cross_validate_hp_cost(validated)
        
        # Recalculate overall confidence
        validated['overall_confidence'] = self._calculate_overall_confidence(validated)
        
        return validated
    
    def _apply_validation_rules(self, field_data: Dict[str, Any], config: FieldConfig):
        """Apply validation rules to a field"""
        value = field_data.get('value')
        if value is None:
            return
        
        confidence = field_data.get('confidence', 0.5)
        issues = []
        
        # Apply all rules
        for rule in config.rules:
            try:
                if rule.condition(value):
                    if rule.is_boost:
                        confidence *= rule.confidence_multiplier
                else:
                    if not rule.is_boost:
                        confidence *= rule.confidence_multiplier
                        issues.append(rule.name)
            except Exception as e:
                logger.debug(f"Rule {rule.name} failed: {e}")
                continue
        
        # Clamp confidence
        field_data['confidence'] = max(config.min_confidence, min(confidence, config.max_confidence))
        
        if issues:
            field_data['validation_issues'] = issues
    
    @lru_cache(maxsize=256)
    def _find_model_hp_range(self, model_str: str) -> Optional[Tuple[int, int]]:
        """Cached model HP range lookup"""
        model_upper = model_str.upper()
        for model_key, hp_range in self.model_hp_map.items():
            if model_key in model_upper:
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
            multiplier = 1.15 if min_hp <= hp_value <= max_hp else 0.85
            
            model_data['confidence'] = min(model_data.get('confidence', 0.5) * multiplier, 0.95)
            hp_data['confidence'] = min(hp_data.get('confidence', 0.5) * multiplier, 0.95)
            
            logger.debug(f"Model-HP cross-validation: {model_value} ({min_hp}-{max_hp} HP) vs {hp_value} HP")
    
    def _cross_validate_hp_cost(self, data: Dict[str, Any]):
        """Cross-validate horse power with asset cost"""
        hp_value = data['horse_power'].get('value')
        cost_value = data['asset_cost'].get('value')
        
        if not (isinstance(hp_value, (int, float)) and isinstance(cost_value, (int, float))):
            return
        
        expected_min = hp_value * self.COST_PER_HP_MIN
        expected_max = hp_value * self.COST_PER_HP_MAX
        
        multiplier = 1.1 if expected_min <= cost_value <= expected_max else 0.95
        
        data['horse_power']['confidence'] = min(data['horse_power'].get('confidence', 0.5) * multiplier, 0.95)
        data['asset_cost']['confidence'] = min(data['asset_cost'].get('confidence', 0.5) * multiplier, 0.95)
    
    def _calculate_overall_confidence(self, data: Dict[str, Any]) -> float:
        """
        Calculate weighted overall confidence
        
        Args:
            data: Validated data with field confidences
            
        Returns:
            Overall confidence score (0-1)
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field_name, config in self.field_configs.items():
            if field_name not in data:
                continue
            
            field_data = data[field_name]
            
            # For binary fields (signature/stamp)
            if field_name in ('signature', 'stamp'):
                if field_data.get('present'):
                    confidence = field_data.get('confidence', 0.5)
                    weighted_sum += confidence * config.weight
                    total_weight += config.weight
            # For value fields
            elif field_data.get('value') is not None:
                confidence = field_data.get('confidence', 0.5)
                weighted_sum += confidence * config.weight
                total_weight += config.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def check_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check document completeness efficiently
        
        Args:
            data: Extracted data
            
        Returns:
            Completeness report with score and field lists
        """
        required_fields = [name for name, config in self.field_configs.items() if config.required]
        
        complete = [f for f in required_fields if f in data and data[f].get('value') is not None]
        missing = [f for f in required_fields if f not in complete]
        
        return {
            'missing_fields': missing,
            'complete_fields': complete,
            'completeness_score': len(complete) / len(required_fields) if required_fields else 1.0,
            'total_required': len(required_fields)
        }
    
    def get_validation_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive validation summary
        
        Args:
            data: Validated document data
            
        Returns:
            Summary with scores, issues, and recommendations
        """
        completeness = self.check_completeness(data)
        overall_conf = data.get('overall_confidence', 0.0)
        
        # Collect all validation issues
        all_issues = {}
        for field_name, field_data in data.items():
            if isinstance(field_data, dict) and 'validation_issues' in field_data:
                all_issues[field_name] = field_data['validation_issues']
        
        # Determine quality level
        quality = 'high' if overall_conf >= 0.8 else 'medium' if overall_conf >= 0.6 else 'low'
        
        # Generate recommendations
        recommendations = []
        if completeness['completeness_score'] < 1.0:
            recommendations.append(f"Missing fields: {', '.join(completeness['missing_fields'])}")
        if overall_conf < 0.7:
            recommendations.append("Review low-confidence fields manually")
        if all_issues:
            recommendations.append(f"Validation issues in: {', '.join(all_issues.keys())}")
        
        return {
            'overall_confidence': overall_conf,
            'completeness_score': completeness['completeness_score'],
            'quality_level': quality,
            'missing_fields': completeness['missing_fields'],
            'validation_issues': all_issues,
            'recommendations': recommendations,
            'fields_validated': len([f for f in data if isinstance(data[f], dict) and 'confidence' in data[f]])
        }
    
    def batch_validate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate multiple documents efficiently
        
        Args:
            documents: List of extracted document data
            
        Returns:
            List of validated documents
        """
        return [self.validate_document(doc) for doc in documents]
    
    def add_custom_rule(self, field_name: str, rule: ValidationRule):
        """Add a custom validation rule to a field"""
        if field_name in self.field_configs:
            self.field_configs[field_name].rules.append(rule)
            # Clear cache if model-related
            if field_name == 'model_name':
                self._find_model_hp_range.cache_clear()
        else:
            logger.warning(f"Field {field_name} not found in configurations")


# Singleton accessor
def get_validator() -> EfficientDocumentValidator:
    """Get validator instance (singleton)"""
    return EfficientDocumentValidator()


if __name__ == "__main__":
    print("Testing Efficient Document Validator\n")
    
    validator = get_validator()
    
    # Test data
    test_doc = {
        'dealer_name': {'value': 'ABC Tractors Pvt Ltd', 'confidence': 0.85},
        'model_name': {'value': '575 DI', 'confidence': 0.80},
        'horse_power': {'value': 50, 'confidence': 0.75},
        'asset_cost': {'value': 525000, 'confidence': 0.82},
        'signature': {'present': True, 'confidence': 0.70, 'bbox': [10, 20, 100, 80]},
        'stamp': {'present': True, 'confidence': 0.65}
    }
    
    print("📄 Original Data:")
    for field, data in test_doc.items():
        if isinstance(data, dict) and 'value' in data:
            print(f"  {field}: {data['value']} (conf: {data['confidence']:.2%})")
    
    # Validate
    validated = validator.validate_document(test_doc)
    
    print("\n✓ Validated Data:")
    for field, data in validated.items():
        if isinstance(data, dict) and 'confidence' in data:
            conf = data['confidence']
            issues = data.get('validation_issues', [])
            issues_str = f" [Issues: {', '.join(issues)}]" if issues else ""
            
            if 'value' in data:
                print(f"  {field}: {data['value']} (conf: {conf:.2%}){issues_str}")
            else:
                print(f"  {field}: present={data.get('present')} (conf: {conf:.2%}){issues_str}")
    
    print(f"\n📊 Overall Confidence: {validated['overall_confidence']:.2%}")
    
    # Completeness check
    completeness = validator.check_completeness(validated)
    print(f"\n📋 Completeness: {completeness['completeness_score']:.0%}")
    print(f"   Complete: {', '.join(completeness['complete_fields'])}")
    if completeness['missing_fields']:
        print(f"   Missing: {', '.join(completeness['missing_fields'])}")
    
    # Summary
    summary = validator.get_validation_summary(validated)
    print(f"\n🎯 Quality Level: {summary['quality_level'].upper()}")
    print(f"   Fields Validated: {summary['fields_validated']}")
    
    if summary['recommendations']:
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"   • {rec}")
    
    # Test custom rule
    print("\n\n🔧 Testing Custom Rule...")
    custom_rule = ValidationRule(
        'premium_model',
        lambda x: 'PREMIUM' in str(x).upper(),
        1.3,
        True
    )
    validator.add_custom_rule('model_name', custom_rule)
    
    test_doc2 = test_doc.copy()
    test_doc2['model_name'] = {'value': '575 DI PREMIUM', 'confidence': 0.70}
    validated2 = validator.validate_document(test_doc2)
    
    print(f"Model with custom rule: {validated2['model_name']['value']}")
    print(f"Confidence boost: {validated2['model_name']['confidence']:.2%}")
    
    print("\n✅ Validation tests completed!")