import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from thefuzz import fuzz, process
from functools import lru_cache
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleBasedExtractor:
    """
    Optimized rule-based extractor with compiled patterns and caching
    """
    
    # Class-level constants
    FIELD_WEIGHTS = {
        'dealer_name': 0.20,
        'model_name': 0.25,
        'horse_power': 0.20,
        'asset_cost': 0.25,
        'signature': 0.05,
        'stamp': 0.05
    }
    
    COMMON_BRANDS = frozenset([
        'MAHINDRA', 'SWARAJ', 'ESCORTS', 'EICHER', 
        'JOHN DEERE', 'CASE', 'NEW HOLLAND', 'SONALIKA'
    ])
    
    HP_RANGE = (10, 200)
    COST_RANGE = (100000, 5000000)
    
    def __init__(self, master_data_path: Optional[str] = None):
        """Initialize with compiled patterns and master data"""
        self.dealer_master = []
        self.model_master = []
        self._dealer_set = set()
        self._model_set = set()
        
        if master_data_path:
            self._load_master_data(master_data_path)
        
        # Compile all patterns once
        self._compile_patterns()
        
        # Model-HP validation map
        self._model_hp_map = {
            '475': 45, '485': 48, '575': 50, '595': 55,
            'DI': 50, 'XP': 55, 'XT': 60, 'MX': 65
        }
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency"""
        # Dealer patterns - combined for efficiency
        self.dealer_patterns = [
            re.compile(r'(?:dealer|seller|vendor|supplier|distributor)[\s:]+([A-Za-z\s&.,]{3,50})', re.I),
            re.compile(r'([A-Za-z\s&.,]{3,50})\s*(?:dealer|seller|vendor)', re.I),
            re.compile(r'^(?:M/s|M\.s\.|M\/s)[\s:]+([A-Za-z\s&.,]{3,50})', re.I | re.M),
            re.compile(r'(?:डीलर|विक्रेता|ડીલર)[\s:]+([^\n]{3,50})', re.I)
        ]
        
        # Model patterns - optimized
        self.model_patterns = [
            re.compile(r'(?:model|type|variant)[\s:]+([A-Za-z0-9\s\-]{2,30})', re.I),
            re.compile(r'\b(\d{3,4}\s*(?:DI|XP|XT|MX|TA|FE|GT|R|S|E|PRO|PLUS))\b', re.I),
            re.compile(r'(?:मॉडल|મોડેલ)[\s:]+([^\n]{2,30})', re.I),
            re.compile(r'tractor[\s:]+([A-Za-z0-9\s\-]{2,30})', re.I)
        ]
        
        # HP patterns - combined
        self.hp_patterns = [
            re.compile(r'(?:horse\s*power|hp|h\.p\.)[\s:]*(\d+(?:\.\d+)?)', re.I),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:hp|h\.p\.|horse\s*power)', re.I),
            re.compile(r'(?:हॉर्स\s*पावर|શક્તિ)[\s:]*(\d+(?:\.\d+)?)', re.I)
        ]
        
        # Cost patterns - optimized for common formats
        self.cost_patterns = [
            re.compile(r'(?:total\s*cost|asset\s*cost|price|amount|value)[\s:]*[₹$Rs]?\s*([0-9,]+(?:\.[0-9]+)?)', re.I),
            re.compile(r'[₹$Rs]\s*([0-9,]+(?:\.[0-9]+)?)', re.I),
            re.compile(r'([0-9,]+(?:\.[0-9]+)?)\s*(?:inr|rs|rupees)', re.I),
            re.compile(r'(?:कुल|કુલ)[\s:]*[₹रुરૂ]?\s*([0-9,]+)', re.I),
            re.compile(r'(\d[\d,]*\.?\d*)\s*(lakh|crore)', re.I)
        ]
        
        # Section detection - single pattern per section
        self.section_keywords = {
            'dealer': frozenset(['dealer', 'seller', 'vendor', 'supplier']),
            'model': frozenset(['model', 'tractor', 'variant', 'type']),
            'specs': frozenset(['specification', 'specs', 'technical', 'engine', 'power']),
            'financial': frozenset(['financial', 'cost', 'price', 'amount', 'total', 'payment']),
            'signature': frozenset(['signature', 'authorized', 'approved', 'sign']),
            'stamp': frozenset(['stamp', 'seal'])
        }
    
    def _load_master_data(self, path: str):
        """Load and cache master data"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                master_data = json.load(f)
            
            self.dealer_master = master_data.get('dealers', [])
            self.model_master = master_data.get('models', [])
            
            # Create sets for fast lookup
            self._dealer_set = set(d.lower() for d in self.dealer_master)
            self._model_set = set(m.upper() for m in self.model_master)
            
            logger.info(f"Loaded {len(self.dealer_master)} dealers, {len(self.model_master)} models")
            
        except Exception as e:
            logger.error(f"Error loading master data: {e}")
    
    def extract_fields(self, normalized_ocr_results: List[Dict[str, Any]], 
                      image_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Optimized field extraction with early returns
        """
        # Initialize results
        results = {
            'dealer_name': {'value': None, 'confidence': 0.0, 'source': None},
            'model_name': {'value': None, 'confidence': 0.0, 'source': None},
            'horse_power': {'value': None, 'confidence': 0.0, 'source': None},
            'asset_cost': {'value': None, 'confidence': 0.0, 'source': None},
            'signature': {'present': False, 'bbox': None, 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': None, 'confidence': 0.0}
        }
        
        if not normalized_ocr_results:
            results['overall_confidence'] = 0.0
            return results
        
        try:
            # Pre-process text blocks once
            text_blocks = self._prepare_text_blocks(normalized_ocr_results)
            
            # Extract all fields in parallel-friendly manner
            results['dealer_name'] = self._extract_dealer_name(text_blocks) or results['dealer_name']
            results['model_name'] = self._extract_model_name(text_blocks) or results['model_name']
            results['horse_power'] = self._extract_horse_power(text_blocks) or results['horse_power']
            results['asset_cost'] = self._extract_asset_cost(text_blocks) or results['asset_cost']
            
            # Calculate confidence
            results['overall_confidence'] = self._calculate_overall_confidence(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            results['overall_confidence'] = 0.0
            return results
    
    def _prepare_text_blocks(self, ocr_results: List[Dict]) -> List[Dict]:
        """Pre-process OCR results into efficient text blocks"""
        blocks = []
        
        for result in ocr_results:
            # Extract once, reuse many times
            text = result.get('text', '')
            normalized = result.get('normalized_text', text.lower())
            
            block = {
                'text': normalized,
                'original': text,
                'bbox': result.get('bbox', [0, 0, 0, 0]),
                'conf': result.get('confidence', 0.0),
                'nums': result.get('numbers', []),
                'keywords': result.get('contains_keywords', {}),
                'lang': result.get('language', 'unknown')
            }
            blocks.append(block)
        
        return blocks
    
    def _extract_dealer_name(self, text_blocks: List[Dict]) -> Optional[Dict]:
        """Optimized dealer extraction with early termination"""
        best_candidate = None
        best_score = 0.0
        
        # Strategy 1: Pattern matching (highest confidence)
        for block in text_blocks:
            if best_score >= 0.95:  # Early termination
                break
            
            original = block['original']
            
            for pattern in self.dealer_patterns:
                match = pattern.search(original)
                if match:
                    candidate = match.group(1).strip()
                    score = block['conf'] * 0.9
                    
                    # Fuzzy match if master data available
                    if self._dealer_set:
                        matched, match_pct = self._fuzzy_match_dealer(candidate)
                        if match_pct > 80:
                            candidate = matched
                            score = min(score * (match_pct / 100), 0.95)
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = {
                            'value': candidate,
                            'confidence': score,
                            'source': 'pattern'
                        }
                    break
        
        # Strategy 2: Keyword context (if no pattern match)
        if best_score < 0.7:
            for block in text_blocks:
                if block['keywords'].get('dealer', False):
                    # Extract capitalized words
                    words = [w for w in block['original'].split() 
                            if w and len(w) > 2 and w[0].isupper()]
                    
                    if words:
                        candidate = ' '.join(words[:3])
                        score = block['conf'] * 0.75
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = {
                                'value': candidate,
                                'confidence': score,
                                'source': 'keyword'
                            }
                        break
        
        # Strategy 3: Position-based (last resort)
        if best_score < 0.6 and text_blocks:
            # Sort by Y position
            sorted_blocks = sorted(text_blocks, key=lambda x: x['bbox'][1])[:3]
            
            for block in sorted_blocks:
                text = block['original'].strip()
                if len(text) > 10:
                    # Avoid headers
                    lower = text.lower()
                    if not any(w in lower for w in ['invoice', 'quotation', 'bill']):
                        score = block['conf'] * 0.6
                        
                        if score > best_score:
                            best_candidate = {
                                'value': text,
                                'confidence': score,
                                'source': 'position'
                            }
                        break
        
        return best_candidate
    
    def _extract_model_name(self, text_blocks: List[Dict]) -> Optional[Dict]:
        """Optimized model extraction"""
        best_candidate = None
        best_score = 0.0
        
        # Strategy 1: Pattern matching
        for block in text_blocks:
            if best_score >= 0.95:
                break
            
            original = block['original']
            
            for pattern in self.model_patterns:
                match = pattern.search(original)
                if match:
                    candidate = match.group(1).strip().upper()
                    score = block['conf'] * 0.95
                    
                    # Check against master data
                    if self._model_set:
                        if candidate in self._model_set:
                            score = min(score * 1.05, 0.98)
                        else:
                            matched, match_pct = self._fuzzy_match_model(candidate)
                            if match_pct > 90:
                                candidate = matched
                                score = min(score * (match_pct / 100), 0.98)
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = {
                            'value': candidate,
                            'confidence': score,
                            'source': 'pattern'
                        }
                    break
        
        # Strategy 2: Brand + model combination
        if best_score < 0.8:
            for block in text_blocks:
                text_upper = block['original'].upper()
                
                for brand in self.COMMON_BRANDS:
                    if brand in text_upper:
                        parts = text_upper.split(brand, 1)
                        if len(parts) > 1 and parts[1].strip():
                            # Get first word after brand
                            model_part = parts[1].strip().split()[0]
                            if len(model_part) >= 2:
                                candidate = f"{brand} {model_part}"
                                score = block['conf'] * 0.85
                                
                                if score > best_score:
                                    best_score = score
                                    best_candidate = {
                                        'value': candidate,
                                        'confidence': score,
                                        'source': 'brand_context'
                                    }
                                break
        
        return best_candidate
    
    def _extract_horse_power(self, text_blocks: List[Dict]) -> Optional[Dict]:
        """Optimized HP extraction with range validation"""
        best_candidate = None
        best_score = 0.0
        min_hp, max_hp = self.HP_RANGE
        
        # Strategy 1: Pattern matching
        for block in text_blocks:
            if best_score >= 0.95:
                break
            
            original = block['original']
            
            for pattern in self.hp_patterns:
                match = pattern.search(original)
                if match:
                    try:
                        hp_value = float(match.group(1))
                        
                        # Validate range
                        if min_hp <= hp_value <= max_hp:
                            score = block['conf'] * 0.95
                            
                            if score > best_score:
                                best_score = score
                                best_candidate = {
                                    'value': hp_value,
                                    'confidence': score,
                                    'source': 'pattern'
                                }
                            break
                    except (ValueError, IndexError):
                        continue
        
        # Strategy 2: Specification section
        if best_score < 0.7:
            spec_blocks = self._find_section_blocks(text_blocks, 'specs', context=3)
            
            for block in spec_blocks:
                for num in block['nums']:
                    if min_hp <= num <= max_hp:
                        score = block['conf'] * 0.8
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = {
                                'value': float(num),
                                'confidence': score,
                                'source': 'spec_section'
                            }
                        break
        
        return best_candidate
    
    def _extract_asset_cost(self, text_blocks: List[Dict]) -> Optional[Dict]:
        """Optimized cost extraction with multiplier handling"""
        best_candidate = None
        best_score = 0.0
        min_cost, max_cost = self.COST_RANGE
        
        # Strategy 1: Pattern matching
        for block in text_blocks:
            if best_score >= 0.9:
                break
            
            original = block['original']
            
            for pattern in self.cost_patterns:
                match = pattern.search(original)
                if match:
                    try:
                        # Handle multipliers in single pass
                        groups = match.groups()
                        cost_str = groups[0].replace(',', '')
                        cost_value = float(cost_str)
                        
                        # Check for multiplier in same match
                        if len(groups) > 1 and groups[1]:
                            multiplier_text = groups[1].lower()
                            if multiplier_text == 'lakh':
                                cost_value *= 100000
                            elif multiplier_text == 'crore':
                                cost_value *= 10000000
                        else:
                            # Check context
                            lower = original.lower()
                            if 'lakh' in lower:
                                cost_value *= 100000
                            elif 'crore' in lower:
                                cost_value *= 10000000
                        
                        # Validate range
                        if min_cost <= cost_value <= max_cost:
                            score = block['conf'] * 0.9
                            
                            if score > best_score:
                                best_score = score
                                best_candidate = {
                                    'value': cost_value,
                                    'confidence': score,
                                    'source': 'pattern'
                                }
                            break
                    except (ValueError, IndexError):
                        continue
        
        # Strategy 2: Financial section - find max valid number
        if best_score < 0.7:
            financial_blocks = self._find_section_blocks(text_blocks, 'financial', context=5)
            
            for block in financial_blocks:
                for num in block['nums']:
                    if min_cost <= num <= max_cost:
                        score = block['conf'] * 0.75
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = {
                                'value': float(num),
                                'confidence': score,
                                'source': 'financial_section'
                            }
        
        # Strategy 3: Currency symbols
        if best_score < 0.6:
            for block in text_blocks:
                if any(sym in block['original'] for sym in ['₹', 'Rs', '$']):
                    for num in block['nums']:
                        if min_cost <= num <= max_cost:
                            score = block['conf'] * 0.65
                            
                            if score > best_score:
                                best_score = score
                                best_candidate = {
                                    'value': float(num),
                                    'confidence': score,
                                    'source': 'currency_symbol'
                                }
        
        return best_candidate
    
    def _find_section_blocks(self, text_blocks: List[Dict], 
                            section: str, context: int = 3) -> List[Dict]:
        """Fast section block finder"""
        keywords = self.section_keywords.get(section, frozenset())
        
        for i, block in enumerate(text_blocks):
            text_lower = block['text']
            if any(kw in text_lower for kw in keywords):
                # Return this block plus context
                return text_blocks[i:min(i + context, len(text_blocks))]
        
        return []
    
    @lru_cache(maxsize=256)
    def _fuzzy_match_dealer(self, candidate: str) -> Tuple[str, float]:
        """Cached fuzzy matching for dealers"""
        if not self.dealer_master:
            return candidate, 0.0
        
        best_match, best_score = process.extractOne(
            candidate,
            self.dealer_master,
            scorer=fuzz.token_sort_ratio
        )
        
        return best_match, best_score
    
    @lru_cache(maxsize=256)
    def _fuzzy_match_model(self, candidate: str) -> Tuple[str, float]:
        """Cached fuzzy matching for models"""
        if not self.model_master:
            return candidate, 0.0
        
        best_match, best_score = process.extractOne(
            candidate,
            self.model_master,
            scorer=fuzz.token_set_ratio
        )
        
        return best_match, best_score
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Vectorized confidence calculation"""
        weights = []
        scores = []
        
        for field, weight in self.FIELD_WEIGHTS.items():
            if field in ['signature', 'stamp']:
                if results[field]['present']:
                    weights.append(weight)
                    scores.append(results[field]['confidence'])
            else:
                if results[field]['value'] is not None:
                    weights.append(weight)
                    scores.append(results[field]['confidence'])
        
        if not weights:
            return 0.0
        
        # Vectorized weighted average
        weights_array = np.array(weights)
        scores_array = np.array(scores)
        
        return float(np.sum(weights_array * scores_array) / np.sum(weights_array))
    
    def validate_extraction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fast validation with inline modifications"""
        validated = results.copy()
        
        # Validate HP
        hp = validated['horse_power'].get('value')
        if hp is not None:
            min_hp, max_hp = self.HP_RANGE
            if hp < min_hp or hp > max_hp:
                validated['horse_power']['confidence'] *= 0.5
                logger.warning(f"HP {hp} outside range ({min_hp}-{max_hp})")
        
        # Validate cost
        cost = validated['asset_cost'].get('value')
        if cost is not None:
            min_cost, max_cost = self.COST_RANGE
            if cost < min_cost or cost > max_cost:
                validated['asset_cost']['confidence'] *= 0.6
                logger.warning(f"Cost {cost} outside range ({min_cost}-{max_cost})")
        
        # Cross-validate model-HP
        model = validated['model_name'].get('value')
        if model and hp:
            model_str = str(model)
            for key, expected_hp in self._model_hp_map.items():
                if key in model_str:
                    if abs(hp - expected_hp) > 10:
                        validated['horse_power']['confidence'] *= 0.7
                        logger.warning(f"HP {hp} doesn't match model {model} (~{expected_hp})")
                    break
        
        # Recalculate confidence
        validated['overall_confidence'] = self._calculate_overall_confidence(validated)
        
        return validated
    
    def clear_cache(self):
        """Clear LRU caches"""
        self._fuzzy_match_dealer.cache_clear()
        self._fuzzy_match_model.cache_clear()


# Singleton
_extractor = None

def get_extractor(master_data_path: Optional[str] = None) -> RuleBasedExtractor:
    """Get or create extractor instance"""
    global _extractor
    if _extractor is None:
        _extractor = RuleBasedExtractor(master_data_path)
    return _extractor


if __name__ == "__main__":
    # Test the extractor
    extractor = RuleBasedExtractor()
    
    # Sample test data
    test_ocr_results = [
        {
            'text': 'Dealer: Mahindra Tractors Pvt Ltd',
            'normalized_text': 'dealer mahindra tractors pvt ltd',
            'bbox': [50, 100, 300, 120],
            'confidence': 0.95,
            'numbers': [],
            'contains_keywords': {'dealer': True},
            'language': 'en'
        },
        {
            'text': 'Model: 575 DI',
            'normalized_text': 'model 575 di',
            'bbox': [50, 150, 200, 170],
            'confidence': 0.92,
            'numbers': [575],
            'contains_keywords': {'model': True},
            'language': 'en'
        },
        {
            'text': 'Horse Power: 50 HP',
            'normalized_text': 'horse power 50 hp',
            'bbox': [50, 200, 250, 220],
            'confidence': 0.94,
            'numbers': [50],
            'contains_keywords': {'horse_power': True},
            'language': 'en'
        },
        {
            'text': 'Total Cost: ₹5,25,000',
            'normalized_text': 'total cost 525000',
            'bbox': [50, 250, 300, 270],
            'confidence': 0.91,
            'numbers': [525000],
            'contains_keywords': {'cost': True},
            'language': 'en'
        }
    ]
    
    print("Optimized Rule-Based Extractor")
    print("=" * 60)
    
    # Benchmark
    import time
    start = time.time()
    
    results = extractor.extract_fields(test_ocr_results)
    
    elapsed = (time.time() - start) * 1000
    
    print(f"\nExtraction time: {elapsed:.2f}ms")
    print("\nExtracted Fields:")
    print("-" * 60)
    
    for field, data in results.items():
        if field not in ['overall_confidence', 'signature', 'stamp']:
            print(f"{field.replace('_', ' ').title():15}: {data.get('value', 'Not found')}")
            print(f"  Confidence: {data.get('confidence', 0):.2%}")
            print(f"  Source: {data.get('source', 'N/A')}")
    
    print(f"\nOverall Confidence: {results.get('overall_confidence', 0):.2%}")
    
    # Test validation
    validated = extractor.validate_extraction(results)
    print(f"Validated Confidence: {validated.get('overall_confidence', 0):.2%}")