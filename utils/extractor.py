import re
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from thefuzz import fuzz, process
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleBasedExtractor:
    """
    Rule-based extractor for invoice fields using heuristics and patterns
    """
    
    def __init__(self, master_data_path: Optional[str] = None):
        """
        Initialize rule-based extractor
        
        Args:
            master_data_path: Path to master data JSON file
        """
        # Load master data for fuzzy matching
        self.dealer_master = []
        self.model_master = []
        
        if master_data_path:
            self.load_master_data(master_data_path)
        
        # Compile patterns for efficiency
        self.compile_patterns()
        
        # Field weights for confidence calculation
        self.field_weights = {
            'dealer_name': 0.20,
            'model_name': 0.25,
            'horse_power': 0.20,
            'asset_cost': 0.25,
            'signature': 0.05,
            'stamp': 0.05
        }
    
    def compile_patterns(self):
        """Compile all regex patterns"""
        # Dealer patterns
        self.dealer_patterns = [
            re.compile(r'(?:dealer|seller|vendor|supplier|distributor)[\s:]+([A-Za-z\s&.,]+)', re.IGNORECASE),
            re.compile(r'([A-Za-z\s&.,]+)\s*(?:dealer|seller|vendor)', re.IGNORECASE),
            re.compile(r'^(?:M/s|M\.s\.|M\/s)[\s:]+([A-Za-z\s&.,]+)', re.IGNORECASE),
            re.compile(r'डीलर[\s:]+([^\n]+)', re.IGNORECASE),
            re.compile(r'विक्रेता[\s:]+([^\n]+)', re.IGNORECASE),
            re.compile(r'ડીલર[\s:]+([^\n]+)', re.IGNORECASE)
        ]
        
        # Model patterns
        self.model_patterns = [
            re.compile(r'(?:model|type|variant)[\s:]+([A-Za-z0-9\s\-]+)', re.IGNORECASE),
            re.compile(r'([0-9]{3,4}\s*(?:DI|XP|XT|MX|TA|FE|GT|R|S|E|PRO|PLUS))', re.IGNORECASE),
            re.compile(r'मॉडल[\s:]+([^\n]+)', re.IGNORECASE),
            re.compile(r'મોડેલ[\s:]+([^\n]+)', re.IGNORECASE),
            re.compile(r'tractor[\s:]+([A-Za-z0-9\s\-]+)', re.IGNORECASE)
        ]
        
        # Horse power patterns
        self.hp_patterns = [
            re.compile(r'(?:horse\s*power|hp|h\.p\.)[\s:]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:hp|h\.p\.|horse\s*power)', re.IGNORECASE),
            re.compile(r'हॉर्स पावर[\s:]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            re.compile(r'શક્તિ[\s:]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:cc|kw)', re.IGNORECASE)
        ]
        
        # Cost patterns
        self.cost_patterns = [
            re.compile(r'(?:total\s*cost|asset\s*cost|price|amount|value)[\s:]*[₹$Rs]?\s*([0-9,]+(?:\.[0-9]+)?)', re.IGNORECASE),
            re.compile(r'[₹$Rs]\s*([0-9,]+(?:\.[0-9]+)?)', re.IGNORECASE),
            re.compile(r'([0-9,]+(?:\.[0-9]+)?)\s*(?:inr|rs|rupees)', re.IGNORECASE),
            re.compile(r'कुल[\s:]*[₹रु]?\s*([0-9,]+)', re.IGNORECASE),
            re.compile(r'કુલ[\s:]*[₹રૂ]?\s*([0-9,]+)', re.IGNORECASE),
            re.compile(r'(\d[\d,]*\.?\d*)\s*lakh', re.IGNORECASE),
            re.compile(r'(\d[\d,]*\.?\d*)\s*crore', re.IGNORECASE)
        ]
        
        # Section headers (for context)
        self.section_patterns = {
            'dealer': re.compile(r'(?:dealer|seller|vendor)', re.IGNORECASE),
            'model': re.compile(r'(?:model|tractor\s*details)', re.IGNORECASE),
            'specifications': re.compile(r'(?:specifications|specs|technical)', re.IGNORECASE),
            'financial': re.compile(r'(?:financial|cost|price|amount)', re.IGNORECASE),
            'signature': re.compile(r'(?:signature|authorized|approved)', re.IGNORECASE),
            'stamp': re.compile(r'(?:stamp|seal|rubber\s*stamp)', re.IGNORECASE)
        }
    
    def load_master_data(self, path: str):
        """Load master data for fuzzy matching"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                master_data = json.load(f)
            
            self.dealer_master = master_data.get('dealers', [])
            self.model_master = master_data.get('models', [])
            
            logger.info(f"Loaded {len(self.dealer_master)} dealers and {len(self.model_master)} models")
            
        except Exception as e:
            logger.error(f"Error loading master data: {e}")
    
    def extract_fields(self, normalized_ocr_results: List[Dict[str, Any]], 
                      image_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Extract all fields using rule-based methods
        
        Args:
            normalized_ocr_results: Normalized OCR results
            image_shape: Image dimensions (height, width) for context
            
        Returns:
            Dictionary with extracted fields and confidence scores
        """
        results = {
            'dealer_name': {'value': None, 'confidence': 0.0, 'source': None},
            'model_name': {'value': None, 'confidence': 0.0, 'source': None},
            'horse_power': {'value': None, 'confidence': 0.0, 'source': None},
            'asset_cost': {'value': None, 'confidence': 0.0, 'source': None},
            'signature': {'present': False, 'bbox': None, 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': None, 'confidence': 0.0}
        }
        
        try:
            # Create text blocks for analysis
            text_blocks = []
            for result in normalized_ocr_results:
                text_blocks.append({
                    'text': result.get('normalized_text', result.get('text', '')),
                    'original_text': result.get('text', ''),
                    'bbox': result.get('bbox'),
                    'confidence': result.get('confidence', 0.0),
                    'numbers': result.get('numbers', []),
                    'keywords': result.get('contains_keywords', {}),
                    'language': result.get('language', 'unknown')
                })
            
            # Extract each field using multiple strategies
            dealer_result = self.extract_dealer_name(text_blocks)
            model_result = self.extract_model_name(text_blocks)
            hp_result = self.extract_horse_power(text_blocks)
            cost_result = self.extract_asset_cost(text_blocks)
            
            # Update results
            if dealer_result:
                results['dealer_name'] = dealer_result
            
            if model_result:
                results['model_name'] = model_result
            
            if hp_result:
                results['horse_power'] = hp_result
            
            if cost_result:
                results['asset_cost'] = cost_result
            
            # Calculate overall confidence
            results['overall_confidence'] = self.calculate_overall_confidence(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in field extraction: {e}")
            return results
    
    def extract_dealer_name(self, text_blocks: List[Dict]) -> Optional[Dict]:
        """
        Extract dealer name using multiple strategies
        
        Args:
            text_blocks: List of text blocks
            
        Returns:
            Dictionary with dealer info or None
        """
        candidates = []
        
        # Strategy 1: Pattern matching
        for block in text_blocks:
            text = block['text']
            original = block['original_text']
            
            for pattern in self.dealer_patterns:
                match = pattern.search(original)
                if match:
                    candidate = match.group(1).strip()
                    confidence = min(block['confidence'] * 0.9, 0.9)
                    candidates.append({
                        'value': candidate,
                        'confidence': confidence,
                        'source': 'pattern',
                        'context': text
                    })
        
        # Strategy 2: Keyword context
        for block in text_blocks:
            if block['keywords'].get('dealer', False):
                # Look for capitalized words in original text
                words = block['original_text'].split()
                dealer_words = []
                
                for word in words:
                    if word and word[0].isupper() and len(word) > 2:
                        dealer_words.append(word)
                
                if dealer_words:
                    candidate = ' '.join(dealer_words[:3])  # Take first 3 capitalized words
                    confidence = block['confidence'] * 0.8
                    candidates.append({
                        'value': candidate,
                        'confidence': confidence,
                        'source': 'keyword',
                        'context': block['text']
                    })
        
        # Strategy 3: Position-based (often at top of document)
        if not candidates and text_blocks:
            # Take first significant text block
            first_blocks = sorted(text_blocks, key=lambda x: x['bbox'][1])[:3]
            for block in first_blocks:
                text = block['original_text'].strip()
                if len(text) > 10 and not any(word in text.lower() for word in ['invoice', 'quotation', 'bill']):
                    confidence = block['confidence'] * 0.7
                    candidates.append({
                        'value': text,
                        'confidence': confidence,
                        'source': 'position',
                        'context': text
                    })
        
        # Fuzzy match with master data if available
        if candidates and self.dealer_master:
            best_candidate = max(candidates, key=lambda x: x['confidence'])
            matched_dealer, match_score = self.fuzzy_match_dealer(best_candidate['value'])
            
            if match_score > 80:  # Good match threshold
                best_candidate['value'] = matched_dealer
                best_candidate['confidence'] = min(best_candidate['confidence'] * (match_score / 100), 0.95)
                best_candidate['source'] = 'fuzzy_match'
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            return {
                'value': best['value'],
                'confidence': best['confidence'],
                'source': best['source']
            }
        
        return None
    
    def extract_model_name(self, text_blocks: List[Dict]) -> Optional[Dict]:
        """
        Extract model name
        
        Args:
            text_blocks: List of text blocks
            
        Returns:
            Dictionary with model info or None
        """
        candidates = []
        
        # Strategy 1: Pattern matching
        for block in text_blocks:
            text = block['text']
            original = block['original_text']
            
            for pattern in self.model_patterns:
                match = pattern.search(original)
                if match:
                    candidate = match.group(1).strip().upper()
                    confidence = min(block['confidence'] * 0.95, 0.95)
                    candidates.append({
                        'value': candidate,
                        'confidence': confidence,
                        'source': 'pattern',
                        'context': text
                    })
        
        # Strategy 2: Look for model numbers
        for block in text_blocks:
            # Extract potential model numbers (3-4 digits)
            text = block['original_text']
            model_matches = re.findall(r'\b(\d{3,4}\s*(?:DI|XP|XT|MX)?)\b', text, re.IGNORECASE)
            
            for match in model_matches:
                candidate = match.strip().upper()
                confidence = block['confidence'] * 0.9
                candidates.append({
                    'value': candidate,
                    'confidence': confidence,
                    'source': 'model_number',
                    'context': text
                })
        
        # Strategy 3: Brand + model combination
        common_brands = ['MAHINDRA', 'SWARAJ', 'ESCORTS', 'EICHER', 'JOHN DEERE', 'CASE', 'NEW HOLLAND']
        for block in text_blocks:
            text = block['original_text'].upper()
            for brand in common_brands:
                if brand in text:
                    # Extract text after brand as potential model
                    parts = text.split(brand, 1)
                    if len(parts) > 1:
                        after_brand = parts[1].strip()
                        # Take first word after brand
                        model_candidate = after_brand.split()[0] if after_brand.split() else ""
                        if model_candidate and len(model_candidate) >= 2:
                            candidate = f"{brand} {model_candidate}"
                            confidence = block['confidence'] * 0.85
                            candidates.append({
                                'value': candidate,
                                'confidence': confidence,
                                'source': 'brand_context',
                                'context': text
                            })
        
        # Fuzzy match with master data
        if candidates and self.model_master:
            best_candidate = max(candidates, key=lambda x: x['confidence'])
            matched_model, match_score = self.fuzzy_match_model(best_candidate['value'])
            
            if match_score > 90:  # Higher threshold for exact match
                best_candidate['value'] = matched_model
                best_candidate['confidence'] = min(best_candidate['confidence'] * (match_score / 100), 0.98)
                best_candidate['source'] = 'fuzzy_match'
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            return {
                'value': best['value'],
                'confidence': best['confidence'],
                'source': best['source']
            }
        
        return None
    
    def extract_horse_power(self, text_blocks: List[Dict]) -> Optional[Dict]:
        """
        Extract horse power
        
        Args:
            text_blocks: List of text blocks
            
        Returns:
            Dictionary with HP info or None
        """
        candidates = []
        
        # Strategy 1: Pattern matching
        for block in text_blocks:
            text = block['text']
            original = block['original_text']
            
            for pattern in self.hp_patterns:
                match = pattern.search(original)
                if match:
                    try:
                        hp_value = float(match.group(1).replace(',', ''))
                        confidence = min(block['confidence'] * 0.95, 0.95)
                        candidates.append({
                            'value': hp_value,
                            'confidence': confidence,
                            'source': 'pattern',
                            'context': text
                        })
                    except ValueError:
                        continue
        
        # Strategy 2: Look in specification section
        spec_blocks = []
        for i, block in enumerate(text_blocks):
            text = block['text'].lower()
            if any(keyword in text for keyword in ['specification', 'technical', 'engine', 'power']):
                # Include this block and next few blocks
                spec_blocks.extend(text_blocks[i:i+3])
                break
        
        for block in spec_blocks:
            # Look for numbers that could be HP
            for num in block.get('numbers', []):
                if 10 <= num <= 200:  # Reasonable HP range for tractors
                    confidence = block['confidence'] * 0.8
                    candidates.append({
                        'value': num,
                        'confidence': confidence,
                        'source': 'spec_section',
                        'context': block['text']
                    })
        
        # Strategy 3: Look near model name
        model_blocks = [b for b in text_blocks if b['keywords'].get('model', False)]
        for block in model_blocks:
            # Extract numbers from model context
            for num in block.get('numbers', []):
                if 10 <= num <= 200:
                    confidence = block['confidence'] * 0.7
                    candidates.append({
                        'value': num,
                        'confidence': confidence,
                        'source': 'model_context',
                        'context': block['text']
                    })
        
        # Return best candidate (highest confidence)
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            return {
                'value': best['value'],
                'confidence': best['confidence'],
                'source': best['source']
            }
        
        return None
    
    def extract_asset_cost(self, text_blocks: List[Dict]) -> Optional[Dict]:
        """
        Extract asset cost
        
        Args:
            text_blocks: List of text blocks
            
        Returns:
            Dictionary with cost info or None
        """
        candidates = []
        
        # Strategy 1: Pattern matching
        for block in text_blocks:
            text = block['text']
            original = block['original_text']
            
            for pattern in self.cost_patterns:
                match = pattern.search(original)
                if match:
                    try:
                        # Clean and convert cost
                        cost_str = match.group(1).replace(',', '')
                        cost_value = float(cost_str)
                        
                        # Check if it's lakhs/crores
                        if 'lakh' in original.lower():
                            cost_value *= 100000
                        elif 'crore' in original.lower():
                            cost_value *= 10000000
                        
                        confidence = min(block['confidence'] * 0.9, 0.9)
                        candidates.append({
                            'value': cost_value,
                            'confidence': confidence,
                            'source': 'pattern',
                            'context': text
                        })
                    except ValueError:
                        continue
        
        # Strategy 2: Look in financial section
        financial_blocks = []
        for i, block in enumerate(text_blocks):
            text = block['text'].lower()
            if any(keyword in text for keyword in ['total', 'amount', 'cost', 'price', 'payment']):
                financial_blocks.extend(text_blocks[i:i+5])
                break
        
        # Find largest number in financial section
        max_cost = 0
        max_conf = 0
        for block in financial_blocks:
            numbers = block.get('numbers', [])
            if numbers:
                block_max = max(numbers)
                if 10000 <= block_max <= 10000000:  # Reasonable tractor cost range
                    if block_max > max_cost:
                        max_cost = block_max
                        max_conf = block['confidence'] * 0.8
        
        if max_cost > 0:
            candidates.append({
                'value': max_cost,
                'confidence': max_conf,
                'source': 'financial_section',
                'context': 'Largest number in financial section'
            })
        
        # Strategy 3: Look for numbers with currency symbols
        for block in text_blocks:
            if '₹' in block['original_text'] or 'Rs' in block['original_text'] or '$' in block['original_text']:
                numbers = block.get('numbers', [])
                for num in numbers:
                    if 10000 <= num <= 10000000:
                        confidence = block['confidence'] * 0.75
                        candidates.append({
                            'value': num,
                            'confidence': confidence,
                            'source': 'currency_symbol',
                            'context': block['text']
                        })
        
        # Return best candidate
        if candidates:
            # Prefer pattern matches
            pattern_candidates = [c for c in candidates if c['source'] == 'pattern']
            if pattern_candidates:
                best = max(pattern_candidates, key=lambda x: x['confidence'])
            else:
                best = max(candidates, key=lambda x: x['confidence'])
            
            return {
                'value': best['value'],
                'confidence': best['confidence'],
                'source': best['source']
            }
        
        return None
    
    def fuzzy_match_dealer(self, candidate: str) -> Tuple[str, float]:
        """
        Fuzzy match dealer name with master list
        
        Args:
            candidate: Candidate dealer name
            
        Returns:
            Tuple of (matched_name, match_score)
        """
        if not self.dealer_master:
            return candidate, 0.0
        
        best_match, best_score = process.extractOne(
            candidate,
            self.dealer_master,
            scorer=fuzz.token_sort_ratio
        )
        
        return best_match, best_score
    
    def fuzzy_match_model(self, candidate: str) -> Tuple[str, float]:
        """
        Fuzzy match model name with master list
        
        Args:
            candidate: Candidate model name
            
        Returns:
            Tuple of (matched_name, match_score)
        """
        if not self.model_master:
            return candidate, 0.0
        
        best_match, best_score = process.extractOne(
            candidate,
            self.model_master,
            scorer=fuzz.token_set_ratio
        )
        
        return best_match, best_score
    
    def calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate weighted overall confidence
        
        Args:
            results: Field extraction results
            
        Returns:
            Overall confidence score
        """
        total_weight = 0
        weighted_sum = 0
        
        for field, weight in self.field_weights.items():
            if field in ['signature', 'stamp']:
                # For binary fields, use their confidence
                if results[field]['present']:
                    weighted_sum += results[field]['confidence'] * weight
                    total_weight += weight
            else:
                # For text/numeric fields
                if results[field]['value'] is not None:
                    weighted_sum += results[field]['confidence'] * weight
                    total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def validate_extraction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted results for consistency
        
        Args:
            results: Extracted results
            
        Returns:
            Validated results with corrections
        """
        validated = results.copy()
        
        # Validate horse power range
        hp = validated.get('horse_power', {}).get('value')
        if hp is not None:
            if hp < 10 or hp > 200:
                validated['horse_power']['confidence'] *= 0.5  # Reduce confidence
                logger.warning(f"Horse power {hp} outside typical range (10-200)")
        
        # Validate asset cost range
        cost = validated.get('asset_cost', {}).get('value')
        if cost is not None:
            if cost < 100000 or cost > 5000000:
                validated['asset_cost']['confidence'] *= 0.6
                logger.warning(f"Asset cost {cost} outside typical range (1L-50L)")
        
        # Cross-validate: If model is extracted, check if HP makes sense
        model = validated.get('model_name', {}).get('value')
        if model and hp:
            # Simple model-HP mapping (can be extended)
            model_hp_map = {
                '475': 45, '485': 48, '575': 50, '595': 55,
                'DI': 50, 'XP': 55, 'XT': 60, 'MX': 65
            }
            
            for key, expected_hp in model_hp_map.items():
                if key in str(model):
                    if abs(hp - expected_hp) > 10:
                        validated['horse_power']['confidence'] *= 0.7
                        logger.warning(f"HP {hp} doesn't match model {model} (expected ~{expected_hp})")
                    break
        
        # Recalculate overall confidence
        validated['overall_confidence'] = self.calculate_overall_confidence(validated)
        
        return validated

# Singleton instance
_extractor = None

def get_extractor(master_data_path: Optional[str] = None) -> RuleBasedExtractor:
    """
    Get or create RuleBasedExtractor instance
    
    Args:
        master_data_path: Path to master data
        
    Returns:
        RuleBasedExtractor instance
    """
    global _extractor
    if _extractor is None:
        _extractor = RuleBasedExtractor(master_data_path)
    return _extractor

if __name__ == "__main__":
    # Test the extractor
    extractor = RuleBasedExtractor()
    
    # Sample normalized OCR results
    test_ocr_results = [
        {
            'text': 'Dealer: Mahindra Tractors Pvt Ltd',
            'normalized_text': 'dealer mahindra tractors pvt ltd',
            'bbox': [50, 100, 300, 120],
            'confidence': 0.95,
            'numbers': [],
            'contains_keywords': {'dealer': True, 'model': False, 'horse_power': False, 'cost': False, 'invoice': False},
            'language': 'en'
        },
        {
            'text': 'Model: 575 DI',
            'normalized_text': 'model 575 di',
            'bbox': [50, 150, 200, 170],
            'confidence': 0.92,
            'numbers': [575],
            'contains_keywords': {'dealer': False, 'model': True, 'horse_power': False, 'cost': False, 'invoice': False},
            'language': 'en'
        },
        {
            'text': 'Horse Power: 50 HP',
            'normalized_text': 'horse power 50 hp',
            'bbox': [50, 200, 250, 220],
            'confidence': 0.94,
            'numbers': [50],
            'contains_keywords': {'dealer': False, 'model': False, 'horse_power': True, 'cost': False, 'invoice': False},
            'language': 'en'
        },
        {
            'text': 'Total Cost: ₹5,25,000',
            'normalized_text': 'total cost 525000',
            'bbox': [50, 250, 300, 270],
            'confidence': 0.91,
            'numbers': [525000],
            'contains_keywords': {'dealer': False, 'model': False, 'horse_power': False, 'cost': True, 'invoice': False},
            'language': 'en'
        }
    ]
    
    print("Testing Rule-Based Extractor")
    print("=" * 60)
    
    results = extractor.extract_fields(test_ocr_results)
    
    print("\nExtracted Fields:")
    print("-" * 60)
    
    for field, data in results.items():
        if field not in ['overall_confidence', 'signature', 'stamp']:
            print(f"{field.replace('_', ' ').title():15}: {data.get('value', 'Not found')}")
            print(f"  Confidence: {data.get('confidence', 0):.2%}")
            print(f"  Source: {data.get('source', 'N/A')}")
            print()
    
    print(f"\nOverall Confidence: {results.get('overall_confidence', 0):.2%}")
    
    # Test validation
    print("\n" + "=" * 60)
    print("Testing Validation")
    print("=" * 60)
    
    validated = extractor.validate_extraction(results)
    print(f"Validated Confidence: {validated.get('overall_confidence', 0):.2%}")