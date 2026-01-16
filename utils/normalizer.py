import re
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextNormalizer:
    """
    Highly optimized text normalizer with compiled patterns and caching
    """
    
    # Class-level constants (frozen for immutability)
    CURRENCY_SYMBOLS = frozenset([
        '₹', 'Rs', 'Rs.', 'INR', 'रु', 'रुपए', 'રૂ', 'રૂપિયા',
        '$', 'USD', '€', 'EUR', '£', 'GBP', '¥', 'JPY'
    ])
    
    # Unicode ranges for fast character detection
    DEVANAGARI_RANGE = range(0x0900, 0x0980)
    GUJARATI_RANGE = range(0x0A80, 0x0B00)
    
    # Keywords as frozensets for O(1) lookup
    DEALER_KEYWORDS = frozenset([
        'dealer', 'seller', 'vendor', 'supplier', 'distributor',
        'डीलर', 'विक्रेता', 'वेंडर', 'ડીલર', 'વેચનાર'
    ])
    
    MODEL_KEYWORDS = frozenset([
        'model', 'type', 'variant', 'version',
        'मॉडल', 'प्रकार', 'મોડેલ', 'પ્રકાર'
    ])
    
    HP_KEYWORDS = frozenset([
        'horse power', 'hp', 'h.p.', 'power',
        'हॉर्स पावर', 'एचपी', 'શક્તિ'
    ])
    
    COST_KEYWORDS = frozenset([
        'cost', 'price', 'amount', 'total', 'value', 'asset cost',
        'लागत', 'मूल्य', 'राशि', 'कुल', 'ખર્ચ', 'કિંમત', 'રકમ'
    ])
    
    INVOICE_KEYWORDS = frozenset([
        'invoice', 'bill', 'quotation', 'quote', 'estimate'
    ])
    
    def __init__(self):
        """Initialize with pre-compiled patterns"""
        self._compile_patterns()
        self._setup_translation_tables()
    
    def _compile_patterns(self):
        """Compile all regex patterns once"""
        # Single comprehensive currency pattern
        currency_escaped = '|'.join(re.escape(s) for s in self.CURRENCY_SYMBOLS)
        self.currency_pattern = re.compile(f'({currency_escaped})[\\s:]*', re.I)
        
        # Optimized number pattern with optional decimal
        self.number_pattern = re.compile(r'\d+(?:,\d+)*(?:\.\d+)?')
        
        # HP pattern - combined multiple languages
        self.hp_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(?:HP|H\.P\.|Horse\s*Power|हॉर्स\s*पावर|હોર્સ\s*પાવર)',
            re.I
        )
        
        # Model pattern - expanded for common formats
        self.model_pattern = re.compile(
            r'\b(\d{3,4}\s*(?:DI|XP|XT|MX|TA|FE|GT|R|S|E|PRO|PLUS)?)\b',
            re.I
        )
        
        # Combined cleanup patterns
        self.email_pattern = re.compile(r'\b[\w.%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(?:\+91[\-\s]?)?[789]\d{9}')
        self.date_pattern = re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b')
        
        # Single multi-space pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Special chars - keep alphanumeric, spaces, and basic punctuation
        self.special_pattern = re.compile(r'[^\w\s.,:;/()\-]', re.U)
    
    def _setup_translation_tables(self):
        """Setup translation tables for fast character replacement"""
        # Create translation table for removing unwanted chars
        # This is faster than regex for simple character removal
        self.pii_replacements = str.maketrans({
            '@': '[AT]',
            '+': '[PLUS]'
        })
    
    def normalize_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Optimized text normalization with minimal string operations
        """
        if not text:
            return ""
        
        try:
            # Fast path for already normalized text
            if text.isascii() and text.islower() and ' ' not in '  '.join(text.split()):
                return text
            
            # Single pass Unicode normalization
            text = unicodedata.normalize('NFKC', str(text))
            
            # Batch PII removal (faster than multiple regex operations)
            text = self.email_pattern.sub('[EMAIL]', text)
            text = self.phone_pattern.sub('[PHONE]', text)
            text = self.date_pattern.sub('[DATE]', text)
            
            # Remove currency symbols
            text = self.currency_pattern.sub('', text)
            
            # Remove special characters
            text = self.special_pattern.sub(' ', text)
            
            # Collapse whitespace
            text = self.whitespace_pattern.sub(' ', text).strip()
            
            # Case conversion (after all other operations)
            return text if preserve_case else text.lower()
            
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            return text
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        Vectorized number extraction with efficient parsing
        """
        try:
            # Find all matches at once
            matches = self.number_pattern.findall(text)
            
            if not matches:
                return []
            
            # Batch process with list comprehension
            numbers = []
            for match in matches:
                try:
                    # Remove commas and convert in one step
                    num = float(match.replace(',', ''))
                    # Filter unreasonable values
                    if 0 < num < 1e12:  # Reasonable range
                        numbers.append(num)
                except ValueError:
                    continue
            
            return numbers
            
        except Exception as e:
            logger.error(f"Number extraction error: {e}")
            return []
    
    @lru_cache(maxsize=512)
    def extract_horse_power(self, text: str) -> Optional[float]:
        """
        Cached HP extraction with pattern matching
        """
        try:
            # Primary pattern match
            match = self.hp_pattern.search(text)
            if match:
                hp = float(match.group(1))
                # Validate range (10-200 HP typical for tractors)
                return hp if 10 <= hp <= 200 else None
            
            # Fallback: contextual search
            words = text.split()
            hp_indicators = {'hp', 'h.p.', 'horse', 'power', 'हॉर्स', 'પાવર'}
            
            for i, word in enumerate(words):
                if word.lower() in hp_indicators and i > 0:
                    try:
                        hp = float(words[i-1].replace(',', ''))
                        return hp if 10 <= hp <= 200 else None
                    except ValueError:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"HP extraction error: {e}")
            return None
    
    @lru_cache(maxsize=512)
    def extract_model_name(self, text: str) -> Optional[str]:
        """
        Cached model extraction
        """
        try:
            # Pattern-based extraction
            match = self.model_pattern.search(text)
            if match:
                return match.group(1).upper()
            
            # Keyword-based extraction
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in self.MODEL_KEYWORDS and i + 1 < len(words):
                    candidate = words[i + 1]
                    if candidate and len(candidate) >= 2:
                        return candidate.upper()
            
            return None
            
        except Exception as e:
            logger.error(f"Model extraction error: {e}")
            return None
    
    @lru_cache(maxsize=256)
    def detect_language(self, text: str) -> str:
        """
        Fast character-based language detection with caching
        """
        try:
            # Quick checks using character ranges
            has_devanagari = any(ord(c) in self.DEVANAGARI_RANGE for c in text[:100])
            if has_devanagari:
                return 'hi'
            
            has_gujarati = any(ord(c) in self.GUJARATI_RANGE for c in text[:100])
            if has_gujarati:
                return 'gu'
            
            # Check for ASCII/English
            if any(c.isalpha() and c.isascii() for c in text[:100]):
                return 'en'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'unknown'
    
    def normalize_ocr_results(self, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch normalize OCR results with minimal allocations
        """
        if not ocr_results:
            return []
        
        normalized_results = []
        
        for result in ocr_results:
            original_text = result.get('text', '')
            if not original_text:
                continue
            
            # Pre-allocate result dictionary
            normalized = {
                'text': original_text,
                'bbox': result.get('bbox', [0, 0, 0, 0]),
                'confidence': result.get('confidence', 0.0),
                'normalized_text': self.normalize_text(original_text),
                'numbers': self.extract_numbers(original_text),
                'horse_power': self.extract_horse_power(original_text),
                'model_name': self.extract_model_name(original_text),
                'language': self.detect_language(original_text)
            }
            
            # Keyword detection (fast set operations)
            normalized['contains_keywords'] = self._detect_keywords_fast(
                normalized['normalized_text']
            )
            
            normalized_results.append(normalized)
        
        return normalized_results
    
    def _detect_keywords_fast(self, text: str) -> Dict[str, bool]:
        """
        Fast keyword detection using set operations
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Use set intersection for O(1) average case
        return {
            'dealer': bool(words & self.DEALER_KEYWORDS),
            'model': bool(words & self.MODEL_KEYWORDS),
            'horse_power': bool(words & self.HP_KEYWORDS),
            'cost': bool(words & self.COST_KEYWORDS),
            'invoice': bool(words & self.INVOICE_KEYWORDS)
        }
    
    def group_by_region(self, ocr_results: List[Dict[str, Any]], 
                        vertical_threshold: int = 50) -> List[List[Dict[str, Any]]]:
        """
        Optimized region grouping with single-pass algorithm
        """
        if not ocr_results:
            return []
        
        # Single sort operation
        sorted_results = sorted(ocr_results, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        groups = []
        current_group = [sorted_results[0]]
        current_y = sorted_results[0]['bbox'][1]
        
        for result in sorted_results[1:]:
            y_pos = result['bbox'][1]
            
            if abs(y_pos - current_y) <= vertical_threshold:
                current_group.append(result)
            else:
                # No need to re-sort, already sorted by x
                groups.append(current_group)
                current_group = [result]
                current_y = y_pos
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def create_text_sequence(self, ocr_results: List[Dict[str, Any]]) -> str:
        """
        Optimized text sequence creation with pre-allocation
        """
        if not ocr_results:
            return ""
        
        groups = self.group_by_region(ocr_results)
        
        # Pre-allocate list with estimated size
        lines = []
        
        for group in groups:
            # Use generator expression with join for efficiency
            line_text = ' '.join(
                r.get('normalized_text') or r.get('text', '')
                for r in group
                if r.get('normalized_text') or r.get('text')
            )
            
            if line_text:
                lines.append(line_text)
        
        return '\n'.join(lines)
    
    def batch_normalize(self, texts: List[str], preserve_case: bool = False) -> List[str]:
        """
        Batch normalize multiple texts efficiently
        """
        return [self.normalize_text(text, preserve_case) for text in texts]
    
    def clear_cache(self):
        """Clear LRU caches to free memory"""
        self.extract_horse_power.cache_clear()
        self.extract_model_name.cache_clear()
        self.detect_language.cache_clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            'hp_cache': self.extract_horse_power.cache_info()._asdict(),
            'model_cache': self.extract_model_name.cache_info()._asdict(),
            'lang_cache': self.detect_language.cache_info()._asdict()
        }


# Singleton with lazy initialization
_normalizer = None

def get_normalizer() -> TextNormalizer:
    """Get or create normalizer instance"""
    global _normalizer
    if _normalizer is None:
        _normalizer = TextNormalizer()
    return _normalizer


if __name__ == "__main__":
    import time
    
    # Initialize normalizer
    normalizer = TextNormalizer()
    
    test_cases = [
        "Dealer: Mahindra Tractors, Price: ₹5,25,000",
        "Model: 575 DI, Horse Power: 50 HP",
        "Total Cost: Rs. 4,50,000.50",
        "विक्रेता: ABC ट्रैक्टर्स, मूल्य: रु 3,00,000",
        "ડીલર: XYZ ટ્રેક્ટર્સ, કિંમત: રૂ 2,50,000",
        "Contact: dealer@email.com, Phone: +91 9876543210"
    ]
    
    print("Optimized Text Normalizer Benchmark")
    print("=" * 60)
    
    # Warmup
    for _ in range(2):
        for text in test_cases:
            _ = normalizer.normalize_text(text)
    
    # Benchmark
    iterations = 1000
    start = time.time()
    
    for _ in range(iterations):
        for text in test_cases:
            normalized = normalizer.normalize_text(text)
            numbers = normalizer.extract_numbers(text)
            hp = normalizer.extract_horse_power(text)
            model = normalizer.extract_model_name(text)
            lang = normalizer.detect_language(text)
    
    elapsed = (time.time() - start) * 1000
    avg_per_text = elapsed / (iterations * len(test_cases))
    
    print(f"\nBenchmark Results:")
    print(f"Total time: {elapsed:.2f}ms for {iterations * len(test_cases)} operations")
    print(f"Average per text: {avg_per_text:.3f}ms")
    print(f"Throughput: {1000/avg_per_text:.0f} texts/second")
    
    # Cache stats
    print("\nCache Statistics:")
    stats = normalizer.get_cache_stats()
    for cache_name, info in stats.items():
        print(f"{cache_name}: hits={info['hits']}, misses={info['misses']}, "
              f"hit_rate={info['hits']/(info['hits']+info['misses'])*100:.1f}%")
    
    # Test individual cases
    print("\n" + "=" * 60)
    print("Individual Test Cases:")
    print("=" * 60)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Original: {test_text}")
        
        start = time.time()
        normalized = normalizer.normalize_text(test_text)
        elapsed = (time.time() - start) * 1000
        
        print(f"Normalized: {normalized} ({elapsed:.3f}ms)")
        print(f"Numbers: {normalizer.extract_numbers(test_text)}")
        print(f"Horse Power: {normalizer.extract_horse_power(test_text)}")
        print(f"Model: {normalizer.extract_model_name(test_text)}")
        print(f"Language: {normalizer.detect_language(test_text)}")
    
    # Test batch processing
    print("\n" + "=" * 60)
    print("Batch Processing Test:")
    print("=" * 60)
    
    ocr_results = [
        {
            'text': text,
            'bbox': [0, i*50, 500, (i+1)*50],
            'confidence': 0.9
        }
        for i, text in enumerate(test_cases)
    ]
    
    start = time.time()
    normalized_batch = normalizer.normalize_ocr_results(ocr_results)
    elapsed = (time.time() - start) * 1000
    
    print(f"Batch normalized {len(ocr_results)} results in {elapsed:.2f}ms")
    print(f"Average: {elapsed/len(ocr_results):.3f}ms per result")