import re
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
import logging
from functools import lru_cache
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NormalizedText:
    """Structured normalization result"""
    original: str
    normalized: str
    numbers: List[float]
    language: str
    has_currency: bool
    keywords: Dict[str, bool]


class TextNormalizer:
    """
    High-performance text normalizer for multi-language invoice/quotation OCR.
    Supports English, Hindi, and Gujarati with PII removal and field extraction.
    """
    
    # Currency symbols (Indian + international)
    CURRENCY_SYMBOLS = frozenset([
        '₹', 'Rs', 'Rs.', 'INR', 'रु', 'रु.', 'रुपए', 'રૂ', 'રૂ.', 'રૂપિયા',
        '$', 'USD', '€', 'EUR', '£', 'GBP', '¥', 'JPY', 'र', 'रू'
    ])
    
    # Unicode ranges for language detection
    DEVANAGARI_RANGE = range(0x0900, 0x0980)
    GUJARATI_RANGE = range(0x0A80, 0x0B00)
    LATIN_RANGE = range(0x0041, 0x007B)
    
    # Keyword sets for O(1) lookup
    DEALER_KEYWORDS = frozenset([
        'dealer', 'seller', 'vendor', 'supplier', 'distributor', 'authorized',
        'डीलर', 'विक्रेता', 'वेंडर', 'ડીલર', 'વેચનાર', 'વિતરક'
    ])
    
    MODEL_KEYWORDS = frozenset([
        'model', 'type', 'variant', 'version', 'tractor',
        'मॉडल', 'प्रकार', 'किस्म', 'મોડેલ', 'પ્રકાર', 'વિવિધતા'
    ])
    
    HP_KEYWORDS = frozenset([
        'horse power', 'hp', 'h.p.', 'h.p', 'power', 'bhp',
        'हॉर्स पावर', 'एचपी', 'शक्ति', 'હોર્સ પાવર', 'શક્તિ'
    ])
    
    COST_KEYWORDS = frozenset([
        'cost', 'price', 'amount', 'total', 'value', 'asset cost', 'ex-showroom',
        'लागत', 'मूल्य', 'राशि', 'कुल', 'ખર્ચ', 'કિંમત', 'રકમ', 'કુલ'
    ])
    
    INVOICE_KEYWORDS = frozenset([
        'invoice', 'bill', 'quotation', 'quote', 'estimate', 'proforma',
        'बीजक', 'बिल', 'કોટેશન', 'બિલ'
    ])
    
    # Common OCR errors/replacements
    OCR_CORRECTIONS = {
        'O': '0',  # Letter O -> Zero (in numeric context)
        'l': '1',  # Lowercase L -> One (in numeric context)
        'I': '1',  # Capital I -> One (in numeric context)
        'S': '5',  # Sometimes misread
        'Z': '2',  # Sometimes misread
    }
    
    def __init__(self):
        """Initialize with compiled patterns and translation tables"""
        self._compile_patterns()
        self._setup_translation_tables()
        
        # Performance tracking
        self.stats = {
            'normalized_count': 0,
            'avg_time_ms': 0.0,
            'errors': 0
        }
    
    def _compile_patterns(self):
        """Compile all regex patterns once for performance"""
        
        # Currency pattern (comprehensive)
        currency_escaped = '|'.join(re.escape(s) for s in self.CURRENCY_SYMBOLS)
        self.currency_pattern = re.compile(f'({currency_escaped})[\\s:]*', re.I)
        
        # Number patterns
        self.number_pattern = re.compile(r'\d+(?:,\d+)*(?:\.\d+)?')
        self.indian_number_pattern = re.compile(r'\d+(?:,\d{2})*(?:,\d{3})?(?:\.\d+)?')
        
        # HP patterns (multi-language)
        self.hp_patterns = [
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:HP|H\.P\.?|Horse\s*Power|BHP)', re.I),
            re.compile(r'(?:HP|H\.P\.?|Horse\s*Power)[\s:]*(\d+(?:\.\d+)?)', re.I),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:हॉर्स\s*पावर|એચપી|હોર્સ\s*પાવર)', re.I),
            re.compile(r'(\d+(?:\.\d+)?)\s*kw', re.I)  # kW conversion
        ]
        
        # Model patterns (expanded for Indian market)
        self.model_patterns = [
            re.compile(r'\b([2-9]\d{2,3}\s*(?:DI|XP|XT|MX|TA|FE|GT|R|S|E|PRO|PLUS)?)\b', re.I),
            re.compile(r'(?:model|type)[\s:]+([A-Za-z0-9\s\-/]{2,30})', re.I),
            re.compile(r'\b(MAHINDRA|SWARAJ|JOHN\s+DEERE)\s+([A-Za-z0-9\s\-]{2,20})', re.I)
        ]
        
        # PII patterns
        self.email_pattern = re.compile(r'\b[\w.%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(?:\+91[\-\s]?)?[6789]\d{9}\b')
        self.pan_pattern = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
        self.gst_pattern = re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z]\d\b')
        self.aadhar_pattern = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
        
        # Date patterns (various formats)
        self.date_patterns = [
            re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'),
            re.compile(r'\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b')
        ]
        
        # Whitespace normalization
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_pattern = re.compile(r'[^\w\s.,;:()\-/₹]', re.UNICODE)
        
        # Indian number word patterns
        self.indian_multipliers = {
            'lakh': 100000,
            'lac': 100000,
            'lakhs': 100000,
            'crore': 10000000,
            'cr': 10000000,
            'crores': 10000000
        }
        
        self.multiplier_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|cr|crores)',
            re.I
        )
    
    def _setup_translation_tables(self):
        """Setup character translation tables"""
        # Remove common noise characters
        self.noise_chars = str.maketrans('', '', '|_~`^')
    
    def normalize_text(
        self, 
        text: str, 
        preserve_case: bool = False,
        remove_pii: bool = True
    ) -> str:
        """
        Comprehensive text normalization with PII removal
        
        Args:
            text: Input text
            preserve_case: Keep original case
            remove_pii: Remove personally identifiable information
            
        Returns:
            Normalized text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Fast path for simple text
            if text.isascii() and len(text) < 20 and text.islower():
                return text.strip()
            
            # Unicode normalization (NFKC for compatibility)
            text = unicodedata.normalize('NFKC', text)
            
            # Remove PII if requested
            if remove_pii:
                text = self._remove_pii(text)
            
            # Remove currency symbols (we track these separately)
            text = self.currency_pattern.sub('', text)
            
            # Remove noise characters
            text = text.translate(self.noise_chars)
            
            # Clean special characters (keep important punctuation)
            text = self.special_pattern.sub(' ', text)
            
            # Normalize whitespace
            text = self.whitespace_pattern.sub(' ', text).strip()
            
            # Case normalization
            if not preserve_case:
                text = text.lower()
            
            self.stats['normalized_count'] += 1
            
            return text
            
        except Exception as e:
            logger.error(f"Normalization error for text '{text[:50]}...': {e}")
            self.stats['errors'] += 1
            return text
    
    def _remove_pii(self, text: str) -> str:
        """Remove personally identifiable information"""
        # Email
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Phone numbers
        text = self.phone_pattern.sub('[PHONE]', text)
        
        # PAN card
        text = self.pan_pattern.sub('[PAN]', text)
        
        # GST number
        text = self.gst_pattern.sub('[GST]', text)
        
        # Aadhar
        text = self.aadhar_pattern.sub('[AADHAR]', text)
        
        # Dates
        for pattern in self.date_patterns:
            text = pattern.sub('[DATE]', text)
        
        return text
    
    def extract_numbers(self, text: str, handle_indian: bool = True) -> List[float]:
        """
        Extract numbers including Indian notation (lakhs, crores)
        
        Args:
            text: Input text
            handle_indian: Handle Indian number formats
            
        Returns:
            List of extracted numbers
        """
        try:
            numbers = []
            
            # Handle Indian multipliers first
            if handle_indian:
                for match in self.multiplier_pattern.finditer(text):
                    try:
                        value = float(match.group(1))
                        multiplier = self.indian_multipliers.get(match.group(2).lower(), 1)
                        numbers.append(value * multiplier)
                    except ValueError:
                        continue
            
            # Extract regular numbers
            pattern = self.indian_number_pattern if handle_indian else self.number_pattern
            
            for match in pattern.finditer(text):
                try:
                    # Remove commas and convert
                    num_str = match.group().replace(',', '')
                    num = float(num_str)
                    
                    # Filter unreasonable values
                    if 0 < num < 1e12:
                        numbers.append(num)
                except ValueError:
                    continue
            
            # Remove duplicates (from multiplier processing)
            return list(dict.fromkeys(numbers))  # Preserves order
            
        except Exception as e:
            logger.error(f"Number extraction error: {e}")
            return []
    
    @lru_cache(maxsize=512)
    def extract_horse_power(self, text: str) -> Optional[float]:
        """
        Extract horse power with validation
        
        Args:
            text: Input text
            
        Returns:
            Horse power value or None
        """
        try:
            # Try each HP pattern
            for pattern in self.hp_patterns:
                match = pattern.search(text)
                if match:
                    hp = float(match.group(1))
                    
                    # Convert kW to HP if needed
                    if 'kw' in text.lower():
                        hp *= 1.34
                    
                    # Validate range (10-150 typical for tractors)
                    if 10 <= hp <= 150:
                        return round(hp, 1)
            
            # Contextual fallback
            words = text.lower().split()
            for i, word in enumerate(words):
                if any(kw in word for kw in ['hp', 'power', 'हॉर्स', 'શક્તિ']):
                    # Check adjacent numbers
                    for offset in [-1, 1]:
                        idx = i + offset
                        if 0 <= idx < len(words):
                            try:
                                hp = float(words[idx].replace(',', ''))
                                if 10 <= hp <= 150:
                                    return round(hp, 1)
                            except ValueError:
                                continue
            
            return None
            
        except Exception as e:
            logger.error(f"HP extraction error: {e}")
            return None
    
    @lru_cache(maxsize=512)
    def extract_model_name(self, text: str) -> Optional[str]:
        """
        Extract model name from text
        
        Args:
            text: Input text
            
        Returns:
            Model name or None
        """
        try:
            # Try each model pattern
            for pattern in self.model_patterns:
                match = pattern.search(text)
                if match:
                    # Get last group (model part)
                    model = match.group(match.lastindex).strip().upper()
                    if len(model) >= 2:
                        return model
            
            # Keyword-based extraction
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in ['model', 'model:', 'type', 'type:']:
                    if i + 1 < len(words):
                        candidate = words[i + 1].strip()
                        # Remove trailing punctuation
                        candidate = re.sub(r'[,:;.]$', '', candidate)
                        if len(candidate) >= 2:
                            return candidate.upper()
            
            return None
            
        except Exception as e:
            logger.error(f"Model extraction error: {e}")
            return None
    
    @lru_cache(maxsize=256)
    def detect_language(self, text: str) -> str:
        """
        Fast language detection using Unicode ranges
        
        Args:
            text: Input text
            
        Returns:
            Language code ('en', 'hi', 'gu', 'mixed', 'unknown')
        """
        try:
            if not text:
                return 'unknown'
            
            # Sample first 200 chars for efficiency
            sample = text[:200]
            
            has_devanagari = sum(1 for c in sample if ord(c) in self.DEVANAGARI_RANGE)
            has_gujarati = sum(1 for c in sample if ord(c) in self.GUJARATI_RANGE)
            has_latin = sum(1 for c in sample if c.isalpha() and ord(c) in self.LATIN_RANGE)
            
            total_alpha = has_devanagari + has_gujarati + has_latin
            
            if total_alpha == 0:
                return 'unknown'
            
            # Determine primary language (>60% threshold)
            if has_devanagari / total_alpha > 0.6:
                return 'hi'
            elif has_gujarati / total_alpha > 0.6:
                return 'gu'
            elif has_latin / total_alpha > 0.6:
                return 'en'
            elif has_devanagari + has_gujarati > 0:
                return 'mixed'
            else:
                return 'en'  # Default to English
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'unknown'
    
    def has_currency_symbol(self, text: str) -> bool:
        """Check if text contains currency symbols"""
        return any(sym in text for sym in self.CURRENCY_SYMBOLS)
    
    def normalize_ocr_results(
        self, 
        ocr_results: List[Dict[str, Any]],
        remove_pii: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch normalize OCR results with field extraction
        
        Args:
            ocr_results: List of OCR detection dictionaries
            remove_pii: Remove PII from text
            
        Returns:
            List of normalized OCR results with extracted fields
        """
        if not ocr_results:
            return []
        
        normalized_results = []
        
        for result in ocr_results:
            original_text = result.get('text', '')
            if not original_text:
                continue
            
            try:
                # Normalize
                normalized_text = self.normalize_text(original_text, remove_pii=remove_pii)
                
                # Build normalized result
                normalized = {
                    'text': original_text,
                    'bbox': result.get('bbox', [0, 0, 0, 0]),
                    'confidence': result.get('confidence', 0.0),
                    'normalized_text': normalized_text,
                    'numbers': self.extract_numbers(original_text),
                    'horse_power': self.extract_horse_power(original_text),
                    'model_name': self.extract_model_name(original_text),
                    'language': self.detect_language(original_text),
                    'has_currency': self.has_currency_symbol(original_text),
                    'contains_keywords': self._detect_keywords_fast(normalized_text)
                }
                
                normalized_results.append(normalized)
                
            except Exception as e:
                logger.error(f"Error normalizing OCR result: {e}")
                # Include original result on error
                normalized_results.append(result)
        
        return normalized_results
    
    def _detect_keywords_fast(self, text: str) -> Dict[str, bool]:
        """
        Fast keyword detection using set intersection
        
        Args:
            text: Normalized text
            
        Returns:
            Dictionary of keyword presence
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Use set intersection (O(1) average)
        return {
            'dealer': bool(words & self.DEALER_KEYWORDS) or any(kw in text_lower for kw in self.DEALER_KEYWORDS),
            'model': bool(words & self.MODEL_KEYWORDS) or any(kw in text_lower for kw in self.MODEL_KEYWORDS),
            'horse_power': bool(words & self.HP_KEYWORDS) or any(kw in text_lower for kw in self.HP_KEYWORDS),
            'cost': bool(words & self.COST_KEYWORDS) or any(kw in text_lower for kw in self.COST_KEYWORDS),
            'invoice': bool(words & self.INVOICE_KEYWORDS) or any(kw in text_lower for kw in self.INVOICE_KEYWORDS)
        }
    
    def group_by_region(
        self, 
        ocr_results: List[Dict[str, Any]], 
        vertical_threshold: int = 15,
        horizontal_threshold: int = 50
    ) -> List[List[Dict[str, Any]]]:
        """
        Group OCR results into regions/lines
        
        Args:
            ocr_results: OCR results with bboxes
            vertical_threshold: Y-axis grouping threshold
            horizontal_threshold: X-axis sorting threshold
            
        Returns:
            List of grouped OCR results (lines/regions)
        """
        if not ocr_results:
            return []
        
        # Sort by Y then X
        sorted_results = sorted(
            ocr_results, 
            key=lambda x: (x['bbox'][1], x['bbox'][0])
        )
        
        groups = []
        current_group = [sorted_results[0]]
        current_y = sorted_results[0]['bbox'][1]
        
        for result in sorted_results[1:]:
            y_pos = result['bbox'][1]
            
            # Check if same line
            if abs(y_pos - current_y) <= vertical_threshold:
                current_group.append(result)
            else:
                # Sort current group by X
                current_group.sort(key=lambda x: x['bbox'][0])
                groups.append(current_group)
                
                # Start new group
                current_group = [result]
                current_y = y_pos
        
        # Add last group
        if current_group:
            current_group.sort(key=lambda x: x['bbox'][0])
            groups.append(current_group)
        
        return groups
    
    def create_text_sequence(
        self, 
        ocr_results: List[Dict[str, Any]],
        use_normalized: bool = True
    ) -> str:
        """
        Create readable text sequence from OCR results
        
        Args:
            ocr_results: OCR results
            use_normalized: Use normalized text
            
        Returns:
            Formatted text sequence
        """
        if not ocr_results:
            return ""
        
        groups = self.group_by_region(ocr_results)
        lines = []
        
        for group in groups:
            text_key = 'normalized_text' if use_normalized else 'text'
            line_texts = [
                r.get(text_key, r.get('text', ''))
                for r in group
                if r.get(text_key) or r.get('text')
            ]
            
            if line_texts:
                lines.append(' '.join(line_texts))
        
        return '\n'.join(lines)
    
    def batch_normalize(
        self, 
        texts: List[str], 
        preserve_case: bool = False
    ) -> List[str]:
        """Batch normalize multiple texts"""
        return [self.normalize_text(text, preserve_case) for text in texts]
    
    def correct_ocr_errors(self, text: str, context: str = 'numeric') -> str:
        """
        Correct common OCR errors based on context
        
        Args:
            text: Input text
            context: 'numeric', 'alpha', or 'mixed'
            
        Returns:
            Corrected text
        """
        if context == 'numeric':
            # Correct common OCR errors in numbers
            for wrong, right in self.OCR_CORRECTIONS.items():
                if wrong in text and text.replace(wrong, right).replace('.', '').isdigit():
                    text = text.replace(wrong, right)
        
        return text
    
    def clear_cache(self):
        """Clear all LRU caches"""
        self.extract_horse_power.cache_clear()
        self.extract_model_name.cache_clear()
        self.detect_language.cache_clear()
        logger.info("Caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'hp_cache': self.extract_horse_power.cache_info()._asdict(),
            'model_cache': self.extract_model_name.cache_info()._asdict(),
            'lang_cache': self.detect_language.cache_info()._asdict(),
            'processing_stats': self.stats
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()


# Singleton pattern
_normalizer = None

def get_normalizer() -> TextNormalizer:
    """Get or create singleton normalizer instance"""
    global _normalizer
    if _normalizer is None:
        _normalizer = TextNormalizer()
    return _normalizer


if __name__ == "__main__":
    import time
    
    normalizer = TextNormalizer()
    
    test_cases = [
        "Dealer: Mahindra Tractors Pvt Ltd, Price: ₹5,25,000",
        "Model: 575 DI, Horse Power: 50 HP",
        "Total Cost: Rs. 4.5 lakh",
        "Asset Cost: 2.5 crore rupees",
        "विक्रेता: ABC ट्रैक्टर्स, मूल्य: रु 3,00,000",
        "ડીલર: XYZ ટ્રેક્ટર્સ, કિંમત: રૂ 2,50,000",
        "Contact: dealer@email.com, Phone: +91 9876543210",
        "PAN: ABCDE1234F, GST: 27ABCDE1234F1Z5",
        "Invoice Date: 15/01/2024, Aadhar: 1234 5678 9012"
    ]
    
    print("Advanced Text Normalizer - Test Suite")
    print("=" * 70)
    
    # Individual tests
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text[:60]}...")
        
        start = time.time()
        normalized = normalizer.normalize_text(text)
        numbers = normalizer.extract_numbers(text)
        hp = normalizer.extract_horse_power(text)
        model = normalizer.extract_model_name(text)
        lang = normalizer.detect_language(text)
        elapsed = (time.time() - start) * 1000
        
        print(f"  Normalized: {normalized}")
        print(f"  Numbers: {numbers}")
        print(f"  HP: {hp}, Model: {model}, Lang: {lang}")
        print(f"  Time: {elapsed:.2f}ms")
    
    # Benchmark
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    iterations = 1000
    start = time.time()
    
    for _ in range(iterations):
        for text in test_cases:
            _ = normalizer.normalize_text(text)
            _ = normalizer.extract_numbers(text)
    
    elapsed = (time.time() - start) * 1000
    avg_per_text = elapsed / (iterations * len(test_cases))
    
    print(f"Total: {elapsed:.0f}ms for {iterations * len(test_cases)} operations")
    print(f"Average: {avg_per_text:.3f}ms per text")
    print(f"Throughput: {1000/avg_per_text:.0f} texts/sec")
    
    # Cache stats
    print("\n" + "=" * 70)
    print("Cache Statistics")
    print("=" * 70)
    
    stats = normalizer.get_cache_stats()
    for name, info in stats.items():
        if isinstance(info, dict) and 'hits' in info:
            hit_rate = info['hits'] / (info['hits'] + info['misses']) * 100 if info['hits'] + info['misses'] > 0 else 0
            print(f"{name}: hits={info['hits']}, misses={info['misses']}, rate={hit_rate:.1f}%")