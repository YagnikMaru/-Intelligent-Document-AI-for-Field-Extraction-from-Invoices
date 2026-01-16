import re
import unicodedata
from typing import Dict, List, Any, Optional
import logging
from langdetect import detect, LangDetectException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextNormalizer:
    """
    Text normalization for OCR output
    Handles multilingual text, currency symbols, and formatting
    """
    
    # Currency symbols and abbreviations
    CURRENCY_SYMBOLS = {
        '₹', 'Rs', 'Rs.', 'INR', 'रु', 'रुपए', 'રૂ', 'રૂપિયા',
        '$', 'USD', '€', 'EUR', '£', 'GBP', '¥', 'JPY'
    }
    
    # Common invoice terms in different languages
    INVOICE_TERMS = {
        'english': {
            'dealer': ['dealer', 'seller', 'vendor', 'supplier', 'distributor'],
            'model': ['model', 'type', 'variant', 'version'],
            'horse_power': ['horse power', 'hp', 'h.p.', 'power'],
            'cost': ['cost', 'price', 'amount', 'total', 'value', 'asset cost'],
            'invoice': ['invoice', 'bill', 'quotation', 'quote', 'estimate']
        },
        'hindi': {
            'dealer': ['डीलर', 'विक्रेता', 'वेंडर', 'आपूर्तिकर्ता'],
            'model': ['मॉडल', 'प्रकार', 'वेरिएंट'],
            'horse_power': ['हॉर्स पावर', 'एचपी', 'शक्ति'],
            'cost': ['लागत', 'मूल्य', 'राशि', 'कुल', 'संपत्ति लागत']
        },
        'gujarati': {
            'dealer': ['ડીલર', 'વેચનાર', 'વિક્રેતા'],
            'model': ['મોડેલ', 'પ્રકાર', 'વેરિઅન્ટ'],
            'horse_power': ['હોર્સ પાવર', 'એચપી', 'શક્તિ'],
            'cost': ['ખર્ચ', 'કિંમત', 'રકમ', 'કુલ', 'એસેટ કોસ્ટ']
        }
    }
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Compile regex patterns for efficiency"""
        # Currency patterns
        self.currency_pattern = re.compile(
            r'(' + '|'.join(re.escape(sym) for sym in self.CURRENCY_SYMBOLS) + r')[\s:]*',
            re.IGNORECASE
        )
        
        # Number patterns
        self.number_pattern = re.compile(r'[0-9,]+(?:\.[0-9]+)?')
        
        # HP pattern
        self.hp_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(?:HP|H\.P\.|Horse Power|हॉर्स पावर|હોર્સ પાવર)', re.IGNORECASE)
        
        # Model pattern (common tractor models)
        self.model_pattern = re.compile(r'(\d{3,4}\s*(?:DI|XP|XT|MX|TA|FE|GT|R|S|E)?)', re.IGNORECASE)
        
        # Special characters (keep alphanumeric and spaces)
        self.special_char_pattern = re.compile(r'[^\w\s.,:;/()-]', re.UNICODE)
        
        # Multiple spaces
        self.multi_space_pattern = re.compile(r'\s+')
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone pattern (Indian numbers)
        self.phone_pattern = re.compile(r'(\+91[\-\s]?)?[789]\d{9}')
        
        # Date patterns
        self.date_pattern = re.compile(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{2,4}[/\-]\d{1,2}[/\-]\d{1,2})\b')
    
    def normalize_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Normalize text by removing unwanted characters, standardizing format
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        try:
            # Convert to string if not already
            text = str(text)
            
            # Normalize unicode (NFKC form)
            text = unicodedata.normalize('NFKC', text)
            
            # Remove email addresses (PII)
            text = self.email_pattern.sub('[EMAIL]', text)
            
            # Remove phone numbers (PII)
            text = self.phone_pattern.sub('[PHONE]', text)
            
            # Remove dates
            text = self.date_pattern.sub('[DATE]', text)
            
            # Remove currency symbols
            text = self.currency_pattern.sub('', text)
            
            # Remove special characters but keep some punctuation
            text = self.special_char_pattern.sub(' ', text)
            
            # Replace multiple spaces with single space
            text = self.multi_space_pattern.sub(' ', text)
            
            # Trim whitespace
            text = text.strip()
            
            # Convert to lowercase unless preserving case
            if not preserve_case:
                text = text.lower()
            
            return text
            
        except Exception as e:
            logger.error(f"Error normalizing text '{text}': {e}")
            return text
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract all numbers from text
        
        Args:
            text: Input text
            
        Returns:
            List of extracted numbers as floats
        """
        numbers = []
        try:
            matches = self.number_pattern.findall(text)
            for match in matches:
                # Remove commas from numbers like 1,25,000
                clean_num = match.replace(',', '')
                try:
                    numbers.append(float(clean_num))
                except ValueError:
                    continue
        except Exception as e:
            logger.error(f"Error extracting numbers: {e}")
        
        return numbers
    
    def extract_horse_power(self, text: str) -> Optional[float]:
        """
        Extract horse power value from text
        
        Args:
            text: Input text
            
        Returns:
            Horse power value as float or None
        """
        try:
            match = self.hp_pattern.search(text)
            if match:
                return float(match.group(1))
            
            # Alternative: look for number followed by HP
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in ['hp', 'h.p.', 'horse', 'power'] and i > 0:
                    try:
                        return float(words[i-1].replace(',', ''))
                    except ValueError:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting horse power: {e}")
            return None
    
    def extract_model_name(self, text: str) -> Optional[str]:
        """
        Extract model name from text
        
        Args:
            text: Input text
            
        Returns:
            Model name or None
        """
        try:
            # Look for common tractor model patterns
            match = self.model_pattern.search(text)
            if match:
                return match.group(1).upper()
            
            # Look for keywords indicating model information
            model_keywords = ['model', 'type', 'variant', 'मॉडल', 'મોડેલ']
            words = text.split()
            
            for i, word in enumerate(words):
                if word.lower() in model_keywords and i + 1 < len(words):
                    # Return the next word as potential model
                    candidate = words[i + 1]
                    if candidate.isalnum():
                        return candidate.upper()
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting model name: {e}")
            return None
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Input text
            
        Returns:
            Language code ('en', 'hi', 'gu', or 'unknown')
        """
        try:
            # Simple character-based detection first (faster)
            if self._contains_devanagari(text):
                return 'hi'
            elif self._contains_gujarati(text):
                return 'gu'
            elif self._contains_english(text):
                return 'en'
            
            # Use langdetect as fallback
            lang = detect(text)
            return lang if lang in ['en', 'hi', 'gu'] else 'unknown'
            
        except LangDetectException:
            return 'unknown'
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'unknown'
    
    def _contains_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari characters (Hindi)"""
        devanagari_range = range(0x0900, 0x097F + 1)
        return any(ord(char) in devanagari_range for char in text)
    
    def _contains_gujarati(self, text: str) -> bool:
        """Check if text contains Gujarati characters"""
        gujarati_range = range(0x0A80, 0x0AFF + 1)
        return any(ord(char) in gujarati_range for char in text)
    
    def _contains_english(self, text: str) -> bool:
        """Check if text contains English characters"""
        return any(char.isalpha() and char.isascii() for char in text)
    
    def normalize_ocr_results(self, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize OCR results
        
        Args:
            ocr_results: List of OCR result dictionaries
            
        Returns:
            Normalized OCR results
        """
        normalized_results = []
        
        for result in ocr_results:
            normalized = result.copy()
            original_text = result.get('text', '')
            
            # Normalize text
            normalized_text = self.normalize_text(original_text)
            normalized['normalized_text'] = normalized_text
            
            # Extract additional information
            normalized['numbers'] = self.extract_numbers(original_text)
            normalized['horse_power'] = self.extract_horse_power(original_text)
            normalized['model_name'] = self.extract_model_name(original_text)
            normalized['language'] = self.detect_language(original_text)
            
            # Detect if this text block contains key information
            normalized['contains_keywords'] = self._contains_keywords(normalized_text)
            
            normalized_results.append(normalized)
        
        return normalized_results
    
    def _contains_keywords(self, text: str) -> Dict[str, bool]:
        """
        Check if text contains keywords for each field
        
        Args:
            text: Normalized text
            
        Returns:
            Dictionary indicating which keywords are present
        """
        result = {
            'dealer': False,
            'model': False,
            'horse_power': False,
            'cost': False,
            'invoice': False
        }
        
        text_lower = text.lower()
        
        # Check English keywords
        for field, keywords in self.INVOICE_TERMS['english'].items():
            if any(keyword in text_lower for keyword in keywords):
                result[field] = True
        
        return result
    
    def group_by_region(self, ocr_results: List[Dict[str, Any]], 
                        vertical_threshold: int = 50) -> List[List[Dict[str, Any]]]:
        """
        Group OCR results by vertical regions (for reading left-to-right)
        
        Args:
            ocr_results: List of OCR results
            vertical_threshold: Max vertical distance for grouping
            
        Returns:
            List of grouped OCR results
        """
        if not ocr_results:
            return []
        
        # Sort by y-coordinate (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][1])
        
        groups = []
        current_group = [sorted_results[0]]
        current_y = sorted_results[0]['bbox'][1]
        
        for result in sorted_results[1:]:
            y_pos = result['bbox'][1]
            
            # If close vertically, add to current group
            if abs(y_pos - current_y) <= vertical_threshold:
                current_group.append(result)
            else:
                # Sort current group by x-coordinate (left to right)
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
    
    def create_text_sequence(self, ocr_results: List[Dict[str, Any]]) -> str:
        """
        Create logical text sequence from OCR results
        
        Args:
            ocr_results: List of OCR results
            
        Returns:
            Sequential text reading left-to-right, top-to-bottom
        """
        groups = self.group_by_region(ocr_results)
        
        lines = []
        for group in groups:
            line_texts = []
            for result in group:
                text = result.get('normalized_text', result.get('text', ''))
                if text:
                    line_texts.append(text)
            
            if line_texts:
                lines.append(' '.join(line_texts))
        
        return '\n'.join(lines)

# Singleton instance
_normalizer = None

def get_normalizer() -> TextNormalizer:
    """
    Get or create TextNormalizer instance
    
    Returns:
        TextNormalizer instance
    """
    global _normalizer
    if _normalizer is None:
        _normalizer = TextNormalizer()
    return _normalizer

if __name__ == "__main__":
    # Test the normalizer
    normalizer = TextNormalizer()
    
    test_cases = [
        "Dealer: Mahindra Tractors, Price: ₹5,25,000",
        "Model: 575 DI, Horse Power: 50 HP",
        "Total Cost: Rs. 4,50,000.50",
        "विक्रेता: ABC ट्रैक्टर्स, मूल्य: रु 3,00,000",
        "ડીલર: XYZ ટ્રેક્ટર્સ, કિંમત: રૂ 2,50,000",
        "Contact: dealer@email.com, Phone: +91 9876543210"
    ]
    
    print("Text Normalization Test")
    print("=" * 60)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Original: {test_text}")
        normalized = normalizer.normalize_text(test_text)
        print(f"Normalized: {normalized}")
        
        numbers = normalizer.extract_numbers(test_text)
        print(f"Numbers: {numbers}")
        
        hp = normalizer.extract_horse_power(test_text)
        print(f"Horse Power: {hp}")
        
        model = normalizer.extract_model_name(test_text)
        print(f"Model Name: {model}")
        
        language = normalizer.detect_language(test_text)
        print(f"Language: {language}")