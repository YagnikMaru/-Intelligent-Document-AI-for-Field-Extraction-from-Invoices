import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import re
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages"""
    ENGLISH = ("english", "en", "eng")
    HINDI = ("hindi", "hi", "hin")
    GUJARATI = ("gujarati", "gu", "guj")
    
    @classmethod
    def detect(cls, lang_str: str) -> 'Language':
        """Detect language from string"""
        lang_lower = lang_str.lower()
        for lang in cls:
            if lang_lower in lang.value:
                return lang
        return cls.ENGLISH


@dataclass
class ExtractionResult:
    """Structured extraction result"""
    dealer_name: Optional[str] = None
    model_name: Optional[str] = None
    horse_power: Optional[float] = None
    asset_cost: Optional[float] = None
    signature_present: bool = False
    stamp_present: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dealer_name': self.dealer_name,
            'model_name': self.model_name,
            'horse_power': self.horse_power,
            'asset_cost': self.asset_cost,
            'signature_present': self.signature_present,
            'stamp_present': self.stamp_present
        }


class EfficientVLMProcessor:
    """Optimized Vision Language Model processor with caching and batch support"""
    
    _instance = None
    _initialized = False
    
    # Pre-compiled patterns (class-level)
    _JSON_PATTERN = re.compile(r'\{.*?\}', re.DOTALL)
    _NUMBER_PATTERN = re.compile(r'\d+\.?\d*')
    _FIELD_PATTERNS = {
        'dealer': [
            re.compile(r'dealer[:\s]+([^\n,]+)', re.I),
            re.compile(r'डीलर[:\s]+([^\n,]+)', re.I),
            re.compile(r'ડીલર[:\s]+([^\n,]+)', re.I)
        ],
        'model': [
            re.compile(r'model[:\s]+([A-Za-z0-9\s\-]+)', re.I),
            re.compile(r'मॉडल[:\s]+([^\n,]+)', re.I),
            re.compile(r'મોડેલ[:\s]+([^\n,]+)', re.I)
        ],
        'hp': [
            re.compile(r'horse[-\s]*power[:\s]*(\d+(?:\.\d+)?)', re.I),
            re.compile(r'HP[:\s]*(\d+(?:\.\d+)?)', re.I),
            re.compile(r'हॉर्स पावर[:\s]*(\d+(?:\.\d+)?)', re.I)
        ],
        'cost': [
            re.compile(r'cost[:\s]*[₹$Rs]?\s*(\d[\d,]*\.?\d*)', re.I),
            re.compile(r'लागत[:\s]*[₹रु]?\s*(\d[\d,]*)', re.I),
            re.compile(r'ખર્ચ[:\s]*[₹રૂ]?\s*(\d[\d,]*)', re.I)
        ]
    }
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", 
                 device: str = None, max_batch_size: int = 4):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cpu', 'cuda', or None (auto)
            max_batch_size: Maximum batch size for processing
        """
        if self._initialized:
            return
        
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """Load model with optimizations"""
        logger.info(f"Loading VLM: {self.model_name} on {self.device}")
        
        try:
            # Use bfloat16 on CUDA for better performance
            dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else (
                torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Efficient loading
            )
            
            # Set to eval mode for inference
            self.model.eval()
            
            # Enable gradient checkpointing for memory efficiency if available
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            logger.info(f"✓ VLM loaded (dtype: {dtype})")
            
        except Exception as e:
            logger.error(f"✗ Failed to load VLM: {e}")
            self.model = None
            self.processor = None
    
    def is_available(self) -> bool:
        """Check if VLM is ready"""
        return self.model is not None and self.processor is not None
    
    @lru_cache(maxsize=3)
    def _get_prompt(self, language: Language, ocr_context: str = "") -> str:
        """Cached prompt generation"""
        context = f'OCR context: "{ocr_context[:300]}..."' if ocr_context else ""
        
        prompts = {
            Language.ENGLISH: f"""Extract invoice fields as JSON:
{context}

Required: dealer_name, model_name, horse_power (number), asset_cost (number), 
signature_present (bool), stamp_present (bool)

Rules:
- Extract exact values only
- Use null for missing fields
- Numbers only for numeric fields
- Return valid JSON only

Format:
{{"dealer_name": null, "model_name": null, "horse_power": null, 
"asset_cost": null, "signature_present": false, "stamp_present": false}}""",
            
            Language.HINDI: f"""JSON प्रारूप में इनवॉइस फ़ील्ड निकालें:
{context}

आवश्यक: dealer_name, model_name, horse_power (संख्या), asset_cost (संख्या),
signature_present (bool), stamp_present (bool)

JSON लौटाएं।""",
            
            Language.GUJARATI: f"""JSON ફોર્મેટમાં ઇનવોઇસ ફીલ્ડ્સ કાઢો:
{context}

જરૂરી: dealer_name, model_name, horse_power (નંબર), asset_cost (નંબર),
signature_present (bool), stamp_present (bool)

JSON પરત કરો।"""
        }
        
        return prompts.get(language, prompts[Language.ENGLISH])
    
    def extract_fields(self, image: Image.Image, ocr_text: str = "",
                      language: str = "english") -> ExtractionResult:
        """
        Extract fields using VLM (optimized)
        
        Args:
            image: PIL Image
            ocr_text: OCR context (truncated automatically)
            language: Document language
            
        Returns:
            ExtractionResult object
        """
        if not self.is_available():
            logger.error("VLM not available")
            return ExtractionResult()
        
        try:
            lang = Language.detect(language)
            prompt = self._get_prompt(lang, ocr_text[:300])
            
            # Prepare input
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text], images=[image],
                padding=True, return_tensors="pt"
            ).to(self.device)
            
            # Generate with optimizations
            with torch.inference_mode():  # More efficient than no_grad
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=384,  # Reduced from 512
                    do_sample=False,
                    num_beams=1,  # Greedy decoding (fastest)
                    early_stopping=True
                )
            
            # Decode
            response = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            # Parse response
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"VLM extraction error: {e}")
            return ExtractionResult()
    
    def batch_extract(self, images: List[Image.Image], 
                     ocr_texts: List[str] = None,
                     language: str = "english") -> List[ExtractionResult]:
        """
        Batch process multiple images efficiently
        
        Args:
            images: List of PIL Images
            ocr_texts: Optional list of OCR contexts
            language: Document language
            
        Returns:
            List of ExtractionResult objects
        """
        if not self.is_available():
            return [ExtractionResult() for _ in images]
        
        ocr_texts = ocr_texts or [""] * len(images)
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.max_batch_size):
            batch_imgs = images[i:i + self.max_batch_size]
            batch_ocr = ocr_texts[i:i + self.max_batch_size]
            
            batch_results = self._process_batch(batch_imgs, batch_ocr, language)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, images: List[Image.Image], 
                      ocr_texts: List[str], language: str) -> List[ExtractionResult]:
        """Process a batch of images"""
        try:
            lang = Language.detect(language)
            prompts = [self._get_prompt(lang, ocr[:300]) for ocr in ocr_texts]
            
            # Prepare batch inputs
            messages_batch = [
                [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }]
                for prompt in prompts
            ]
            
            texts = [
                self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                for msgs in messages_batch
            ]
            
            inputs = self.processor(
                text=texts, images=images,
                padding=True, return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=False,
                    num_beams=1
                )
            
            responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            return [self._parse_response(resp) for resp in responses]
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [ExtractionResult() for _ in images]
    
    def _parse_response(self, response: str) -> ExtractionResult:
        """Parse VLM response efficiently"""
        try:
            # Try JSON parsing first
            match = self._JSON_PATTERN.search(response)
            if match:
                data = json.loads(match.group(0).replace('```json', '').replace('```', '').strip())
                return self._dict_to_result(data)
            
            # Fallback to pattern matching
            return self._extract_with_patterns(response)
            
        except json.JSONDecodeError:
            return self._extract_with_patterns(response)
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return ExtractionResult()
    
    def _dict_to_result(self, data: Dict[str, Any]) -> ExtractionResult:
        """Convert dict to ExtractionResult with validation"""
        def safe_str(v): return str(v).strip() if v and v not in ('null', 'None', '') else None
        def safe_num(v): 
            if isinstance(v, (int, float)): return float(v)
            if isinstance(v, str):
                nums = self._NUMBER_PATTERN.findall(v.replace(',', ''))
                return float(nums[0]) if nums else None
            return None
        def safe_bool(v): return bool(v) if isinstance(v, bool) else v in (True, 'true', 'True')
        
        return ExtractionResult(
            dealer_name=safe_str(data.get('dealer_name')),
            model_name=safe_str(data.get('model_name')),
            horse_power=safe_num(data.get('horse_power')),
            asset_cost=safe_num(data.get('asset_cost')),
            signature_present=safe_bool(data.get('signature_present', False)),
            stamp_present=safe_bool(data.get('stamp_present', False))
        )
    
    def _extract_with_patterns(self, text: str) -> ExtractionResult:
        """Pattern-based extraction fallback"""
        result = ExtractionResult()
        
        # Extract dealer
        for pattern in self._FIELD_PATTERNS['dealer']:
            if match := pattern.search(text):
                result.dealer_name = match.group(1).strip()
                break
        
        # Extract model
        for pattern in self._FIELD_PATTERNS['model']:
            if match := pattern.search(text):
                result.model_name = match.group(1).strip()
                break
        
        # Extract HP
        for pattern in self._FIELD_PATTERNS['hp']:
            if match := pattern.search(text):
                try:
                    result.horse_power = float(match.group(1))
                except: pass
                break
        
        # Extract cost
        for pattern in self._FIELD_PATTERNS['cost']:
            if match := pattern.search(text):
                try:
                    result.asset_cost = float(match.group(1).replace(',', ''))
                except: pass
                break
        
        return result


def get_vlm_processor(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", 
            device: str = None) -> EfficientVLMProcessor:
    """Get VLM processor instance (singleton)"""
    return EfficientVLMProcessor(model_name, device)


def should_use_vlm_fallback(rule_results: Dict[str, Any], 
                   confidence_threshold: float = 0.7,
                   min_fields_threshold: int = 2) -> bool:
    """
    Determine if VLM fallback is needed
    
    Args:
        rule_results: Rule-based extraction results
        confidence_threshold: Overall confidence threshold
        min_fields_threshold: Min number of low-confidence fields to trigger
        
    Returns:
        True if VLM should be used
    """
    overall = rule_results.get('overall_confidence', 0)
    if overall < confidence_threshold:
        return True
    
    # Check critical fields
    critical = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
    low_conf_count = sum(
        1 for field in critical
        if field in rule_results and (
            rule_results[field].get('value') is None or
            rule_results[field].get('confidence', 0) < 0.6
        )
    )
    
    return low_conf_count >= min_fields_threshold


def merge_results(rule_results: Dict[str, Any], 
                 vlm_result: ExtractionResult,
                 vlm_confidence: float = 0.8) -> Dict[str, Any]:
    """
    Intelligently merge rule-based and VLM results
    
    Args:
        rule_results: Rule-based extraction
        vlm_result: VLM extraction result
        vlm_confidence: Base confidence for VLM results
        
    Returns:
        Merged results dictionary
    """
    merged = rule_results.copy()
    vlm_dict = vlm_result.to_dict()
    
    # Merge text/numeric fields
    for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
        rule_data = rule_results.get(field, {})
        vlm_value = vlm_dict.get(field)
        
        rule_value = rule_data.get('value')
        rule_conf = rule_data.get('confidence', 0)
        
        # VLM fills missing values
        if rule_value is None and vlm_value is not None:
            merged[field] = {
                'value': vlm_value,
                'confidence': vlm_confidence,
                'source': 'vlm_fallback'
            }
        # Both have values - check agreement
        elif rule_value and vlm_value:
            # Normalize for comparison
            rule_norm = str(rule_value).lower().replace(' ', '')
            vlm_norm = str(vlm_value).lower().replace(' ', '')
            
            if rule_norm == vlm_norm or vlm_norm in rule_norm or rule_norm in vlm_norm:
                # Agreement boosts confidence
                merged[field]['confidence'] = min(rule_conf * 1.25, 0.95)
                merged[field]['source'] = 'rule+vlm_confirmed'
            else:
                # Disagreement - prefer higher confidence
                if vlm_confidence > rule_conf:
                    merged[field] = {
                        'value': vlm_value,
                        'confidence': vlm_confidence,
                        'source': 'vlm_override'
                    }
    
    # Merge binary fields
    for field, vlm_key in [('signature', 'signature_present'), ('stamp', 'stamp_present')]:
        if field in merged and vlm_dict.get(vlm_key):
            merged[field]['present'] = True
            merged[field]['confidence'] = max(merged[field].get('confidence', 0), vlm_confidence)
    
    # Recalculate overall confidence
    confidences = [
        merged[f]['confidence'] for f in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        if f in merged and merged[f].get('value') is not None
    ]
    
    if confidences:
        merged['overall_confidence'] = sum(confidences) / len(confidences)
    
    return merged


if __name__ == "__main__":
    print("Testing Efficient VLM Processor\n")
    
    vlm = get_vlm_processor()
    
    if not vlm.is_available():
        print("✗ VLM not available. Install: pip install transformers torch")
        print("  Trying smaller model...")
        vlm = get_vlm_processor("Qwen/Qwen2.5-VL-2B-Instruct")
    
    if vlm.is_available():
        print("✓ VLM initialized\n")
        
        # Create test image
        img = Image.new('RGB', (800, 600), 'white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        texts = [
            (50, 50, "INVOICE #2024-001"),
            (50, 100, "Dealer: ABC Tractors Ltd"),
            (50, 150, "Model: 575 DI"),
            (50, 200, "HP: 50"),
            (50, 250, "Cost: ₹5,25,000")
        ]
        
        for x, y, text in texts:
            draw.text((x, y), text, fill='black')
        
        print("📄 Extracting from test image...")
        ocr = "Dealer: ABC Tractors Ltd Model: 575 DI HP: 50 Cost: ₹525000"
        
        import time
        start = time.time()
        result = vlm.extract_fields(img, ocr)
        elapsed = time.time() - start
        
        print(f"⏱️  Processed in {elapsed:.2f}s\n")
        print("✓ Extracted:")
        for k, v in result.to_dict().items():
            print(f"  {k}: {v}")
        
        # Test merging
        print("\n\n🔀 Testing result merging...")
        rule_results = {
            'dealer_name': {'value': None, 'confidence': 0.3},
            'model_name': {'value': '575', 'confidence': 0.5},
            'horse_power': {'value': 50, 'confidence': 0.9},
            'asset_cost': {'value': 525000, 'confidence': 0.8},
            'overall_confidence': 0.55
        }
        
        should_fallback = should_use_vlm_fallback(rule_results)
        print(f"Use VLM fallback: {should_fallback}")
        
        if should_fallback:
            merged = merge_results(rule_results, result)
            print("\n✓ Merged results:")
            for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                if field in merged:
                    d = merged[field]
                    print(f"  {field}: {d.get('value')} (conf: {d.get('confidence', 0):.2%}, src: {d.get('source', 'rule')})")
            print(f"\n📊 Overall: {merged.get('overall_confidence', 0):.2%}")
        
    else:
        print("✗ VLM not available")
    
    print("\n✅ Tests completed!")