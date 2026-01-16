import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import json
import logging
from typing import Dict, Any, Optional, List
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMProcessor:
    """
    Vision Language Model fallback for complex document understanding
    Uses Qwen2.5-VL for multilingual document analysis
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = None):
        """
        Initialize VLM processor
        
        Args:
            model_name: Hugging Face model name
            device: 'cpu', 'cuda', or None (auto)
        """
        self.model_name = model_name
        
        # Auto detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading VLM model: {model_name} on {self.device}")
        
        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            logger.info(f"VLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            self.model = None
            self.processor = None
    
    def is_available(self) -> bool:
        """Check if VLM is available"""
        return self.model is not None and self.processor is not None
    
    def extract_fields_vlm(self, image: Image.Image, 
                          ocr_text: str = "",
                          language: str = "English") -> Dict[str, Any]:
        """
        Extract fields using VLM
        
        Args:
            image: PIL Image
            ocr_text: Extracted OCR text for context
            language: Language of the document
            
        Returns:
            Dictionary with extracted fields
        """
        if not self.is_available():
            logger.error("VLM model not available")
            return {}
        
        try:
            start_time = time.time()
            
            # Construct prompt based on language
            if language.lower() in ["hindi", "hi"]:
                prompt = self._create_hindi_prompt(ocr_text)
            elif language.lower() in ["gujarati", "gu"]:
                prompt = self._create_gujarati_prompt(ocr_text)
            else:
                prompt = self._create_english_prompt(ocr_text)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode response
            generated_text = self.processor.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            # Extract JSON from response
            extracted_data = self._parse_vlm_response(generated_text)
            
            processing_time = time.time() - start_time
            logger.info(f"VLM processing completed in {processing_time:.2f}s")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error in VLM extraction: {e}")
            return {}
    
    def _create_english_prompt(self, ocr_text: str) -> str:
        """Create English prompt for VLM"""
        return f"""Analyze this invoice document and extract the following fields in JSON format:

OCR Text for reference: "{ocr_text[:500]}..."

Required fields:
1. dealer_name: Name of the dealer/seller/vendor
2. model_name: Model name of the tractor/asset
3. horse_power: Horse power value as a number
4. asset_cost: Total cost as a number (remove currency symbols)
5. signature_present: True/False if signature is present
6. stamp_present: True/False if stamp/seal is present

Instructions:
- Extract exact text values, don't invent information
- If a field is not found, use null
- Return ONLY valid JSON, no explanations
- For numeric fields, extract only the number
- For dealer name, look for words like: Dealer, Seller, Vendor, Supplier
- For model name, look for patterns like: 575 DI, 485 XP, etc.
- For horse power, look for: HP, H.P., Horse Power
- For cost, look for: ₹, Rs, Total, Amount, Cost

Return format:
{{
  "dealer_name": "string or null",
  "model_name": "string or null",
  "horse_power": number or null,
  "asset_cost": number or null,
  "signature_present": boolean,
  "stamp_present": boolean
}}"""
    
    def _create_hindi_prompt(self, ocr_text: str) -> str:
        """Create Hindi prompt for VLM"""
        return f"""इस इनवॉइस दस्तावेज़ का विश्लेषण करें और निम्नलिखित फ़ील्ड्स JSON प्रारूप में निकालें:

संदर्भ के लिए OCR टेक्स्ट: "{ocr_text[:500]}..."

आवश्यक फ़ील्ड्स:
1. dealer_name: डीलर/विक्रेता/वेंडर का नाम
2. model_name: ट्रैक्टर/संपत्ति का मॉडल नाम
3. horse_power: हॉर्स पावर मान संख्या के रूप में
4. asset_cost: कुल लागत संख्या के रूप में (मुद्रा प्रतीक हटाएं)
5. signature_present: हस्ताक्षर मौजूद है तो True/False
6. stamp_present: स्टैम्प/मोहर मौजूद है तो True/False

निर्देश:
- सटीक टेक्स्ट मान निकालें, जानकारी आविष्कार न करें
- यदि फ़ील्ड नहीं मिलता है, तो null का उपयोग करें
- केवल वैध JSON लौटाएं, कोई स्पष्टीकरण नहीं
- संख्यात्मक फ़ील्ड्स के लिए, केवल संख्या निकालें
- डीलर नाम के लिए देखें: डीलर, विक्रेता, वेंडर, आपूर्तिकर्ता
- मॉडल नाम के लिए देखें: 575 DI, 485 XP, आदि।
- हॉर्स पावर के लिए देखें: HP, H.P., हॉर्स पावर
- लागत के लिए देखें: ₹, रु, कुल, राशि, लागत

लौटाने का प्रारूप:
{{
  "dealer_name": "string या null",
  "model_name": "string या null",
  "horse_power": number या null,
  "asset_cost": number या null,
  "signature_present": boolean,
  "stamp_present": boolean
}}"""
    
    def _create_gujarati_prompt(self, ocr_text: str) -> str:
        """Create Gujarati prompt for VLM"""
        return f"""આ ઇનવોઇસ દસ્તાવેજનું વિશ્લેષણ કરો અને નીચેની ફીલ્ડ્સ JSON ફોર્મેટમાં કાઢો:

સંદર્ભ માટે OCR ટેક્સ્ટ: "{ocr_text[:500]}..."

જરૂરી ફીલ્ડ્સ:
1. dealer_name: ડીલર/વિક્રેતા/વેન્ડરનું નામ
2. model_name: ટ્રેક્ટર/એસેટનું મોડેલ નામ
3. horse_power: હોર્સ પાવર મૂલ્ય નંબર તરીકે
4. asset_cost: કુલ ખર્ચ નંબર તરીકે (કરન્સી ચિહ્નો દૂર કરો)
5. signature_present: સહી હાજર છે તો True/False
6. stamp_present: સ્ટેમ્પ/સીલ હાજર છે તો True/False

સૂચનાઓ:
- ચોક્કસ ટેક્સ્ટ મૂલ્યો કાઢો, માહિતી શોધશોળ ન કરો
- જો ફીલ્ડ ન મળે, તો null વાપરો
- ફક્ત માન્ય JSON પરત કરો, કોઈ સમજૂતી નહીં
- ન્યૂમેરિક ફીલ્ડ્સ માટે, ફક્ત નંબર કાઢો
- ડીલર નામ માટે જુઓ: ડીલર, વિક્રેતા, વેન્ડર, સપ્લાયર
- મોડેલ નામ માટે જુઓ: 575 DI, 485 XP, વગેરે.
- હોર્સ પાવર માટે જુઓ: HP, H.P., હોર્સ પાવર
- ખર્ચ માટે જુઓ: ₹, રૂ, કુલ, રકમ, ખર્ચ

પરત કરવાનું ફોર્મેટ:
{{
  "dealer_name": "string અથવા null",
  "model_name": "string અથવા null",
  "horse_power": number અથવા null,
  "asset_cost": number અથવા null,
  "signature_present": boolean,
  "stamp_present": boolean
}}"""
    
    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse VLM response and extract JSON
        
        Args:
            response: VLM generated response
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try to find JSON in the response
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response, re.DOTALL)
            
            if match:
                json_str = match.group(0)
                
                # Clean up common issues
                json_str = json_str.replace('```json', '').replace('```', '')
                json_str = json_str.strip()
                
                # Parse JSON
                data = json.loads(json_str)
                
                # Validate and clean data
                cleaned_data = {}
                
                for key in ['dealer_name', 'model_name', 'horse_power', 'asset_cost', 
                           'signature_present', 'stamp_present']:
                    if key in data:
                        value = data[key]
                        
                        # Handle null/None
                        if value in [None, 'null', 'NULL', 'None', 'N/A', '']:
                            cleaned_data[key] = None
                        
                        # Clean strings
                        elif isinstance(value, str):
                            if key in ['horse_power', 'asset_cost']:
                                # Extract numbers from strings
                                numbers = re.findall(r'\d+\.?\d*', value.replace(',', ''))
                                if numbers:
                                    try:
                                        cleaned_data[key] = float(numbers[0])
                                    except:
                                        cleaned_data[key] = None
                                else:
                                    cleaned_data[key] = None
                            else:
                                cleaned_data[key] = value.strip()
                        
                        # Handle booleans
                        elif isinstance(value, bool):
                            cleaned_data[key] = value
                        
                        # Handle numbers
                        elif isinstance(value, (int, float)):
                            cleaned_data[key] = value
                        
                        else:
                            cleaned_data[key] = None
                    
                    else:
                        cleaned_data[key] = None
                
                return cleaned_data
            
            else:
                logger.warning("No JSON found in VLM response")
                return self._extract_fields_heuristic(response)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return self._extract_fields_heuristic(response)
        
        except Exception as e:
            logger.error(f"Error parsing VLM response: {e}")
            return {}
    
    def _extract_fields_heuristic(self, response: str) -> Dict[str, Any]:
        """
        Extract fields using heuristics when JSON parsing fails
        
        Args:
            response: VLM response text
            
        Returns:
            Dictionary with extracted fields
        """
        extracted = {
            'dealer_name': None,
            'model_name': None,
            'horse_power': None,
            'asset_cost': None,
            'signature_present': False,
            'stamp_present': False
        }
        
        try:
            # Extract dealer name
            dealer_patterns = [
                r'dealer[:\s]+([^\n,]+)',
                r'dealer_name[:\s]+([^\n,]+)',
                r'डीलर[:\s]+([^\n,]+)',
                r'ડીલર[:\s]+([^\n,]+)'
            ]
            
            for pattern in dealer_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    extracted['dealer_name'] = match.group(1).strip()
                    break
            
            # Extract model name
            model_patterns = [
                r'model[:\s]+([A-Za-z0-9\s\-]+)',
                r'model_name[:\s]+([A-Za-z0-9\s\-]+)',
                r'मॉडल[:\s]+([^\n,]+)',
                r'મોડેલ[:\s]+([^\n,]+)'
            ]
            
            for pattern in model_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    extracted['model_name'] = match.group(1).strip()
                    break
            
            # Extract horse power
            hp_patterns = [
                r'horse[-\s]*power[:\s]*(\d+(?:\.\d+)?)',
                r'HP[:\s]*(\d+(?:\.\d+)?)',
                r'हॉर्स पावर[:\s]*(\d+(?:\.\d+)?)',
                r'હોર્સ પાવર[:\s]*(\d+(?:\.\d+)?)'
            ]
            
            for pattern in hp_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        extracted['horse_power'] = float(match.group(1))
                    except:
                        pass
                    break
            
            # Extract asset cost
            cost_patterns = [
                r'cost[:\s]*[₹$Rs]?\s*(\d[\d,]*\.?\d*)',
                r'asset[-\s]*cost[:\s]*[₹$Rs]?\s*(\d[\d,]*\.?\d*)',
                r'लागत[:\s]*[₹रु]?\s*(\d[\d,]*)',
                r'ખર્ચ[:\s]*[₹રૂ]?\s*(\d[\d,]*)'
            ]
            
            for pattern in cost_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        cost_str = match.group(1).replace(',', '')
                        extracted['asset_cost'] = float(cost_str)
                    except:
                        pass
                    break
            
            # Extract signature presence
            sig_patterns = [
                r'signature[-\s]*present[:\s]*(true|false)',
                r'हस्ताक्षर[-\s]*मौजूद[:\s]*(हाँ|नहीं|true|false)',
                r'સહી[-\s]*હાજર[:\s]*(હા|ના|true|false)'
            ]
            
            for pattern in sig_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = match.group(1).lower()
                    extracted['signature_present'] = value in ['true', 'हाँ', 'हा', 'yes']
                    break
            
            # Extract stamp presence
            stamp_patterns = [
                r'stamp[-\s]*present[:\s]*(true|false)',
                r'स्टैम्प[-\s]*मौजूद[:\s]*(हाँ|नहीं|true|false)',
                r'સ્ટેમ્પ[-\s]*હાજર[:\s]*(હા|ના|true|false)'
            ]
            
            for pattern in stamp_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = match.group(1).lower()
                    extracted['stamp_present'] = value in ['true', 'हाँ', 'हा', 'yes']
                    break
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error in heuristic extraction: {e}")
            return extracted
    
    def analyze_document_layout(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze document layout for better understanding
        
        Args:
            image: PIL Image
            
        Returns:
            Layout analysis results
        """
        if not self.is_available():
            return {}
        
        prompt = """Analyze this document and describe its layout structure.
        Identify key sections like: header, dealer information, model specifications, 
        pricing, signature area, stamp area.
        Return JSON with:
        {
          "layout_description": "brief description",
          "has_signature_area": true/false,
          "has_stamp_area": true/false,
          "main_sections": ["section1", "section2", ...]
        }"""
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            generated_text = self.processor.decode(
                generated_ids[0], skip_special_tokens=True
            )
            
            # Parse layout response
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, generated_text, re.DOTALL)
            
            if match:
                layout_data = json.loads(match.group(0))
                return layout_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in layout analysis: {e}")
            return {}

# Singleton instance
_vlm_processor = None

def get_vlm_processor(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", 
                      device: str = None) -> VLMProcessor:
    """
    Get or create VLM processor instance
    
    Args:
        model_name: Model name
        device: Device to use
        
    Returns:
        VLMProcessor instance
    """
    global _vlm_processor
    if _vlm_processor is None:
        _vlm_processor = VLMProcessor(model_name, device)
    return _vlm_processor

def should_use_vlm_fallback(rule_based_results: Dict[str, Any], 
                           confidence_threshold: float = 0.7) -> bool:
    """
    Determine if VLM fallback should be used
    
    Args:
        rule_based_results: Results from rule-based extraction
        confidence_threshold: Confidence threshold for fallback
        
    Returns:
        True if VLM fallback should be used
    """
    # Check overall confidence
    overall_confidence = rule_based_results.get('overall_confidence', 0)
    if overall_confidence < confidence_threshold:
        return True
    
    # Check individual field confidences
    required_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
    missing_or_low = 0
    
    for field in required_fields:
        field_data = rule_based_results.get(field, {})
        if field_data.get('value') is None or field_data.get('confidence', 0) < 0.6:
            missing_or_low += 1
    
    # If more than 1 field is missing or low confidence, use VLM
    return missing_or_low > 1

def merge_results(rule_based: Dict[str, Any], vlm_based: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge rule-based and VLM results
    
    Args:
        rule_based: Rule-based extraction results
        vlm_based: VLM extraction results
        
    Returns:
        Merged results
    """
    merged = rule_based.copy()
    
    # Merge each field
    for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
        rule_data = rule_based.get(field, {})
        vlm_value = vlm_based.get(field)
        
        # If rule-based has no value but VLM does, use VLM
        if rule_data.get('value') is None and vlm_value is not None:
            merged[field] = {
                'value': vlm_value,
                'confidence': 0.8,  # VLM confidence
                'source': 'vlm_fallback'
            }
        
        # If both have values, choose based on confidence
        elif rule_data.get('value') is not None and vlm_value is not None:
            rule_conf = rule_data.get('confidence', 0)
            # If VLM provides same value, increase confidence
            if str(rule_data['value']).lower() == str(vlm_value).lower():
                merged[field]['confidence'] = min(rule_conf * 1.2, 0.95)
                merged[field]['source'] = f"{rule_data.get('source', 'rule')}+vlm"
    
    # Merge signature/stamp info
    if 'signature' in merged:
        vlm_sig = vlm_based.get('signature_present')
        if vlm_sig is not None:
            merged['signature']['present'] = vlm_sig
            merged['signature']['confidence'] = max(
                merged['signature'].get('confidence', 0), 0.7
            )
    
    if 'stamp' in merged:
        vlm_stamp = vlm_based.get('stamp_present')
        if vlm_stamp is not None:
            merged['stamp']['present'] = vlm_stamp
            merged['stamp']['confidence'] = max(
                merged['stamp'].get('confidence', 0), 0.7
            )
    
    # Recalculate overall confidence
    if 'overall_confidence' in merged:
        # Simple average for now
        field_confs = []
        for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            if field in merged:
                field_confs.append(merged[field].get('confidence', 0))
        
        if field_confs:
            merged['overall_confidence'] = sum(field_confs) / len(field_confs)
    
    return merged

if __name__ == "__main__":
    # Test VLM processor
    print("Testing VLM Fallback Processor")
    print("=" * 60)
    
    # Initialize processor
    vlm = get_vlm_processor()
    
    if not vlm.is_available():
        print("VLM model not available. Trying to load smaller model...")
        vlm = get_vlm_processor("Qwen/Qwen2.5-VL-2B-Instruct")
    
    if vlm.is_available():
        print("✅ VLM processor initialized successfully")
        
        # Create a test image
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create a simple invoice image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text
        draw.text((50, 50), "Invoice No: INV-2024-001", fill='black')
        draw.text((50, 100), "Dealer: Mahindra Tractors Pvt Ltd", fill='black')
        draw.text((50, 150), "Model: 575 DI", fill='black')
        draw.text((50, 200), "Horse Power: 50 HP", fill='black')
        draw.text((50, 250), "Total Cost: ₹5,25,000", fill='black')
        
        print("\nTest Image Created")
        print("Extracting with VLM...")
        
        # Extract fields
        ocr_text = "Invoice No: INV-2024-001 Dealer: Mahindra Tractors Pvt Ltd Model: 575 DI Horse Power: 50 HP Total Cost: ₹5,25,000"
        results = vlm.extract_fields_vlm(img, ocr_text)
        
        print("\nVLM Extraction Results:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Test fallback logic
        print("\n" + "=" * 60)
        print("Testing Fallback Logic")
        print("=" * 60)
        
        # Simulate low confidence rule-based results
        rule_results = {
            'dealer_name': {'value': None, 'confidence': 0.3},
            'model_name': {'value': '575', 'confidence': 0.4},
            'horse_power': {'value': None, 'confidence': 0.2},
            'asset_cost': {'value': 525000, 'confidence': 0.9},
            'overall_confidence': 0.45
        }
        
        should_fallback = should_use_vlm_fallback(rule_results)
        print(f"Should use VLM fallback: {should_fallback}")
        
        if should_fallback:
            print("Using VLM as fallback...")
            merged = merge_results(rule_results, results)
            print("\nMerged Results:")
            for field, data in merged.items():
                if isinstance(data, dict) and 'value' in data:
                    print(f"{field}: {data['value']} (conf: {data.get('confidence', 0):.2f})")
        
    else:
        print("❌ VLM processor not available")
        print("Install transformers and torch to use this feature")