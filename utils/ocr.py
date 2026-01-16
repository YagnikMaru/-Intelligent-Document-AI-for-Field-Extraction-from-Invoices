import cv2
import numpy as np
from PIL import Image
import pdf2image
import os
import logging
from typing import List, Dict, Tuple, Any, Optional
import time

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. Please run: pip install pytesseract")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TesseractOCRProcessor:
    """
    OCR processor using Tesseract (works offline, no downloads needed)
    """
    
    def __init__(self, languages: List[str] = ['eng', 'hin', 'guj']):
        """
        Initialize Tesseract OCR
        
        Args:
            languages: List of language codes ['eng', 'hin', 'guj']
        """
        self.languages = languages
        
        if not TESSERACT_AVAILABLE:
            logger.error("pytesseract not installed. Please install: pip install pytesseract")
            return
        
        # Set Tesseract path for Windows
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        ]
        
        found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"✓ Tesseract found at: {path}")
                found = True
                break
        
        if not found:
            logger.warning("Tesseract not found at standard locations.")
            logger.warning("Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.warning("Or set path manually in code.")
            
            # Try to find in PATH
            try:
                pytesseract.get_tesseract_version()
                logger.info("✓ Found Tesseract in PATH")
                found = True
            except:
                logger.error("✗ Tesseract not found anywhere!")
        
        if found:
            logger.info(f"Tesseract initialized with languages: {languages}")
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        """
        Convert PDF to list of images
        """
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            return [np.array(img) for img in images]
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        processed = cv2.medianBlur(processed, 3)
        
        return processed
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> List[Dict[str, Any]]:
        """
        Extract text from single image
        """
        if not TESSERACT_AVAILABLE:
            logger.error("pytesseract not available")
            return []
        
        try:
            start_time = time.time()
            
            # Preprocess if requested
            if preprocess:
                image_to_process = self.preprocess_image(image)
            else:
                image_to_process = image
            
            # Convert to PIL Image for pytesseract
            pil_image = Image.fromarray(image_to_process)
            
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(
                pil_image, 
                lang='+'.join(self.languages),
                output_type=pytesseract.Output.DICT,
                config='--psm 3'
            )
            
            # Format results
            formatted_results = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                conf = int(data['conf'][i])
                if conf > 0:  # Only include confident results
                    text = data['text'][i].strip()
                    if text:  # Only include non-empty text
                        formatted_results.append({
                            'text': text,
                            'bbox': [
                                float(data['left'][i]),
                                float(data['top'][i]),
                                float(data['left'][i] + data['width'][i]),
                                float(data['top'][i] + data['height'][i])
                            ],
                            'confidence': float(conf) / 100.0,
                            'language': self._detect_language(text)
                        })
            
            processing_time = time.time() - start_time
            logger.info(f"OCR completed in {processing_time:.2f}s, found {len(formatted_results)} text blocks")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return []
    
    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image file
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract text
            results = self.extract_text(image)
            total_text = [r['text'] for r in results]
            avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
            
            # Also get full page text
            try:
                full_text = pytesseract.image_to_string(image, lang='+'.join(self.languages))
            except:
                full_text = ' '.join(total_text)
            
            return {
                'image_path': image_path,
                'total_text_blocks': len(results),
                'text_blocks': results,
                'full_text': full_text,
                'avg_confidence': avg_confidence,
                'languages_detected': list(set(r['language'] for r in results))
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'text_blocks': [],
                'full_text': ''
            }
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection
        """
        if not text:
            return 'unknown'
        
        # Character ranges
        devanagari_range = range(0x0900, 0x097F + 1)  # Hindi
        gujarati_range = range(0x0A80, 0x0AFF + 1)    # Gujarati
        
        devanagari_count = sum(1 for char in text if ord(char) in devanagari_range)
        gujarati_count = sum(1 for char in text if ord(char) in gujarati_range)
        english_count = sum(1 for char in text if char.isalpha() and char.isascii())
        
        total_alpha = devanagari_count + gujarati_count + english_count
        
        if total_alpha == 0:
            return 'unknown'
        
        if devanagari_count / total_alpha > 0.5:
            return 'hin'
        elif gujarati_count / total_alpha > 0.5:
            return 'guj'
        elif english_count / total_alpha > 0.5:
            return 'eng'
        else:
            return 'mixed'
    
    def get_text_by_region(self, image: np.ndarray, region_bbox: List[float]) -> List[str]:
        """
        Extract text from specific region of image
        """
        if not TESSERACT_AVAILABLE:
            return []
        
        # Crop image to region
        x1, y1, x2, y2 = map(int, region_bbox)
        cropped = image[y1:y2, x1:x2]
        
        # Extract text from cropped region
        results = self.extract_text(cropped)
        return [r['text'] for r in results]
    
    def visualize_results(self, image: np.ndarray, results: List[Dict], 
                          output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize OCR results on image
        """
        # Create copy for visualization
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text label
            label = f"{text[:20]}... ({confidence:.2f})"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image

# Singleton instance for reuse
_ocr_processor = None

def get_ocr_processor(languages: List[str] = ['eng', 'hin', 'guj']):
    """
    Get or create OCR processor instance
    
    Args:
        languages: Language codes ['eng', 'hin', 'guj']
        
    Returns:
        TesseractOCRProcessor instance
    """
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = TesseractOCRProcessor(languages=languages)
    
    # Check if processor initialized properly
    if not TESSERACT_AVAILABLE:
        logger.error("Cannot create OCR processor: pytesseract not installed")
    
    return _ocr_processor

if __name__ == "__main__":
    print("Testing OCR Processor...")
    
    # Initialize OCR
    processor = get_ocr_processor()
    
    if not TESSERACT_AVAILABLE:
        print("✗ pytesseract not installed. Please run: pip install pytesseract")
        print("Then install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        exit(1)
    
    # Create a test image
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test Invoice", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(test_image, "Dealer: ABC Tractors", (50, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_image, "Model: 575 DI", (50, 230), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_image, "Horse Power: 50 HP", (50, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_image, "Cost: ₹5,25,000", (50, 330), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save test image
    cv2.imwrite("test_ocr.jpg", cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    
    print("Created test image: test_ocr.jpg")
    
    # Extract text
    results = processor.extract_text(test_image)
    print(f"\nFound {len(results)} text blocks:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Text: {result['text']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Language: {result['language']}")
        print(f"   BBox: {result['bbox']}")
        print()
    
    # Test full extraction
    print("\nTesting full extraction from file...")
    full_result = processor.extract_from_image("test_ocr.jpg")
    print(f"Full text: {full_result.get('full_text', '')[:200]}...")
    
    print("\n✅ OCR test completed successfully!")