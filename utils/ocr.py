import cv2
import numpy as np
from PIL import Image
import pdf2image
import os
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not installed. Run: pip install pytesseract")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class EfficientTesseractOCR:
    """Optimized OCR processor with caching, parallel processing, and memory efficiency"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for resource efficiency"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, languages: List[str] = None, num_workers: int = None):
        """
        Args:
            languages: Language codes (default: ['eng', 'hin', 'guj'])
            num_workers: Number of parallel workers (default: CPU count)
        """
        if self._initialized:
            return
            
        self.languages = languages or ['eng', 'hin', 'guj']
        self.lang_string = '+'.join(self.languages)
        self.num_workers = num_workers or min(4, mp.cpu_count())
        
        if not TESSERACT_AVAILABLE:
            logger.error("pytesseract not installed")
            return
        
        self._setup_tesseract()
        self._initialized = True
    
    def _setup_tesseract(self) -> bool:
        """Setup Tesseract executable path"""
        paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract'
        ]
        
        for path in paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"✓ Tesseract: {path}")
                return True
        
        try:
            pytesseract.get_tesseract_version()
            logger.info("✓ Tesseract found in PATH")
            return True
        except:
            logger.error("✗ Tesseract not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki")
            return False
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_preprocess_params(image_shape: Tuple[int, int]) -> Dict[str, int]:
        """Cached preprocessing parameters based on image size"""
        h, w = image_shape
        area = h * w
        
        if area > 4000000:  # Large images
            return {'blur_kernel': 5, 'block_size': 15, 'c': 3}
        elif area > 1000000:  # Medium images
            return {'blur_kernel': 3, 'block_size': 11, 'c': 2}
        else:  # Small images
            return {'blur_kernel': 3, 'block_size': 9, 'c': 2}
    
    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Optimized image preprocessing
        
        Args:
            image: Input image
            enhance: Apply enhancement (slower but better quality)
        """
        # Convert to grayscale efficiently
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image
        
        if not enhance:
            return gray
        
        # Get cached parameters
        params = self._get_preprocess_params(gray.shape)
        
        # Denoise first (more efficient)
        denoised = cv2.medianBlur(gray, params['blur_kernel'])
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, params['block_size'], params['c']
        )
        
        return binary
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True, 
                     psm: int = 3, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Extract text from image with confidence filtering
        
        Args:
            image: Input image
            preprocess: Apply preprocessing
            psm: Page segmentation mode (3=auto, 6=single block)
            min_confidence: Minimum confidence threshold (0-1)
        """
        if not TESSERACT_AVAILABLE:
            return []
        
        try:
            # Preprocess
            img = self.preprocess_image(image) if preprocess else image
            pil_img = Image.fromarray(img)
            
            # OCR with optimized config
            config = f'--psm {psm} --oem 3'  # OEM 3 uses LSTM only (fastest)
            data = pytesseract.image_to_data(
                pil_img, lang=self.lang_string,
                output_type=pytesseract.Output.DICT, config=config
            )
            
            # Vectorized filtering for speed
            conf_array = np.array(data['conf'], dtype=float)
            valid_mask = conf_array > (min_confidence * 100)
            
            results = []
            for i in np.where(valid_mask)[0]:
                text = data['text'][i].strip()
                if text:
                    results.append({
                        'text': text,
                        'bbox': [
                            data['left'][i], data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ],
                        'confidence': conf_array[i] / 100.0,
                        'language': self._detect_language_fast(text)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return []
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def _detect_language_fast(text: str) -> str:
        """Cached language detection using character ranges"""
        if not text:
            return 'unknown'
        
        # Use sets for O(1) lookup
        chars = set(ord(c) for c in text if c.isalpha())
        if not chars:
            return 'unknown'
        
        devanagari = sum(1 for c in chars if 0x0900 <= c <= 0x097F)
        gujarati = sum(1 for c in chars if 0x0A80 <= c <= 0x0AFF)
        ascii_alpha = sum(1 for c in chars if c < 128)
        
        total = len(chars)
        thresholds = [(devanagari, 'hin'), (gujarati, 'guj'), (ascii_alpha, 'eng')]
        
        lang = max(thresholds, key=lambda x: x[0])
        return lang[1] if lang[0] / total > 0.3 else 'mixed'
    
    def extract_from_image(self, image_path: str, return_full_text: bool = True) -> Dict[str, Any]:
        """Extract text from image file with optional full text"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read: {image_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.extract_text(img)
            
            output = {
                'image_path': image_path,
                'total_blocks': len(results),
                'text_blocks': results,
                'avg_confidence': float(np.mean([r['confidence'] for r in results])) if results else 0.0,
                'languages': list(set(r['language'] for r in results))
            }
            
            if return_full_text:
                try:
                    output['full_text'] = pytesseract.image_to_string(img, lang=self.lang_string)
                except:
                    output['full_text'] = ' '.join(r['text'] for r in results)
            
            return output
            
        except Exception as e:
            logger.error(f"Error: {image_path}: {e}")
            return {'image_path': image_path, 'error': str(e), 'text_blocks': []}
    
    def extract_from_pdf(self, pdf_path: str, dpi: int = 200, 
                         max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with parallel processing
        
        Args:
            pdf_path: Path to PDF
            dpi: Resolution for conversion
            max_pages: Limit number of pages (None = all)
        """
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path, dpi=dpi, 
                last_page=max_pages, 
                fmt='jpeg',  # JPEG is faster than PNG
                thread_count=self.num_workers
            )
            
            # Process pages in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(
                    lambda img: self.extract_text(np.array(img)),
                    images
                ))
            
            return [
                {
                    'page': i + 1,
                    'text_blocks': result,
                    'full_text': ' '.join(r['text'] for r in result)
                }
                for i, result in enumerate(results)
            ]
            
        except Exception as e:
            logger.error(f"PDF error: {e}")
            return []
    
    def batch_process(self, image_paths: List[str], 
                      parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple images efficiently
        
        Args:
            image_paths: List of image paths
            parallel: Use parallel processing
        """
        if parallel and len(image_paths) > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                return list(executor.map(self.extract_from_image, image_paths))
        else:
            return [self.extract_from_image(path) for path in image_paths]
    
    def extract_region(self, image: np.ndarray, bbox: List[float]) -> str:
        """Extract text from specific region (optimized for speed)"""
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return ""
        
        # Use simpler PSM for regions
        results = self.extract_text(cropped, psm=7)  # PSM 7 = single text line
        return ' '.join(r['text'] for r in results)
    
    def visualize(self, image: np.ndarray, results: List[Dict],
                  output_path: Optional[str] = None, 
                  show_confidence: bool = True) -> np.ndarray:
        """Visualize OCR results"""
        vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        for r in results:
            x1, y1, x2, y2 = map(int, r['bbox'])
            color = (0, 255, 0) if r['confidence'] > 0.7 else (255, 165, 0)
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            if show_confidence:
                label = f"{r['text'][:15]}... {r['confidence']:.0%}"
                cv2.putText(vis, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        return vis


# Convenience function
def get_ocr_processor(languages: List[str] = None) -> EfficientTesseractOCR:
    """Get OCR processor instance (singleton)"""
    return EfficientTesseractOCR(languages=languages)


if __name__ == "__main__":
    print("Testing Efficient OCR Processor...\n")
    
    ocr = get_ocr_processor()
    
    if not TESSERACT_AVAILABLE:
        print("✗ Install pytesseract: pip install pytesseract")
        exit(1)
    
    # Create test image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    test_texts = [
        ("Test Invoice", 50, 80, 2),
        ("Dealer: ABC Tractors", 50, 160, 1),
        ("Model: 575 DI", 50, 210, 1),
        ("HP: 50", 50, 260, 1),
        ("Cost: ₹5,25,000", 50, 310, 1)
    ]
    
    for text, x, y, scale in test_texts:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
    
    cv2.imwrite("test_ocr.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Test extraction
    import time
    start = time.time()
    results = ocr.extract_text(img, min_confidence=0.5)
    elapsed = time.time() - start
    
    print(f"⏱️  Processed in {elapsed:.3f}s")
    print(f"📄 Found {len(results)} text blocks:\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['text']:<30} [{r['confidence']:.0%}] ({r['language']})")
    
    # Test batch processing
    print("\n\n📦 Testing batch processing...")
    batch_start = time.time()
    batch_results = ocr.batch_process(["test_ocr.jpg"] * 3, parallel=True)
    batch_elapsed = time.time() - batch_start
    print(f"⏱️  Processed 3 images in {batch_elapsed:.3f}s")
    
    print("\n✅ All tests completed!")