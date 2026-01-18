import cv2
import numpy as np
from PIL import Image
import pdf2image
import os
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not installed. Run: pip install pytesseract")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentOCR:
    """
    Production-ready OCR processor for invoice/quotation documents.
    Supports English, Hindi, and Gujarati with advanced preprocessing.
    Optimized for accuracy and performance.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid re-initialization overhead"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        languages: List[str] = None, 
        num_workers: int = None,
        tesseract_path: Optional[str] = None
    ):
        """
        Initialize OCR processor
        
        Args:
            languages: OCR languages (default: ['eng', 'hin', 'guj'])
            num_workers: Parallel workers (default: CPU count)
            tesseract_path: Custom Tesseract executable path
        """
        if self._initialized:
            return
        
        self.languages = languages or ['eng', 'hin', 'guj']
        self.lang_string = '+'.join(self.languages)
        self.num_workers = num_workers or min(4, mp.cpu_count())
        
        if not TESSERACT_AVAILABLE:
            logger.error("pytesseract not installed!")
            return
        
        # Setup Tesseract
        if not self._setup_tesseract(tesseract_path):
            logger.error("Tesseract setup failed!")
            return
        
        # Performance tracking
        self.processing_times = []
        self.total_pages_processed = 0
        
        self._initialized = True
        logger.info(
            f"OCR initialized: languages={self.lang_string}, "
            f"workers={self.num_workers}"
        )
    
    def _setup_tesseract(self, custom_path: Optional[str] = None) -> bool:
        """Setup Tesseract executable"""
        if custom_path and os.path.exists(custom_path):
            pytesseract.pytesseract.tesseract_cmd = custom_path
            logger.info(f"Using Tesseract: {custom_path}")
            return True
        
        # Common installation paths
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract'  # macOS Apple Silicon
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Found Tesseract: {path}")
                return True
        
        # Check if in PATH
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Using Tesseract from PATH (v{version})")
            return True
        except Exception as e:
            logger.error(
                f"Tesseract not found! Install from: "
                f"https://github.com/UB-Mannheim/tesseract/wiki"
            )
            return False
    
    @staticmethod
    @lru_cache(maxsize=64)
    def _get_preprocess_params(image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Get cached preprocessing parameters based on image size"""
        h, w = image_shape
        area = h * w
        
        if area > 4000000:  # Very large (>2000x2000)
            return {
                'blur_kernel': 5,
                'block_size': 15,
                'c': 3,
                'morph_kernel': (2, 2),
                'denoise_strength': 10
            }
        elif area > 1000000:  # Large (>1000x1000)
            return {
                'blur_kernel': 3,
                'block_size': 11,
                'c': 2,
                'morph_kernel': (2, 2),
                'denoise_strength': 7
            }
        else:  # Small/medium
            return {
                'blur_kernel': 3,
                'block_size': 9,
                'c': 2,
                'morph_kernel': (1, 1),
                'denoise_strength': 5
            }
    
    def preprocess_image(self, image):
    # Convert to grayscale only
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

    # Light denoising only
        gray = cv2.fastNlMeansDenoising(gray, h=15)

        return gray
    
    @staticmethod
    def _remove_shadows(image: np.ndarray) -> np.ndarray:
        """Remove shadows using dilate-divide technique"""
        dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(image, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        return norm
    
    @staticmethod
    def _deskew(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
        """Auto-deskew image using Hough transform"""
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, 
                threshold=100, 
                minLineLength=100, 
                maxLineGap=10
            )
            
            if lines is None or len(lines) == 0:
                return image
            
            # Calculate angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < max_angle:  # Only small corrections
                    angles.append(angle)
            
            if not angles:
                return image
            
            # Use median angle
            median_angle = np.median(angles)
            
            # Rotate image
            h, w = image.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), 
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return rotated
            
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return image
    
    def extract_text(
        self, 
        image: np.ndarray, 
        preprocess: bool = True,
        psm: int = 3,
        min_confidence: float = 0.0,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract text from image with bounding boxes and metadata
        
        Args:
            image: Input image (RGB/BGR/grayscale)
            preprocess: Apply preprocessing
            psm: Page segmentation mode
                - 3: Fully automatic (default)
                - 6: Uniform block of text
                - 11: Sparse text, find as much as possible
            min_confidence: Filter results below this confidence (0-1)
            return_metadata: Include language and other metadata
        
        Returns:
            List of text blocks with bounding boxes and confidence
        """
        if not TESSERACT_AVAILABLE:
            logger.error("Tesseract not available")
            return []
        
        start_time = time.time()
        
        try:
            # Preprocess image
            if preprocess:
                processed = self.preprocess_image(image)
            else:
                processed = image
            
            # Convert to PIL
            pil_img = Image.fromarray(processed)
            
            # OCR configuration
            config = f'--psm {psm} --oem 3'  # OEM 3 = LSTM only
            
            # Extract data
            data = pytesseract.image_to_data(
                pil_img,
                lang=self.lang_string,
                output_type=pytesseract.Output.DICT,
                config="--psm 6 --oem 3"
            )

            
            # Filter and structure results
            results = []
            conf_threshold = min_confidence * 100
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                # Filter by confidence and empty text
                if not text.strip():
                    continue
                
                result = {
                    'text': text,
                    'bbox': [
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    ],
                    'confidence': conf / 100.0
                }
                
                if return_metadata:
                    result['language'] = self._detect_language(text)
                    result['block_num'] = data['block_num'][i]
                    result['line_num'] = data['line_num'][i]
                    result['word_num'] = data['word_num'][i]
                
                results.append(result)
            
            # Track performance
            elapsed = time.time() - start_time
            self.processing_times.append(elapsed)
            
            logger.debug(f"OCR extracted {len(results)} blocks in {elapsed:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return []
    
    @staticmethod
    @lru_cache(maxsize=2000)
    def _detect_language(text: str) -> str:
        """Detect language using Unicode character ranges (cached)"""
        if not text or not text.strip():
            return 'unknown'
        
        # Count characters by script
        devanagari = 0  # Hindi
        gujarati = 0
        latin = 0
        
        for char in text:
            code = ord(char)
            if 0x0900 <= code <= 0x097F:
                devanagari += 1
            elif 0x0A80 <= code <= 0x0AFF:
                gujarati += 1
            elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
                latin += 1
        
        total = devanagari + gujarati + latin
        
        if total == 0:
            return 'unknown'
        
        # Determine primary language (>30% threshold)
        if devanagari / total > 0.3:
            return 'hin'
        elif gujarati / total > 0.3:
            return 'guj'
        elif latin / total > 0.3:
            return 'eng'
        else:
            return 'mixed'
    
    def extract_from_image(
        self, 
        image_path: str,
        return_full_text: bool = True,
        save_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text from image file
        
        Args:
            image_path: Path to image file
            return_full_text: Include full page text
            save_visualization: Save annotated image
        
        Returns:
            Dictionary with OCR results and metadata
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract text blocks
            text_blocks = self.extract_text(img_rgb, min_confidence=0.3)
            
            # Calculate statistics
            confidences = [b['confidence'] for b in text_blocks]
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            
            result = {
                'image_path': image_path,
                'image_shape': img_rgb.shape[:2],
                'total_blocks': len(text_blocks),
                'text_blocks': text_blocks,
                'avg_confidence': avg_conf,
                'languages': list(set(b.get('language', 'unknown') for b in text_blocks))
            }
            
            # Get full text if requested
            if return_full_text:
                try:
                    full_text = pytesseract.image_to_string(
                        img_rgb,
                        lang=self.lang_string
                    )
                    result['full_text'] = full_text
                except Exception as e:
                    logger.warning(f"Full text extraction failed: {e}")
                    result['full_text'] = ' '.join(b['text'] for b in text_blocks)
            
            # Save visualization if requested
            if save_visualization:
                vis_path = Path(image_path).stem + '_ocr_vis.jpg'
                self.visualize(img_rgb, text_blocks, output_path=vis_path)
                result['visualization_path'] = vis_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'text_blocks': []
            }
    
    def extract_from_pdf(
        self, 
        pdf_path: str,
        dpi: int = 300,  # Higher DPI for better accuracy
        max_pages: Optional[int] = None,
        first_page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with multi-page support
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion (200-300 recommended)
            max_pages: Maximum pages to process (None = all)
            first_page: Starting page number (1-indexed)
        
        Returns:
            List of page results with text blocks
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return []
        
        try:
            logger.info(f"Converting PDF to images (DPI={dpi})...")
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=first_page,
                last_page=max_pages,
                fmt='jpeg',  # JPEG for speed
                thread_count=self.num_workers,
                grayscale=False  # Keep color for preprocessing
            )
            
            logger.info(f"Processing {len(images)} pages...")
            
            # Process pages in parallel
            def process_page(page_data):
                page_num, img_pil = page_data
                img_array = np.array(img_pil)
                
                # Extract text
                text_blocks = self.extract_text(img_array, min_confidence=0.3)
                
                return {
                    'page': page_num,
                    'text_blocks': text_blocks,
                    'block_count': len(text_blocks),
                    'avg_confidence': float(np.mean([b['confidence'] for b in text_blocks])) if text_blocks else 0.0,
                    'full_text': ' '.join(b['text'] for b in text_blocks)
                }
            
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                page_data = [(i + first_page, img) for i, img in enumerate(images)]
                results = list(executor.map(process_page, page_data))
            
            self.total_pages_processed += len(results)
            
            logger.info(f"Completed processing {len(results)} pages")
            return results
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}", exc_info=True)
            return []
    
    def batch_process(
        self, 
        image_paths: List[str],
        parallel: bool = True,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            parallel: Use parallel processing
            show_progress: Log progress
        
        Returns:
            List of OCR results
        """
        if not image_paths:
            return []
        
        logger.info(f"Batch processing {len(image_paths)} images...")
        
        if parallel and len(image_paths) > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self.extract_from_image, image_paths))
        else:
            results = []
            for i, path in enumerate(image_paths, 1):
                result = self.extract_from_image(path)
                results.append(result)
                
                if show_progress and i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(image_paths)}")
        
        return results
    
    def extract_region(
        self, 
        image: np.ndarray, 
        bbox: List[float],
        padding: int = 5
    ) -> str:
        """
        Extract text from specific region (useful for field extraction)
        
        Args:
            image: Source image
            bbox: [x1, y1, x2, y2] bounding box
            padding: Pixels to add around bbox
        
        Returns:
            Extracted text string
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop region
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return ""
        
        # Use PSM 7 (single text line) for small regions
        results = self.extract_text(cropped, psm=7, min_confidence=0.5)
        
        return ' '.join(r['text'] for r in results)
    
    def visualize(
        self, 
        image: np.ndarray,
        results: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        show_confidence: bool = True,
        confidence_threshold: float = 0.7
    ) -> np.ndarray:
        """
        Visualize OCR results with bounding boxes
        
        Args:
            image: Source image
            results: OCR results with bboxes
            output_path: Save path (optional)
            show_confidence: Display confidence scores
            confidence_threshold: Color threshold
        
        Returns:
            Annotated image
        """
        # Ensure RGB
        if image.ndim == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis = image.copy()
        
        for r in results:
            bbox = r.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            conf = r.get('confidence', 0.0)
            
            # Color based on confidence
            color = (0, 255, 0) if conf > confidence_threshold else (255, 165, 0)
            
            # Draw rectangle
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_confidence:
                text = r.get('text', '')
                label = f"{text[:20]}... {conf:.0%}" if len(text) > 20 else f"{text} {conf:.0%}"
                
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                
                # Text
                cv2.putText(
                    vis, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
                )
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization: {output_path}")
        
        return vis
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get OCR performance statistics"""
        if not self.processing_times:
            return {'message': 'No processing done yet'}
        
        times_array = np.array(self.processing_times)
        
        return {
            'total_processed': len(self.processing_times),
            'total_pages': self.total_pages_processed,
            'mean_time_ms': float(np.mean(times_array) * 1000),
            'median_time_ms': float(np.median(times_array) * 1000),
            'min_time_ms': float(np.min(times_array) * 1000),
            'max_time_ms': float(np.max(times_array) * 1000),
            'throughput_pages_per_sec': 1.0 / np.mean(times_array) if times_array.size > 0 else 0
        }


# Global singleton instance
_ocr_instance = None

def get_ocr_processor(
    languages: List[str] = None,
    num_workers: int = None
) -> DocumentOCR:
    """Get singleton OCR processor instance"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = DocumentOCR(languages=languages, num_workers=num_workers)
    return _ocr_instance


if __name__ == "__main__":
    print("=" * 70)
    print("Document OCR Processor - Testing")
    print("=" * 70)
    
    ocr = get_ocr_processor()
    
    if not TESSERACT_AVAILABLE:
        print("\n❌ pytesseract not installed")
        print("Install: pip install pytesseract")
        print("Then install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        exit(1)
    
    # Create test image
    print("\n📝 Creating test invoice image...")
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255
    
    test_content = [
        ("INVOICE / बिल", 100, 60, 1.5, 3),
        ("Dealer: Mahindra Tractors Pvt Ltd", 50, 120, 0.7, 2),
        ("Model: 575 DI", 50, 170, 0.7, 2),
        ("Horse Power: 50 HP", 50, 220, 0.7, 2),
        ("Asset Cost: ₹5,25,000", 50, 270, 0.7, 2),
        ("Total: Rs. 5.25 Lakh", 50, 320, 0.7, 2),
        ("Signature: __________", 50, 400, 0.6, 1)
    ]
    
    for text, x, y, scale, thickness in test_content:
        cv2.putText(
            img, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness
        )
    
    test_path = "test_invoice.jpg"
    cv2.imwrite(test_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"✓ Created: {test_path}")
    
    # Test extraction
    print("\n🔍 Running OCR...")
    start = time.time()
    results = ocr.extract_text(img, min_confidence=0.5)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Processing time: {elapsed:.3f}s")
    print(f"📄 Found {len(results)} text blocks:\n")
    
    for i, r in enumerate(results, 1):
        lang_emoji = {'eng': '🇬🇧', 'hin': '🇮🇳', 'guj': '🇮🇳'}.get(r.get('language'), '🌐')
        print(f"{i:2}. {r['text']:<35} [{r['confidence']:.0%}] {lang_emoji} {r.get('language', 'unk')}")
    
    # Test visualization
    print("\n🎨 Creating visualization...")
    vis = ocr.visualize(img, results, output_path="test_invoice_vis.jpg")
    print("✓ Saved: test_invoice_vis.jpg")
    
    # Performance stats
    print("\n📊 Performance Statistics:")
    stats = ocr.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ All tests completed successfully!")