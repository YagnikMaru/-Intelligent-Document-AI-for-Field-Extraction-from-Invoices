#!/usr/bin/env python3
"""
Optimized Document AI Pipeline
Processes invoice images with improved efficiency and variable batch sizes
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures
from tqdm import tqdm
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache
import gc

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ocr import get_ocr_processor
from utils.normalizer import get_normalizer
from utils.extractor import get_extractor
from utils.detector import get_detector
from utils.vlm_fallback import get_vlm_processor, should_use_vlm_fallback, merge_results
from utils.validator import EfficientDocumentValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Data class for processing results"""
    doc_id: str
    fields: Dict[str, Any]
    confidence: float
    processing_time_sec: float
    cost_estimate_usd: float
    processing_details: Dict[str, Any]
    status: str
    error: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class ImageCache:
    """Simple LRU cache for preprocessed images"""
    def __init__(self, max_size: int = 10):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: str):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()


class DocumentAIPipeline:
    """Optimized pipeline for document processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pipeline with components"""
        self.config = config or {}
        self.image_cache = ImageCache(max_size=self.config.get('cache_size', 10))
        
        logger.info("Initializing Document AI Pipeline...")
        
        # Initialize components lazily to save memory
        self._ocr_processor = None
        self._normalizer = None
        self._extractor = None
        self._detector = None
        self._vlm_processor = None
        self._validator = None
        
        # Batch processing settings
        self.batch_size = self.config.get('batch_size', None)  # None = process all
        self.checkpoint_interval = self.config.get('checkpoint_interval', 50)
        
        logger.info("Pipeline initialized successfully")
    
    @property
    def ocr_processor(self):
        """Lazy initialization of OCR processor"""
        if self._ocr_processor is None:
            self._ocr_processor = get_ocr_processor(
                languages=self.config.get('languages', ['eng', 'hin', 'guj'])
            )
        return self._ocr_processor
    
    @property
    def normalizer(self):
        """Lazy initialization of normalizer"""
        if self._normalizer is None:
            self._normalizer = get_normalizer()
        return self._normalizer
    
    @property
    def extractor(self):
        """Lazy initialization of extractor"""
        if self._extractor is None:
            master_data_path = self.config.get('master_data_path')
            self._extractor = get_extractor(master_data_path)
        return self._extractor
    
    @property
    def detector(self):
        """Lazy initialization of detector"""
        if self._detector is None:
            detector_model_path = self.config.get('detector_model_path')
            self._detector = get_detector(
                model_path=detector_model_path,
                device='cuda' if self.config.get('use_gpu', False) else 'cpu'
            )
        return self._detector
    
    @property
    def vlm_processor(self):
        """Lazy initialization of VLM processor"""
        if self._vlm_processor is None and self.config.get('enable_vlm', True):
            try:
                processor = get_vlm_processor(
                    model_name=self.config.get('vlm_model', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                    device='cuda' if self.config.get('use_gpu', False) else 'cpu'
                )
                if processor.is_available():
                    self._vlm_processor = processor
                else:
                    logger.warning("VLM model not available")
            except Exception as e:
                logger.warning(f"Could not initialize VLM: {e}")
        return self._vlm_processor
    
    @property
    def validator(self):
        """Lazy initialization of validator"""
        if self._validator is None:
            self._validator = EfficientDocumentValidator()
        return self._validator
    
    def preprocess_image(self, image_path: str, max_size: int = 2048):
        """
        Efficiently preprocess image with resizing for memory optimization
        
        Args:
            image_path: Path to image
            max_size: Maximum dimension size
            
        Returns:
            Tuple of (image, image_rgb, image_pil, dimensions)
        """
        import cv2
        import numpy as np
        from PIL import Image
        
        # Check cache first
        cached = self.image_cache.get(image_path)
        if cached is not None:
            return cached
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Resize if too large to save memory
        if max(orig_height, orig_width) > max_size:
            scale = max_size / max(orig_height, orig_width)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {orig_width}x{orig_height} to {new_width}x{new_height}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        img_height, img_width = image.shape[:2]
        
        result = (image, image_rgb, image_pil, (img_height, img_width))
        
        # Cache if enabled
        if self.config.get('enable_cache', True):
            self.image_cache.put(image_path, result)
        
        return result
    
    def process_single_document(self, image_path: str) -> ProcessingResult:
        """
        Process a single document with optimizations
        
        Args:
            image_path: Path to document image
            
        Returns:
            ProcessingResult object
        """
        start_time = time.time()
        doc_id = os.path.basename(image_path).split('.')[0]
        
        try:
            logger.info(f"Processing document: {doc_id}")
            
            # Step 1: Load and preprocess image (with memory optimization)
            image, image_rgb, image_pil, (img_height, img_width) = self.preprocess_image(
                image_path, 
                max_size=self.config.get('max_image_size', 2048)
            )
            
            # Step 2: OCR Processing
            ocr_start = time.time()
            ocr_results = self.ocr_processor.extract_from_image(image_path)
            ocr_time = time.time() - ocr_start
            
            if 'error' in ocr_results:
                raise ValueError(f"OCR failed: {ocr_results['error']}")
            
            # Step 3: Text Normalization (fast)
            normalized_results = self.normalizer.normalize_ocr_results(
                ocr_results['text_blocks']
            )
            
            # Step 4: Rule-based Extraction
            extraction_start = time.time()
            rule_based_results = self.extractor.extract_fields(
                normalized_results,
                image_shape=(img_height, img_width)
            )
            extraction_time = time.time() - extraction_start
            
            # Step 5: Signature/Stamp Detection (parallel with extraction possible)
            detection_start = time.time()
            signature_stamp_info = self.detector.extract_signature_stamp_info(image_rgb)
            detection_time = time.time() - detection_start
            
            # Update results with detection info
            rule_based_results.update({
                'signature': signature_stamp_info.get('signature', {'present': False, 'bbox': None}),
                'stamp': signature_stamp_info.get('stamp', {'present': False, 'bbox': None})
            })
            
            # Step 6: VLM Fallback (only if needed and available)
            vlm_time = 0
            vlm_used = False
            
            if self.vlm_processor and should_use_vlm_fallback(rule_based_results):
                vlm_start = time.time()
                try:
                    languages = ocr_results.get('languages_detected', ['en'])
                    main_language = languages[0] if languages else 'en'
                    
                    # Limit text length for VLM efficiency
                    vlm_results = self.vlm_processor.extract_fields_vlm(
                        image_pil,
                        ocr_results.get('full_text', '')[:500],  # Reduced from 1000
                        language=main_language
                    )
                    
                    if vlm_results:
                        rule_based_results = merge_results(rule_based_results, vlm_results)
                        vlm_used = True
                    
                except Exception as e:
                    logger.error(f"VLM processing failed: {e}")
                
                vlm_time = time.time() - vlm_start
            
            # Step 7: Validation
            validation_start = time.time()
            validated_results = self.validator.validate_document(rule_based_results)
            validation_time = time.time() - validation_start
            
            # Calculate metrics
            total_time = time.time() - start_time
            cost_estimate = self._calculate_cost_estimate(
                ocr_time, extraction_time, detection_time, vlm_time, validation_time,
                vlm_used=vlm_used
            )
            
            # Prepare result
            result = ProcessingResult(
                doc_id=doc_id,
                fields=self._extract_fields(validated_results),
                confidence=validated_results.get('overall_confidence', 0.0),
                processing_time_sec=round(total_time, 3),
                cost_estimate_usd=round(cost_estimate, 6),
                processing_details={
                    "ocr_time_sec": round(ocr_time, 3),
                    "extraction_time_sec": round(extraction_time, 3),
                    "detection_time_sec": round(detection_time, 3),
                    "vlm_time_sec": round(vlm_time, 3),
                    "validation_time_sec": round(validation_time, 3),
                    "vlm_used": vlm_used
                },
                status="success"
            )
            
            logger.info(f"✅ Processed {doc_id} in {total_time:.2f}s (Confidence: {result.confidence:.2%})")
            
            # Clear image from memory if not cached
            if not self.config.get('enable_cache', True):
                del image, image_rgb, image_pil
                gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process {doc_id}: {e}")
            
            return ProcessingResult(
                doc_id=doc_id,
                fields={
                    "dealer_name": None,
                    "model_name": None,
                    "horse_power": None,
                    "asset_cost": None,
                    "signature": {"present": False, "bbox": None},
                    "stamp": {"present": False, "bbox": None}
                },
                confidence=0.0,
                processing_time_sec=time.time() - start_time,
                cost_estimate_usd=0.0,
                processing_details={},
                status="failed",
                error=str(e)
            )
    
    def _extract_fields(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from validation results"""
        return {
            "dealer_name": results.get('dealer_name', {}).get('value'),
            "model_name": results.get('model_name', {}).get('value'),
            "horse_power": results.get('horse_power', {}).get('value'),
            "asset_cost": results.get('asset_cost', {}).get('value'),
            "signature": {
                "present": results.get('signature', {}).get('present', False),
                "bbox": results.get('signature', {}).get('bbox')
            },
            "stamp": {
                "present": results.get('stamp', {}).get('present', False),
                "bbox": results.get('stamp', {}).get('bbox')
            }
        }
    
    def _calculate_cost_estimate(self, ocr_time: float, extraction_time: float,
                               detection_time: float, vlm_time: float,
                               validation_time: float, vlm_used: bool = False) -> float:
        """Calculate cost estimate for processing"""
        costs = {
            'ocr': ocr_time * 0.001,
            'extraction': extraction_time * 0.0005,
            'detection': detection_time * 0.0003,
            'vlm': vlm_time * 0.005 if vlm_used else 0,
            'validation': validation_time * 0.0001
        }
        return max(sum(costs.values()), 0.0005)
    
    def save_checkpoint(self, results: List[ProcessingResult], output_dir: str, checkpoint_num: int):
        """Save intermediate checkpoint"""
        checkpoint_file = os.path.join(output_dir, f"checkpoint_{checkpoint_num}.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def process_batch(self, input_dir: str, output_dir: str = "sample_output",
                     max_workers: int = 4, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Process batch of documents with optimizations
        
        Args:
            input_dir: Directory containing documents
            output_dir: Directory to save results
            max_workers: Number of parallel workers
            limit: Maximum number of documents to process (None = all)
            
        Returns:
            Batch processing summary
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_files = self._find_image_files(input_dir)
        
        if not image_files:
            logger.error(f"No image files found in {input_dir}")
            return {"error": "No image files found"}
        
        # Apply limit
        if limit:
            image_files = image_files[:limit]
        
        logger.info(f"Found {len(image_files)} documents to process")
        
        # Process documents
        all_results = []
        successful = 0
        failed = 0
        
        # Use ProcessPoolExecutor for better CPU utilization
        executor_class = (concurrent.futures.ProcessPoolExecutor 
                         if self.config.get('use_process_pool', False) 
                         else concurrent.futures.ThreadPoolExecutor)
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_doc = {
                executor.submit(self.process_single_document, str(doc_path)): str(doc_path)
                for doc_path in image_files
            }
            
            # Process results as they complete
            with tqdm(total=len(future_to_doc), desc="Processing documents", 
                     unit="doc", smoothing=0.1) as pbar:
                for i, future in enumerate(concurrent.futures.as_completed(future_to_doc), 1):
                    doc_path = future_to_doc[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        if result.status == 'success':
                            successful += 1
                        else:
                            failed += 1
                        
                        # Save checkpoint periodically
                        if i % self.checkpoint_interval == 0:
                            self.save_checkpoint(all_results, output_dir, i // self.checkpoint_interval)
                            gc.collect()  # Force garbage collection
                            
                    except Exception as e:
                        logger.error(f"Error processing {doc_path}: {e}")
                        failed += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({'Success': successful, 'Failed': failed})
        
        # Save all results
        output_file = os.path.join(output_dir, "all_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2, ensure_ascii=False)
        
        # Generate statistics
        summary = self._create_summary(all_results, successful, failed)
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Create analysis report
        analysis = self._create_analysis_report(all_results)
        analysis_file = os.path.join(output_dir, "analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        self._print_summary(successful, failed, len(all_results), output_file, 
                           summary_file, analysis_file)
        
        return summary
    
    def _find_image_files(self, input_dir: str) -> List[Path]:
        """Find all image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        # Sort for consistent processing order
        return sorted(set(image_files))
    
    def _create_summary(self, results: List[ProcessingResult], 
                       successful: int, failed: int) -> Dict[str, Any]:
        """Create processing summary"""
        successful_results = [r for r in results if r.status == 'success']
        
        if not successful_results:
            return {
                "total_documents": len(results),
                "successful": 0,
                "failed": failed,
                "success_rate": 0
            }
        
        confidences = [r.confidence for r in successful_results]
        processing_times = [r.processing_time_sec for r in successful_results]
        costs = [r.cost_estimate_usd for r in successful_results]
        
        return {
            "total_documents": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) if results else 0,
            "confidence_stats": self._calc_stats(confidences),
            "processing_time_stats": {
                **self._calc_stats(processing_times),
                "total": sum(processing_times)
            },
            "cost_stats": {
                **self._calc_stats(costs),
                "total": sum(costs)
            },
            "field_extraction_stats": self._calculate_field_stats(results)
        }
    
    def _calc_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values"""
        if not values:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values) if len(values) > 1 else 0
        
        return {
            "mean": round(mean, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std": round(variance ** 0.5, 4)
        }
    
    def _calculate_field_stats(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Calculate field extraction statistics"""
        field_stats = {
            field: {"extracted": 0, "null": 0}
            for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        }
        field_stats.update({
            'signature': {"present": 0, "absent": 0},
            'stamp': {"present": 0, "absent": 0}
        })
        
        for result in results:
            if result.status == 'success':
                fields = result.fields
                
                # Text fields
                for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                    if fields.get(field) is not None:
                        field_stats[field]["extracted"] += 1
                    else:
                        field_stats[field]["null"] += 1
                
                # Binary fields
                for field in ['signature', 'stamp']:
                    if fields.get(field, {}).get('present'):
                        field_stats[field]['present'] += 1
                    else:
                        field_stats[field]['absent'] += 1
        
        return field_stats
    
    def _create_analysis_report(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Create detailed analysis report"""
        successful_results = [r for r in results if r.status == 'success']
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        # Analyze by confidence levels
        confidence_levels = {
            "high": [r for r in successful_results if r.confidence >= 0.9],
            "medium": [r for r in successful_results if 0.7 <= r.confidence < 0.9],
            "low": [r for r in successful_results if r.confidence < 0.7]
        }
        
        # VLM usage analysis
        vlm_used = sum(1 for r in successful_results 
                      if r.processing_details.get('vlm_used', False))
        
        return {
            "total_analyzed": len(successful_results),
            "confidence_distribution": {
                level: {
                    "count": len(docs),
                    "percentage": round(len(docs) / len(successful_results), 4)
                }
                for level, docs in confidence_levels.items()
            },
            "vlm_usage": {
                "used": vlm_used,
                "not_used": len(successful_results) - vlm_used,
                "percentage": round(vlm_used / len(successful_results), 4) if successful_results else 0
            },
            "average_processing_time": round(
                sum(r.processing_time_sec for r in successful_results) / len(successful_results), 3
            ),
            "average_cost": round(
                sum(r.cost_estimate_usd for r in successful_results) / len(successful_results), 6
            ),
            "processing_time_distribution": {
                "fast": len([r for r in successful_results if r.processing_time_sec < 2]),
                "medium": len([r for r in successful_results if 2 <= r.processing_time_sec < 5]),
                "slow": len([r for r in successful_results if r.processing_time_sec >= 5])
            }
        }
    
    def _print_summary(self, successful: int, failed: int, total: int,
                      output_file: str, summary_file: str, analysis_file: str):
        """Print processing summary"""
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Batch Processing Complete")
        logger.info(f"   Successful: {successful}/{total} ({successful/total*100:.1f}%)")
        logger.info(f"   Failed: {failed}/{total} ({failed/total*100:.1f}%)")
        logger.info(f"   Results: {output_file}")
        logger.info(f"   Summary: {summary_file}")
        logger.info(f"   Analysis: {analysis_file}")
        logger.info(f"{'='*60}")


def main():
    """Main entry point with enhanced options"""
    parser = argparse.ArgumentParser(
        description="Optimized Document AI Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", type=str, default="train",
                       help="Input directory containing documents")
    parser.add_argument("--output", "-o", type=str, default="sample_output",
                       help="Output directory")
    parser.add_argument("--max_workers", "-w", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--limit", "-l", type=int, default=None,
                       help="Maximum number of documents to process (None = all)")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU if available")
    parser.add_argument("--no_vlm", action="store_true",
                       help="Disable VLM fallback")
    parser.add_argument("--single", type=str,
                       help="Process single document")
    parser.add_argument("--cache_size", type=int, default=10,
                       help="Image cache size")
    parser.add_argument("--max_image_size", type=int, default=2048,
                       help="Maximum image dimension for memory optimization")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                       help="Save checkpoint every N documents")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "use_gpu": args.gpu,
        "enable_vlm": not args.no_vlm,
        "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "cache_size": args.cache_size,
        "max_image_size": args.max_image_size,
        "checkpoint_interval": args.checkpoint_interval,
        "enable_cache": True,
        "use_process_pool": False  # Use threads for better memory sharing
    }
    
    # Initialize pipeline
    pipeline = DocumentAIPipeline(config)
    
    if args.single:
        # Process single document
        if not os.path.exists(args.single):
            print(f"Error: File not found: {args.single}")
            return 1
        
        result = pipeline.process_single_document(args.single)
        
        # Save result
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, "single_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Result saved to: {output_file}")
        print(f"\nExtracted Fields:")
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        
    else:
        # Process batch
        if not os.path.exists(args.input):
            print(f"Error: Input directory not found: {args.input}")
            return 1
        
        print(f"\n{'='*60}")
        print(f"Starting Optimized Batch Processing")
        print(f"{'='*60}")
        print(f"Input:              {args.input}")
        print(f"Output:             {args.output}")
        print(f"Workers:            {args.max_workers}")
        print(f"Limit:              {args.limit or 'All files'}")
        print(f"GPU:                {'Yes' if args.gpu else 'No'}")
        print(f"VLM:                {'Enabled' if not args.no_vlm else 'Disabled'}")
        print(f"Cache Size:         {args.cache_size}")
        print(f"Max Image Size:     {args.max_image_size}px")
        print(f"Checkpoint Every:   {args.checkpoint_interval} docs")
        print(f"{'='*60}\n")
        
        summary = pipeline.process_batch(
            input_dir=args.input,
            output_dir=args.output,
            max_workers=args.max_workers,
            limit=args.limit
        )
        
        print(f"\n📊 Summary Statistics:")
        print(json.dumps(summary, indent=2))
    
    return 0


if __name__ == "__main__":
    exit(main())