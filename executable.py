#!/usr/bin/env python3
"""
Main executable for Document AI Pipeline
Processes 500 invoice images and extracts structured data
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import concurrent.futures
from tqdm import tqdm
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ocr import get_ocr_processor
from utils.normalizer import get_normalizer
from utils.extractor import get_extractor
from utils.detector import get_detector
from utils.vlm_fallback import get_vlm_processor, should_use_vlm_fallback, merge_results
from utils.validator import DocumentValidator

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

class DocumentAIPipeline:
    """
    Main pipeline for document processing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pipeline with components
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        logger.info("Initializing Document AI Pipeline...")
        
        # OCR Processor (Multilingual)
        self.ocr_processor = get_ocr_processor(
        languages=['eng', 'hin', 'guj']
        # Remove gpu parameter since Tesseract doesn't use GPU
        )
        
        # Text Normalizer
        self.normalizer = get_normalizer()
        
        # Rule-based Extractor
        master_data_path = self.config.get('master_data_path')
        self.extractor = get_extractor(master_data_path)
        
        # Signature/Stamp Detector
        detector_model_path = self.config.get('detector_model_path')
        self.detector = get_detector(
            model_path=detector_model_path,
            device='cuda' if self.config.get('use_gpu', False) else 'cpu'
        )
        
        # VLM Fallback (Optional)
        self.vlm_processor = None
        if self.config.get('enable_vlm', True):
            try:
                self.vlm_processor = get_vlm_processor(
                    model_name=self.config.get('vlm_model', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                    device='cuda' if self.config.get('use_gpu', False) else 'cpu'
                )
                if not self.vlm_processor.is_available():
                    logger.warning("VLM model not available, continuing without it")
                    self.vlm_processor = None
            except Exception as e:
                logger.warning(f"Could not initialize VLM: {e}")
                self.vlm_processor = None
        
        # Document Validator
        self.validator = DocumentValidator()
        
        logger.info("Pipeline initialized successfully")
    
    def process_single_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single document
        
        Args:
            image_path: Path to document image
            
        Returns:
            Processing results
        """
        start_time = time.time()
        doc_id = os.path.basename(image_path).split('.')[0]
        
        try:
            logger.info(f"Processing document: {doc_id}")
            
            # Step 1: Load and preprocess image
            import cv2
            import numpy as np
            from PIL import Image
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Get image dimensions
            img_height, img_width = image.shape[:2]
            
            # Step 2: OCR Processing
            ocr_start = time.time()
            ocr_results = self.ocr_processor.extract_from_image(image_path)
            ocr_time = time.time() - ocr_start
            
            if 'error' in ocr_results:
                raise ValueError(f"OCR failed: {ocr_results['error']}")
            
            # Step 3: Text Normalization
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
            
            # Step 5: Signature/Stamp Detection
            detection_start = time.time()
            signature_stamp_info = self.detector.extract_signature_stamp_info(image_rgb)
            detection_time = time.time() - detection_start
            
            # Update results with detection info
            if 'signature' in signature_stamp_info:
                rule_based_results['signature'] = signature_stamp_info['signature']
            
            if 'stamp' in signature_stamp_info:
                rule_based_results['stamp'] = signature_stamp_info['stamp']
            
            # Step 6: VLM Fallback (if needed and available)
            vlm_time = 0
            vlm_used = False
            vlm_results = {}
            
            if self.vlm_processor and should_use_vlm_fallback(rule_based_results):
                vlm_start = time.time()
                try:
                    # Determine language for VLM prompt
                    languages = ocr_results.get('languages_detected', ['en'])
                    main_language = languages[0] if languages else 'en'
                    
                    # Extract with VLM
                    vlm_results = self.vlm_processor.extract_fields_vlm(
                        image_pil,
                        ocr_results.get('full_text', '')[:1000],  # Limit text length
                        language=main_language
                    )
                    
                    # Merge VLM results with rule-based results
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
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Calculate cost estimate (simplified)
            cost_estimate = self._calculate_cost_estimate(
                ocr_time, extraction_time, detection_time, vlm_time, validation_time,
                vlm_used=vlm_used
            )
            
            # Prepare final output
            output = self._prepare_output(
                doc_id=doc_id,
                results=validated_results,
                processing_time=total_time,
                cost_estimate=cost_estimate,
                ocr_time=ocr_time,
                extraction_time=extraction_time,
                detection_time=detection_time,
                vlm_time=vlm_time,
                validation_time=validation_time,
                vlm_used=vlm_used
            )
            
            logger.info(f"✅ Processed {doc_id} in {total_time:.2f}s (Confidence: {output['confidence']:.2%})")
            
            return output
            
        except Exception as e:
            logger.error(f"❌ Failed to process {doc_id}: {e}")
            
            # Return error result
            return {
                "doc_id": doc_id,
                "fields": {
                    "dealer_name": None,
                    "model_name": None,
                    "horse_power": None,
                    "asset_cost": None,
                    "signature": {"present": False, "bbox": None},
                    "stamp": {"present": False, "bbox": None}
                },
                "confidence": 0.0,
                "processing_time_sec": time.time() - start_time,
                "cost_estimate_usd": 0.0,
                "error": str(e),
                "status": "failed"
            }
    
    def _calculate_cost_estimate(self, ocr_time: float, extraction_time: float,
                               detection_time: float, vlm_time: float,
                               validation_time: float, vlm_used: bool = False) -> float:
        """
        Calculate cost estimate for processing
        
        Simplified cost model:
        - OCR: $0.001 per second
        - Extraction: $0.0005 per second
        - Detection: $0.0003 per second
        - VLM: $0.005 per second (if used)
        - Validation: $0.0001 per second
        """
        ocr_cost = ocr_time * 0.001
        extraction_cost = extraction_time * 0.0005
        detection_cost = detection_time * 0.0003
        vlm_cost = vlm_time * 0.005 if vlm_used else 0
        validation_cost = validation_time * 0.0001
        
        total_cost = ocr_cost + extraction_cost + detection_cost + vlm_cost + validation_cost
        
        # Add minimum cost
        total_cost = max(total_cost, 0.0005)
        
        return total_cost
    
    def _prepare_output(self, doc_id: str, results: Dict[str, Any],
                       processing_time: float, cost_estimate: float,
                       **kwargs) -> Dict[str, Any]:
        """
        Prepare final JSON output
        
        Args:
            doc_id: Document ID
            results: Extracted results
            processing_time: Total processing time
            cost_estimate: Cost estimate
            **kwargs: Additional timing information
            
        Returns:
            Formatted output
        """
        # Extract fields from results
        fields = {
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
        
        # Get confidence
        confidence = results.get('overall_confidence', 0.0)
        
        # Prepare metadata
        metadata = {
            "doc_id": doc_id,
            "fields": fields,
            "confidence": confidence,
            "processing_time_sec": round(processing_time, 3),
            "cost_estimate_usd": round(cost_estimate, 6),
            "processing_details": {
                "ocr_time_sec": round(kwargs.get('ocr_time', 0), 3),
                "extraction_time_sec": round(kwargs.get('extraction_time', 0), 3),
                "detection_time_sec": round(kwargs.get('detection_time', 0), 3),
                "vlm_time_sec": round(kwargs.get('vlm_time', 0), 3),
                "validation_time_sec": round(kwargs.get('validation_time', 0), 3),
                "vlm_used": kwargs.get('vlm_used', False)
            },
            "status": "success"
        }
        
        return metadata
    
    def process_batch(self, input_dir: str, output_dir: str = "sample_output",
                     max_workers: int = 4) -> Dict[str, Any]:
        """
        Process batch of documents
        
        Args:
            input_dir: Directory containing documents
            output_dir: Directory to save results
            max_workers: Number of parallel workers
            
        Returns:
            Batch processing summary
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
            image_files.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
        
        if not image_files:
            logger.error(f"No image files found in {input_dir}")
            return {"error": "No image files found"}
        
        logger.info(f"Found {len(image_files)} documents to process")
        
        # Process documents
        all_results = []
        successful = 0
        failed = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_doc = {
                executor.submit(self.process_single_document, str(doc_path)): str(doc_path)
                for doc_path in image_files[:500]  # Limit to 500 as per requirement
            }
            
            # Process results as they complete
            with tqdm(total=len(future_to_doc), desc="Processing documents") as pbar:
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc_path = future_to_doc[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        if result.get('status') == 'success':
                            successful += 1
                        else:
                            failed += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing {doc_path}: {e}")
                        failed += 1
                    
                    pbar.update(1)
        
        # Save all results
        output_file = os.path.join(output_dir, "all_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Save summary statistics
        summary = self._create_summary(all_results, successful, failed)
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Create analysis report
        analysis = self._create_analysis_report(all_results)
        analysis_file = os.path.join(output_dir, "analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Batch Processing Complete")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total: {len(all_results)}")
        logger.info(f"   Results saved to: {output_file}")
        logger.info(f"   Summary saved to: {summary_file}")
        logger.info(f"   Analysis saved to: {analysis_file}")
        logger.info(f"{'='*60}")
        
        return summary
    
    def _create_summary(self, results: List[Dict], successful: int, failed: int) -> Dict[str, Any]:
        """Create processing summary"""
        # Calculate statistics
        confidences = [r.get('confidence', 0) for r in results if r.get('status') == 'success']
        processing_times = [r.get('processing_time_sec', 0) for r in results if r.get('status') == 'success']
        costs = [r.get('cost_estimate_usd', 0) for r in results if r.get('status') == 'success']
        
        summary = {
            "total_documents": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) if results else 0,
            "confidence_stats": {
                "mean": sum(confidences) / len(confidences) if confidences else 0,
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0,
                "std": self._calculate_std(confidences) if confidences else 0
            },
            "processing_time_stats": {
                "mean": sum(processing_times) / len(processing_times) if processing_times else 0,
                "min": min(processing_times) if processing_times else 0,
                "max": max(processing_times) if processing_times else 0,
                "total": sum(processing_times) if processing_times else 0
            },
            "cost_stats": {
                "mean": sum(costs) / len(costs) if costs else 0,
                "min": min(costs) if costs else 0,
                "max": max(costs) if costs else 0,
                "total": sum(costs) if costs else 0
            },
            "field_extraction_stats": self._calculate_field_stats(results)
        }
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_field_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate field extraction statistics"""
        field_stats = {
            "dealer_name": {"extracted": 0, "null": 0},
            "model_name": {"extracted": 0, "null": 0},
            "horse_power": {"extracted": 0, "null": 0},
            "asset_cost": {"extracted": 0, "null": 0},
            "signature": {"present": 0, "absent": 0},
            "stamp": {"present": 0, "absent": 0}
        }
        
        for result in results:
            if result.get('status') == 'success':
                fields = result.get('fields', {})
                
                # Text fields
                for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                    if fields.get(field) is not None:
                        field_stats[field]["extracted"] += 1
                    else:
                        field_stats[field]["null"] += 1
                
                # Signature and stamp
                signature = fields.get('signature', {})
                stamp = fields.get('stamp', {})
                
                if signature.get('present'):
                    field_stats['signature']['present'] += 1
                else:
                    field_stats['signature']['absent'] += 1
                
                if stamp.get('present'):
                    field_stats['stamp']['present'] += 1
                else:
                    field_stats['stamp']['absent'] += 1
        
        return field_stats
    
    def _create_analysis_report(self, results: List[Dict]) -> Dict[str, Any]:
        """Create detailed analysis report"""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        # Analyze by confidence levels
        confidence_levels = {
            "high": [r for r in successful_results if r.get('confidence', 0) >= 0.9],
            "medium": [r for r in successful_results if 0.7 <= r.get('confidence', 0) < 0.9],
            "low": [r for r in successful_results if r.get('confidence', 0) < 0.7]
        }
        
        # Analyze VLM usage
        vlm_used = sum(1 for r in successful_results 
                      if r.get('processing_details', {}).get('vlm_used', False))
        
        # Find examples for each confidence level
        examples = {
            "high_confidence_example": next(
                (r for r in confidence_levels["high"]), 
                successful_results[0]
            ),
            "low_confidence_example": next(
                (r for r in confidence_levels["low"]), 
                successful_results[-1]
            )
        }
        
        analysis = {
            "total_analyzed": len(successful_results),
            "confidence_distribution": {
                level: {
                    "count": len(docs),
                    "percentage": len(docs) / len(successful_results)
                }
                for level, docs in confidence_levels.items()
            },
            "vlm_usage": {
                "used": vlm_used,
                "not_used": len(successful_results) - vlm_used,
                "percentage": vlm_used / len(successful_results) if successful_results else 0
            },
            "average_processing_time": sum(
                r.get('processing_time_sec', 0) for r in successful_results
            ) / len(successful_results),
            "average_cost": sum(
                r.get('cost_estimate_usd', 0) for r in successful_results
            ) / len(successful_results),
            "examples": examples
        }
        
        return analysis

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Document AI Pipeline")
    parser.add_argument("--input", "-i", type=str, default="train",
                       help="Input directory containing documents (default: train)")
    parser.add_argument("--output", "-o", type=str, default="sample_output",
                       help="Output directory (default: sample_output)")
    parser.add_argument("--max_workers", "-w", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU if available")
    parser.add_argument("--no_vlm", action="store_true",
                       help="Disable VLM fallback")
    parser.add_argument("--single", type=str,
                       help="Process single document instead of batch")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "use_gpu": args.gpu,
        "enable_vlm": not args.no_vlm,
        "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct"
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
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResult saved to: {output_file}")
        print(f"\nExtracted Fields:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    else:
        # Process batch
        if not os.path.exists(args.input):
            print(f"Error: Input directory not found: {args.input}")
            return 1
        
        print(f"Starting batch processing...")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Workers: {args.max_workers}")
        print(f"GPU: {'Yes' if args.gpu else 'No'}")
        print(f"VLM: {'Enabled' if not args.no_vlm else 'Disabled'}")
        print("=" * 60)
        
        summary = pipeline.process_batch(
            input_dir=args.input,
            output_dir=args.output,
            max_workers=args.max_workers
        )
        
        print(f"\nSummary:")
        print(json.dumps(summary, indent=2))
    
    return 0

if __name__ == "__main__":
    exit(main())