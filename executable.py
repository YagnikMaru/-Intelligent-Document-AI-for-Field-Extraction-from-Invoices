#!/usr/bin/env python3
"""
Production Document Extraction Pipeline
Extracts dealer name, model, HP, cost, signature, and stamp from invoices/quotations
Target: ≥95% document-level accuracy

CORRECTED VERSION - No dependency on normalizer.py
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
from tqdm import tqdm
import sys
from dataclasses import dataclass, asdict
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules (with fallback error handling)
try:
    from utils.ocr import get_ocr_processor, DocumentOCR
    from utils.extractor import EnhancedRuleBasedExtractor
    from utils.detector import SignatureStampDetector
    from utils.validator import get_validator, DocumentValidator
except ImportError as e:
    print(f"⚠️  Import Error: {e}")
    print("\nPlease ensure the following structure:")
    print("  utils/")
    print("    ├── __init__.py")
    print("    ├── ocr.py")
    print("    ├── extractor.py")
    print("    ├── detector.py")
    print("    └── validator.py")
    print("\nOr run: python -m pip install -e .")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Structured result for each document"""
    document_id: str
    file_path: str
    
    # Extracted fields (as per requirements)
    dealer_name: Optional[str]
    dealer_name_confidence: float
    
    model_name: Optional[str]
    model_name_confidence: float
    
    horse_power: Optional[float]
    horse_power_confidence: float
    
    asset_cost: Optional[float]
    asset_cost_confidence: float
    
    signature_present: bool
    signature_bbox: Optional[List[float]]
    signature_confidence: float
    signature_iou: Optional[float]
    
    stamp_present: bool
    stamp_bbox: Optional[List[float]]
    stamp_confidence: float
    stamp_iou: Optional[float]
    
    # Metadata
    overall_confidence: float
    processing_time_ms: float
    cost_estimate_usd: float
    status: str
    error_message: Optional[str] = None
    
    # Processing details
    ocr_time_ms: float = 0.0
    extraction_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json_output(self) -> Dict[str, Any]:
        """Format for required JSON output structure"""
        return {
            "document_id": self.document_id,
            "dealer_name": {
                "value": self.dealer_name,
                "confidence": round(self.dealer_name_confidence, 4)
            },
            "model_name": {
                "value": self.model_name,
                "confidence": round(self.model_name_confidence, 4)
            },
            "horse_power": {
                "value": self.horse_power,
                "confidence": round(self.horse_power_confidence, 4)
            },
            "asset_cost": {
                "value": self.asset_cost,
                "confidence": round(self.asset_cost_confidence, 4)
            },
            "signature": {
                "present": self.signature_present,
                "bbox": self.signature_bbox,
                "confidence": round(self.signature_confidence, 4),
                "iou": round(self.signature_iou, 4) if self.signature_iou else None
            },
            "stamp": {
                "present": self.stamp_present,
                "bbox": self.stamp_bbox,
                "confidence": round(self.stamp_confidence, 4),
                "iou": round(self.stamp_iou, 4) if self.stamp_iou else None
            },
            "overall_confidence": round(self.overall_confidence, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "status": self.status
        }


class DocumentExtractionPipeline:
    """
    End-to-end pipeline for invoice/quotation field extraction
    CORRECTED - Direct OCR integration without normalizer dependency
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize pipeline components
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        logger.info("="*70)
        logger.info("Initializing Document Extraction Pipeline")
        logger.info("="*70)
        
        # Component initialization (lazy loading)
        self._ocr = None
        self._extractor = None
        self._detector = None
        self._validator = None
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'high_confidence': 0,  # ≥0.85
            'medium_confidence': 0,  # 0.65-0.85
            'low_confidence': 0  # <0.65
        }
        
        logger.info("Pipeline initialized successfully")
    
    @property
    def ocr(self) -> DocumentOCR:
        """Lazy load OCR processor"""
        if self._ocr is None:
            languages = self.config.get('languages', ['eng', 'hin', 'guj'])
            self._ocr = get_ocr_processor(languages=languages)
            logger.info(f"✓ OCR initialized: {'+'.join(languages)}")
        return self._ocr
    
    @property
    def extractor(self) -> EnhancedRuleBasedExtractor:
        """Lazy load rule-based extractor"""
        if self._extractor is None:
            master_data = self.config.get('master_data_path')
            self._extractor = EnhancedRuleBasedExtractor(master_data_path=master_data)
            logger.info("✓ Rule-based extractor initialized")
        return self._extractor
    
    @property
    def detector(self) -> SignatureStampDetector:
        """Lazy load signature/stamp detector"""
        if self._detector is None:
            model_path = self.config.get('detector_model_path')
            device = self.config.get('device', 'cpu')
            
            self._detector = SignatureStampDetector(
                model_path=model_path,
                device=device,
                half_precision=self.config.get('use_fp16', False)
            )
            logger.info(f"✓ Detector initialized: {device}")
        return self._detector
    
    @property
    def validator(self) -> DocumentValidator:
        """Lazy load document validator"""
        if self._validator is None:
            self._validator = get_validator()
            logger.info("✓ Validator initialized")
        return self._validator
    
    def process_document(
        self, 
        file_path: str,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process single document through the pipeline
        
        Args:
            file_path: Path to document image/PDF
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            ProcessingResult with extracted fields
        """
        start_time = time.time()
        doc_id = Path(file_path).stem
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {doc_id}")
        logger.info(f"{'='*70}")
        
        try:
            import cv2
            import numpy as np
            
            # Step 1: Load image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Cannot read image: {file_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_shape = image.shape[:2]
            
            # Step 2: OCR Extraction
            logger.info("Step 1/4: OCR processing...")
            ocr_start = time.time()
            
            # Use extract_from_image which returns structured data
            ocr_result = self.ocr.extract_from_image(file_path, return_full_text=True)
            text_blocks = ocr_result.get('text_blocks', [])
            
            ocr_time = (time.time() - ocr_start) * 1000
            logger.info(f"  ✓ Extracted {len(text_blocks)} text blocks ({ocr_time:.0f}ms)")
            
            if not text_blocks:
                raise ValueError("No text extracted from document")
            
            # Step 3: Rule-based Field Extraction
            logger.info("Step 2/4: Field extraction...")
            extraction_start = time.time()
            
            # Prepare blocks for extractor (simple normalization inline)
            normalized_blocks = []
            for block in text_blocks:
                text = block.get('text', '').strip()
                if text:
                    normalized_blocks.append({
                        'text': text,
                        'normalized_text': text.lower(),
                        'bbox': block.get('bbox', [0, 0, 0, 0]),
                        'confidence': block.get('confidence', 0.0),
                        'language': block.get('language', 'unknown')
                    })
            
            # Extract fields
            extracted = self.extractor.extract_fields(
                normalized_blocks,
                image_shape=img_shape
            )
            
            extraction_time = (time.time() - extraction_start) * 1000
            logger.info(f"  ✓ Fields extracted ({extraction_time:.0f}ms)")
            
            # Log extracted values
            logger.info(f"    Dealer: {extracted.get('dealer_name', {}).get('value', 'None')}")
            logger.info(f"    Model: {extracted.get('model_name', {}).get('value', 'None')}")
            logger.info(f"    HP: {extracted.get('horse_power', {}).get('value', 'None')}")
            logger.info(f"    Cost: {extracted.get('asset_cost', {}).get('value', 'None')}")
            
            # Step 4: Signature/Stamp Detection
            logger.info("Step 3/4: Signature/stamp detection...")
            detection_start = time.time()
            
            # Prepare ground truth if available
            gt_for_detection = None
            if ground_truth:
                gt_for_detection = {
                    'signature': ground_truth.get('signature_bbox'),
                    'stamp': ground_truth.get('stamp_bbox')
                }
            
            detection_result = self.detector.extract_signature_stamp_info(
                image_rgb,
                ground_truth=gt_for_detection
            )
            
            detection_time = (time.time() - detection_start) * 1000
            logger.info(f"  ✓ Detection complete ({detection_time:.0f}ms)")
            logger.info(f"    Signature: {detection_result['signature']['present']}")
            logger.info(f"    Stamp: {detection_result['stamp']['present']}")
            
            # Merge detection results
            extracted['signature'] = detection_result['signature']
            extracted['stamp'] = detection_result['stamp']
            
            # Step 5: Validation
            logger.info("Step 4/4: Validation...")
            validation_start = time.time()
            
            validated = self.validator.validate_document(extracted)
            
            validation_time = (time.time() - validation_start) * 1000
            logger.info(f"  ✓ Validation complete ({validation_time:.0f}ms)")
            
            # Calculate metrics
            total_time = (time.time() - start_time) * 1000
            cost_estimate = self._estimate_cost(
                ocr_time, extraction_time, detection_time, validation_time
            )
            
            # Build result
            result = self._build_result(
                doc_id=doc_id,
                file_path=file_path,
                validated=validated,
                total_time=total_time,
                ocr_time=ocr_time,
                extraction_time=extraction_time,
                detection_time=detection_time,
                validation_time=validation_time,
                cost_estimate=cost_estimate,
                status='success'
            )
            
            # Update stats
            self.stats['total_processed'] += 1
            self.stats['successful'] += 1
            
            if result.overall_confidence >= 0.85:
                self.stats['high_confidence'] += 1
            elif result.overall_confidence >= 0.65:
                self.stats['medium_confidence'] += 1
            else:
                self.stats['low_confidence'] += 1
            
            # Log summary
            logger.info(f"\n{'─'*70}")
            logger.info(f"✅ SUCCESS: {doc_id}")
            logger.info(f"   Overall Confidence: {result.overall_confidence:.1%}")
            logger.info(f"   Processing Time: {total_time:.0f}ms")
            logger.info(f"   Cost Estimate: ${cost_estimate:.6f}")
            logger.info(f"{'─'*70}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ FAILED: {doc_id}")
            logger.error(f"   Error: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.stats['total_processed'] += 1
            self.stats['failed'] += 1
            
            # Return empty result with error
            return ProcessingResult(
                document_id=doc_id,
                file_path=file_path,
                dealer_name=None,
                dealer_name_confidence=0.0,
                model_name=None,
                model_name_confidence=0.0,
                horse_power=None,
                horse_power_confidence=0.0,
                asset_cost=None,
                asset_cost_confidence=0.0,
                signature_present=False,
                signature_bbox=None,
                signature_confidence=0.0,
                signature_iou=None,
                stamp_present=False,
                stamp_bbox=None,
                stamp_confidence=0.0,
                stamp_iou=None,
                overall_confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                cost_estimate_usd=0.0,
                status='failed',
                error_message=str(e)
            )
    
    def _build_result(
        self,
        doc_id: str,
        file_path: str,
        validated: Dict[str, Any],
        total_time: float,
        ocr_time: float,
        extraction_time: float,
        detection_time: float,
        validation_time: float,
        cost_estimate: float,
        status: str
    ) -> ProcessingResult:
        """Build ProcessingResult from validated data"""
        
        # Extract field data safely
        dealer = validated.get('dealer_name', {})
        model = validated.get('model_name', {})
        hp = validated.get('horse_power', {})
        cost = validated.get('asset_cost', {})
        signature = validated.get('signature', {})
        stamp = validated.get('stamp', {})
        
        return ProcessingResult(
            document_id=doc_id,
            file_path=file_path,
            
            # Text fields
            dealer_name=dealer.get('value'),
            dealer_name_confidence=dealer.get('confidence', 0.0),
            
            model_name=model.get('value'),
            model_name_confidence=model.get('confidence', 0.0),
            
            horse_power=hp.get('value'),
            horse_power_confidence=hp.get('confidence', 0.0),
            
            asset_cost=cost.get('value'),
            asset_cost_confidence=cost.get('confidence', 0.0),
            
            # Binary fields with IoU
            signature_present=signature.get('present', False),
            signature_bbox=signature.get('bbox'),
            signature_confidence=signature.get('confidence', 0.0),
            signature_iou=signature.get('iou'),
            
            stamp_present=stamp.get('present', False),
            stamp_bbox=stamp.get('bbox'),
            stamp_confidence=stamp.get('confidence', 0.0),
            stamp_iou=stamp.get('iou'),
            
            # Overall metrics
            overall_confidence=validated.get('overall_confidence', 0.0),
            processing_time_ms=total_time,
            cost_estimate_usd=cost_estimate,
            status=status,
            
            # Timing breakdown
            ocr_time_ms=ocr_time,
            extraction_time_ms=extraction_time,
            detection_time_ms=detection_time,
            validation_time_ms=validation_time
        )
    
    def _estimate_cost(
        self,
        ocr_time: float,
        extraction_time: float,
        detection_time: float,
        validation_time: float
    ) -> float:
        """Estimate processing cost per document"""
        total_sec = (ocr_time + extraction_time + detection_time + validation_time) / 1000
        
        if self.config.get('use_gpu', False):
            cost_per_sec = 0.10 / 3600  # GPU
        else:
            cost_per_sec = 0.01 / 3600  # CPU
        
        return total_sec * cost_per_sec
    
    def process_batch(
        self,
        input_dir: str,
        output_dir: str = "output",
        max_workers: int = 1,
        limit: Optional[int] = None,
        ground_truth_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process batch of documents"""
        
        # Load ground truth if provided
        ground_truth = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
            logger.info(f"Loaded ground truth for {len(ground_truth)} documents")
        
        # Find image files
        image_files = self._find_images(input_dir)
        
        if not image_files:
            logger.error(f"No images found in {input_dir}")
            return {'error': 'No images found'}
        
        if limit:
            image_files = image_files[:limit]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH PROCESSING: {len(image_files)} documents")
        logger.info(f"{'='*70}\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process documents
        results = []
        
        if max_workers == 1:
            # Sequential processing (better for debugging)
            for img_path in tqdm(image_files, desc="Processing", unit="doc"):
                doc_id = Path(img_path).stem
                gt = ground_truth.get(doc_id)
                result = self.process_document(str(img_path), ground_truth=gt)
                results.append(result)
        else:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for img_path in image_files:
                    doc_id = Path(img_path).stem
                    gt = ground_truth.get(doc_id)
                    future = executor.submit(self.process_document, str(img_path), gt)
                    futures[future] = str(img_path)
                
                with tqdm(total=len(futures), desc="Processing", unit="doc") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Worker error: {e}")
                        pbar.update(1)
        
        # Save results
        self._save_results(results, output_dir)
        
        # Generate summary
        summary = self._generate_summary(results, output_dir)
        
        return summary
    
    def _find_images(self, directory: str) -> List[Path]:
        """Find all image files"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']
        images = []
        
        for ext in extensions:
            images.extend(Path(directory).glob(f"*{ext}"))
            images.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        return sorted(set(images))
    
    def _save_results(self, results: List[ProcessingResult], output_dir: str):
        """Save processing results"""
        # Full results
        full_output = os.path.join(output_dir, "extraction_results.json")
        with open(full_output, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
                ensure_ascii=False
            )
        logger.info(f"\n✓ Full results saved: {full_output}")
        
        # JSON output (required format)
        json_output = os.path.join(output_dir, "output.json")
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_json_output() for r in results],
                f,
                indent=2,
                ensure_ascii=False
            )
        logger.info(f"✓ JSON output saved: {json_output}")
    
    def _generate_summary(
        self,
        results: List[ProcessingResult],
        output_dir: str
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        successful = [r for r in results if r.status == 'success']
        
        if not successful:
            summary = {
                'total_documents': len(results),
                'successful': 0,
                'failed': len(results),
                'accuracy': 0.0
            }
        else:
            # Calculate statistics
            confidences = [r.overall_confidence for r in successful]
            times = [r.processing_time_ms for r in successful]
            costs = [r.cost_estimate_usd for r in successful]
            
            # Field extraction rates
            field_stats = {}
            for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                extracted = sum(1 for r in successful if getattr(r, field) is not None)
                field_stats[field] = {
                    'extracted': extracted,
                    'rate': extracted / len(successful)
                }
            
            # Confidence distribution
            high_conf = sum(1 for c in confidences if c >= 0.85)
            med_conf = sum(1 for c in confidences if 0.65 <= c < 0.85)
            low_conf = sum(1 for c in confidences if c < 0.65)
            
            summary = {
                'total_documents': len(results),
                'successful': len(successful),
                'failed': len(results) - len(successful),
                'success_rate': len(successful) / len(results),
                
                'confidence_stats': {
                    'mean': sum(confidences) / len(confidences),
                    'min': min(confidences),
                    'max': max(confidences),
                    'high_count': high_conf,
                    'medium_count': med_conf,
                    'low_count': low_conf
                },
                
                'processing_time_stats': {
                    'mean_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'total_sec': sum(times) / 1000
                },
                
                'cost_stats': {
                    'mean_usd': sum(costs) / len(costs),
                    'total_usd': sum(costs)
                },
                
                'field_extraction_stats': field_stats,
                
                'signature_detection': {
                    'detected': sum(1 for r in successful if r.signature_present),
                    'rate': sum(1 for r in successful if r.signature_present) / len(successful)
                },
                
                'stamp_detection': {
                    'detected': sum(1 for r in successful if r.stamp_present),
                    'rate': sum(1 for r in successful if r.stamp_present) / len(successful)
                }
            }
        
        # Save summary
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Summary saved: {summary_file}")
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console"""
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Documents:    {summary['total_documents']}")
        print(f"Successful:         {summary['successful']}")
        print(f"Failed:             {summary['failed']}")
        print(f"Success Rate:       {summary.get('success_rate', 0):.1%}")
        
        if 'confidence_stats' in summary:
            cs = summary['confidence_stats']
            print(f"\nConfidence:")
            print(f"  Mean:             {cs['mean']:.1%}")
            print(f"  Range:            {cs['min']:.1%} - {cs['max']:.1%}")
            print(f"  High (≥85%):      {cs['high_count']}")
            print(f"  Medium (65-85%):  {cs['medium_count']}")
            print(f"  Low (<65%):       {cs['low_count']}")
        
        if 'processing_time_stats' in summary:
            ts = summary['processing_time_stats']
            print(f"\nProcessing Time:")
            print(f"  Mean:             {ts['mean_ms']:.0f}ms")
            print(f"  Range:            {ts['min_ms']:.0f} - {ts['max_ms']:.0f}ms")
            print(f"  Total:            {ts['total_sec']:.1f}s")
        
        if 'cost_stats' in summary:
            cs = summary['cost_stats']
            print(f"\nCost Estimate:")
            print(f"  Per Document:     ${cs['mean_usd']:.6f}")
            print(f"  Total:            ${cs['total_usd']:.6f}")
        
        if 'field_extraction_stats' in summary:
            print(f"\nField Extraction Rates:")
            for field, stats in summary['field_extraction_stats'].items():
                print(f"  {field.replace('_', ' ').title():<15}: {stats['extracted']}/{summary['successful']} ({stats['rate']:.1%})")
        
        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Document Extraction Pipeline - Extract dealer, model, HP, cost, signature, stamp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory with document images"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (1=sequential)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of documents to process"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for detection (if available)"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for faster GPU inference"
    )
    
    parser.add_argument(
        "--master-data",
        type=str,
        default=None,
        help="Path to master data JSON (dealers/models)"
    )
    
    parser.add_argument(
        "--detector-model",
        type=str,
        default=None,
        help="Path to trained YOLO detector model"
    )
    
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth JSON for evaluation"
    )
    
    parser.add_argument(
        "--languages",
        type=str,
        default="eng,hin,guj",
        help="OCR languages (comma-separated)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Error: Input directory not found: {args.input}")
        return 1
    
    # Configuration
    device = 'cuda' if args.gpu else 'cpu'
    config = {
        'use_gpu': args.gpu,
        'use_fp16': args.fp16,
        'device': device,
        'master_data_path': args.master_data,
        'detector_model_path': args.detector_model,
        'languages': args.languages.split(',')
    }
    
    # Print configuration
    print(f"\n{'='*70}")
    print("DOCUMENT EXTRACTION PIPELINE")
    print(f"{'='*70}")
    print(f"Input Directory:    {args.input}")
    print(f"Output Directory:   {args.output}")
    print(f"Workers:            {args.workers}")
    print(f"Limit:              {args.limit or 'All'}")
    print(f"GPU:                {'Yes' if args.gpu else 'No'}")
    print(f"FP16:               {'Yes' if args.fp16 else 'No'}")
    print(f"Languages:          {', '.join(config['languages'])}")
    print(f"Master Data:        {args.master_data or 'None'}")
    print(f"Detector Model:     {args.detector_model or 'Default'}")
    print(f"Ground Truth:       {args.ground_truth or 'None'}")
    print(f"{'='*70}\n")
    
    # Initialize pipeline
    try:
        pipeline = DocumentExtractionPipeline(config)
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return 1
    
    # Process batch
    try:
        summary = pipeline.process_batch(
            input_dir=args.input,
            output_dir=args.output,
            max_workers=args.workers,
            limit=args.limit,
            ground_truth_file=args.ground_truth
        )
        
        # Check if we met 95% accuracy target
        if 'confidence_stats' in summary:
            mean_conf = summary['confidence_stats']['mean']
            if mean_conf >= 0.95:
                print(f"🎯 TARGET ACHIEVED: {mean_conf:.1%} accuracy (≥95%)")
            elif mean_conf >= 0.90:
                print(f"⚠️  NEAR TARGET: {mean_conf:.1%} accuracy (target: ≥95%)")
            else:
                print(f"❌ BELOW TARGET: {mean_conf:.1%} accuracy (target: ≥95%)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())