import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import torch
import json
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structured detection result for better type safety"""
    bbox: List[float]
    class_name: str
    confidence: float
    area: float
    center: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def calculate_iou(self, other_bbox: List[float]) -> float:
        """Calculate IoU with ground truth bounding box"""
        x1 = max(self.bbox[0], other_bbox[0])
        y1 = max(self.bbox[1], other_bbox[1])
        x2 = min(self.bbox[2], other_bbox[2])
        y2 = min(self.bbox[3], other_bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        area2 = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class SignatureStampDetector:
    """
    YOLOv8-based detector for signatures and stamps in document images.
    Optimized for invoice/quotation processing with IoU-based evaluation support.
    """
    
    # Class-level constants
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
    CLASS_NAMES = {0: 'signature', 1: 'stamp'}
    DEFAULT_CONF_THRESHOLDS = {'signature': 0.4, 'stamp': 0.5}  # Lower for detection
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        train_data_path: Optional[str] = None,
        device: str = 'auto',
        half_precision: bool = False,
        img_size: int = 1024  # Larger for document images
    ):
        """
        Initialize YOLO detector for document signature/stamp detection
        
        Args:
            model_path: Path to trained YOLO model weights
            train_data_path: Path to training images directory
            device: 'auto', 'cpu', or 'cuda'
            half_precision: Use FP16 for faster inference (GPU only)
            img_size: Input image size for YOLO (larger for documents)
        """
        self.device = self._setup_device(device)
        self.train_data_path = Path(train_data_path) if train_data_path else None
        self.half_precision = half_precision and self.device != 'cpu'
        self.img_size = img_size
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Configuration
        self.conf_thresholds = self.DEFAULT_CONF_THRESHOLDS.copy()
        self.iou_threshold = 0.4  # Lower for better recall
        
        # Pre-create CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Cache for optimization
        self._dim_cache = {}
        
        # Metrics tracking
        self.inference_times = []
        
        logger.info(
            f"Detector initialized: device={self.device}, "
            f"fp16={self.half_precision}, img_size={self.img_size}"
        )
    
    @staticmethod
    def _setup_device(device: str) -> str:
        """Setup and validate compute device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, using CPU")
                return 'cpu'
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        
        return 'cpu'
    
    def _load_model(self, model_path: Optional[str]) -> YOLO:
        """Load YOLO model with error handling"""
        try:
            if model_path and os.path.exists(model_path):
                model = YOLO(model_path)
                logger.info(f"Loaded custom model from {model_path}")
            else:
                # Use YOLOv8s for better accuracy on documents
                model = YOLO('yolov8s.pt')
                logger.info("Loaded YOLOv8s pretrained model")
            
            model.to(self.device)
            
            if self.half_precision:
                model.half()
                logger.info("Enabled FP16 inference")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_image_files(self, path: Path) -> List[Path]:
        """Get all valid image files from directory"""
        files = []
        for ext in self.IMAGE_EXTENSIONS:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(files)
    
    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Preprocess document image for better detection
        
        Args:
            image: Input BGR image
            enhance: Apply contrast enhancement
        """
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not enhance:
            return image
        
        # Adaptive contrast enhancement for documents
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def detect(
        self, 
        image: np.ndarray, 
        conf: Optional[float] = None,
        return_timing: bool = False
    ) -> List[DetectionResult]:
        """
        Detect signatures and stamps in document image
        
        Args:
            image: Input image (BGR or RGB)
            conf: Confidence threshold (uses class-specific if None)
            return_timing: Return inference time
            
        Returns:
            List of DetectionResult objects
        """
        import time
        start_time = time.time()
        
        try:
            # Preprocess
            processed_image = self.preprocess_image(image)
            img_height, img_width = processed_image.shape[:2]
            img_area = img_width * img_height
            
            # Run inference
            results = self.model.predict(
                processed_image,
                conf=conf or 0.25,
                iou=self.iou_threshold,
                device=self.device,
                half=self.half_precision,
                verbose=False,
                imgsz=self.img_size,
                augment=False  # No TTA for speed
            )
            
            detections = []
            
            # Process results
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Clip to image bounds
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_width)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_height)
                
                # Calculate areas
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                # Filter and create DetectionResult objects
                for box, conf_val, cls_id, area in zip(boxes, confidences, class_ids, areas):
                    class_name = self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')
                    
                    # Apply class-specific confidence threshold
                    if conf_val < self.conf_thresholds.get(class_name, 0.5):
                        continue
                    
                    # Filter by area (reject very small or very large detections)
                    if area < 0.00005 * img_area or area > 0.6 * img_area:
                        continue
                    
                    detections.append(DetectionResult(
                        bbox=box.tolist(),
                        class_name=class_name,
                        confidence=float(conf_val),
                        area=float(area),
                        center=[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    ))
            
            # Apply NMS
            detections = self._fast_nms(detections)
            
            # Track timing
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            if return_timing:
                return detections, inference_time
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            return [] if not return_timing else ([], 0.0)
    
    def _fast_nms(
        self, 
        detections: List[DetectionResult], 
        iou_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """
        Fast Non-Maximum Suppression using vectorized operations
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Convert to arrays
        boxes = np.array([d.bbox for d in detections])
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        keep = []
        indices = np.arange(len(detections))
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            # Vectorized IoU calculation
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[indices[1:]] - intersection
            iou = intersection / union
            
            # Keep only boxes with IoU below threshold
            indices = indices[1:][iou <= iou_threshold]
        
        return [detections[i] for i in keep]
    
    def extract_signature_stamp_info(
        self, 
        image: np.ndarray,
        ground_truth: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Extract signature and stamp information with IoU evaluation support
        
        Args:
            image: Input document image
            ground_truth: Optional GT bboxes {'signature': [x1,y1,x2,y2], 'stamp': [...]}
            
        Returns:
            Dictionary with detection results and IoU scores if GT provided
        """
        detections = self.detect(image)
        
        results = {
            'signature': {
                'present': False, 
                'bbox': None, 
                'confidence': 0.0, 
                'count': 0,
                'iou': None
            },
            'stamp': {
                'present': False, 
                'bbox': None, 
                'confidence': 0.0, 
                'count': 0,
                'iou': None
            },
            'all_detections': [d.to_dict() for d in detections],
            'evaluation': {'passes_threshold': None}
        }
        
        # Process signatures
        sig_dets = [d for d in detections if d.class_name == 'signature']
        if sig_dets:
            best = max(sig_dets, key=lambda x: x.confidence)
            results['signature'] = {
                'present': True,
                'bbox': best.bbox,
                'confidence': best.confidence,
                'count': len(sig_dets),
                'iou': None
            }
            
            # Calculate IoU if GT provided
            if ground_truth and 'signature' in ground_truth:
                iou = best.calculate_iou(ground_truth['signature'])
                results['signature']['iou'] = iou
        
        # Process stamps
        stamp_dets = [d for d in detections if d.class_name == 'stamp']
        if stamp_dets:
            best = max(stamp_dets, key=lambda x: x.confidence)
            results['stamp'] = {
                'present': True,
                'bbox': best.bbox,
                'confidence': best.confidence,
                'count': len(stamp_dets),
                'iou': None
            }
            
            # Calculate IoU if GT provided
            if ground_truth and 'stamp' in ground_truth:
                iou = best.calculate_iou(ground_truth['stamp'])
                results['stamp']['iou'] = iou
        
        # Overall evaluation (IoU > 0.5 threshold)
        if ground_truth:
            sig_passes = (results['signature']['iou'] or 0) > 0.5 if 'signature' in ground_truth else True
            stamp_passes = (results['stamp']['iou'] or 0) > 0.5 if 'stamp' in ground_truth else True
            results['evaluation']['passes_threshold'] = sig_passes and stamp_passes
        
        return results
    
    def create_yolo_dataset(
        self, 
        annotations_file: Optional[str] = None,
        split_ratio: float = 0.8
    ) -> Optional[str]:
        """
        Create YOLO dataset structure from training images
        
        Args:
            annotations_file: JSON file with annotations
            split_ratio: Train/val split ratio
            
        Returns:
            Path to dataset YAML file
        """
        if not self.train_data_path or not self.train_data_path.exists():
            logger.error("Invalid training data path")
            return None
        
        image_files = self._get_image_files(self.train_data_path)
        logger.info(f"Found {len(image_files)} training images")
        
        if not image_files:
            logger.error("No images found in training directory")
            return None
        
        # Setup directories
        dataset_dir = Path("yolo_dataset")
        dirs = {
            'train_img': dataset_dir / "images" / "train",
            'val_img': dataset_dir / "images" / "val",
            'train_lbl': dataset_dir / "labels" / "train",
            'val_lbl': dataset_dir / "labels" / "val"
        }
        
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # Split data
        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        logger.info(f"Split: {len(train_files)} train, {len(val_files)} val")
        
        # Load annotations
        annotations = self._load_annotations(annotations_file)
        
        # Process splits
        self._process_dataset_split(train_files, dirs['train_img'], dirs['train_lbl'], annotations)
        self._process_dataset_split(val_files, dirs['val_img'], dirs['val_lbl'], annotations)
        
        # Create YAML
        yaml_path = self._create_yaml(dataset_dir, len(train_files), len(val_files))
        
        return str(yaml_path)
    
    @staticmethod
    def _load_annotations(annotations_file: Optional[str]) -> Dict:
        """Load annotations JSON with error handling"""
        if not annotations_file or not os.path.exists(annotations_file):
            logger.warning("No annotations file provided")
            return {}
        
        try:
            with open(annotations_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded annotations for {len(data)} images")
            return data
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return {}
    
    def _process_dataset_split(
        self, 
        files: List[Path], 
        img_dir: Path, 
        lbl_dir: Path, 
        annotations: Dict
    ):
        """Process and copy/symlink dataset split"""
        for img_path in files:
            img_name = img_path.stem
            dest_path = img_dir / img_path.name
            
            # Create symlink or copy
            if not dest_path.exists():
                try:
                    dest_path.symlink_to(img_path.resolve())
                except OSError:
                    import shutil
                    shutil.copy2(img_path, dest_path)
            
            # Create label file if annotations exist
            if img_name in annotations:
                label_path = lbl_dir / f"{img_name}.txt"
                self._create_label_file(label_path, annotations[img_name], img_path)
    
    def _create_label_file(
        self, 
        label_path: Path, 
        label_data: Dict, 
        img_path: Path
    ):
        """Create YOLO format label file"""
        try:
            # Get image dimensions (with caching)
            img_key = str(img_path)
            if img_key in self._dim_cache:
                width, height = self._dim_cache[img_key]
            else:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Cannot read image: {img_path}")
                    return
                height, width = img.shape[:2]
                self._dim_cache[img_key] = (width, height)
            
            lines = []
            
            # Process signatures (class 0)
            for bbox in label_data.get('signatures', []):
                x_center = ((bbox[0] + bbox[2]) / 2) / width
                y_center = ((bbox[1] + bbox[3]) / 2) / height
                w = (bbox[2] - bbox[0]) / width
                h = (bbox[3] - bbox[1]) / height
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # Process stamps (class 1)
            for bbox in label_data.get('stamps', []):
                x_center = ((bbox[0] + bbox[2]) / 2) / width
                y_center = ((bbox[1] + bbox[3]) / 2) / height
                w = (bbox[2] - bbox[0]) / width
                h = (bbox[3] - bbox[1]) / height
                lines.append(f"1 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # Write label file
            if lines:
                label_path.write_text('\n'.join(lines))
                
        except Exception as e:
            logger.error(f"Error creating label {label_path}: {e}")
    
    @staticmethod
    def _create_yaml(dataset_dir: Path, train_count: int, val_count: int) -> Path:
        """Create YOLO dataset configuration YAML"""
        yaml_content = f"""path: {dataset_dir.resolve()}
train: images/train
val: images/val

nc: 2
names:
  0: signature
  1: stamp
"""
        yaml_path = dataset_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content)
        
        logger.info(f"Created dataset YAML: {train_count} train, {val_count} val")
        return yaml_path
    
    def train_on_existing_data(
        self, 
        epochs: int = 100,
        batch_size: int = 16,
        save_path: str = "models/signature_stamp_detector.pt",
        augment: bool = True,
        patience: int = 20
    ) -> bool:
        """
        Train YOLO model on signature/stamp dataset
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            save_path: Where to save trained model
            augment: Use data augmentation
            patience: Early stopping patience
        """
        try:
            # Create dataset
            dataset_yaml = self.create_yolo_dataset()
            
            if not dataset_yaml:
                logger.error("Failed to create dataset")
                return False
            
            logger.info(f"Training model with dataset: {dataset_yaml}")
            
            # Train with optimized settings for documents
            results = self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=self.img_size,
                batch=batch_size,
                device=self.device,
                workers=min(8, mp.cpu_count()),
                save=True,
                save_period=10,
                pretrained=True,
                augment=augment,
                patience=patience,
                cache='ram' if self.device == 'cpu' else False,
                amp=self.device == 'cuda',
                # Document-specific augmentations
                hsv_h=0.01,  # Minimal hue variation
                hsv_s=0.5,   # Some saturation variation
                hsv_v=0.3,   # Some value variation
                degrees=5,   # Small rotation
                translate=0.1,
                scale=0.2,
                mosaic=0.5
            )
            
            # Save model
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            
            logger.info(f"Model training complete. Saved to: {save_path}")
            logger.info(f"Training results: {results}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            return False
    
    def auto_annotate(
        self, 
        output_file: str = "auto_annotations.json",
        conf_threshold: float = 0.6,
        workers: int = 4
    ) -> Dict:
        """
        Auto-annotate images using current model (for semi-supervised learning)
        
        Args:
            output_file: Where to save annotations
            conf_threshold: Minimum confidence for auto-annotation
            workers: Number of parallel workers
        """
        if not self.train_data_path:
            logger.error("No training data path specified")
            return {}
        
        image_files = self._get_image_files(self.train_data_path)
        logger.info(f"Auto-annotating {len(image_files)} images (conf >= {conf_threshold})")
        
        annotations = {}
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._annotate_image, img_path, conf_threshold): img_path 
                for img_path in image_files
            }
            
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result(timeout=30)
                    if result:
                        img_name, ann_data = result
                        annotations[img_name] = ann_data
                    
                    if i % 10 == 0:
                        logger.info(f"Progress: {i}/{len(image_files)}")
                        
                except Exception as e:
                    logger.error(f"Annotation error: {e}")
        
        # Save annotations
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Auto-annotation complete: {len(annotations)}/{len(image_files)} images")
        return annotations
    
    def _annotate_image(
        self, 
        img_path: Path, 
        conf_threshold: float
    ) -> Optional[Tuple[str, Dict]]:
        """Annotate single image"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = self.detect(img_rgb)
            
            # Filter by confidence
            signatures = [
                d.bbox for d in detections 
                if d.class_name == 'signature' and d.confidence >= conf_threshold
            ]
            stamps = [
                d.bbox for d in detections 
                if d.class_name == 'stamp' and d.confidence >= conf_threshold
            ]
            
            if signatures or stamps:
                return (img_path.stem, {
                    'signatures': signatures,
                    'stamps': stamps,
                    'image_path': str(img_path)
                })
            
            return None
            
        except Exception as e:
            logger.error(f"Error annotating {img_path}: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detector performance statistics"""
        if not self.inference_times:
            return {'error': 'No inferences run yet'}
        
        return {
            'total_inferences': len(self.inference_times),
            'mean_time_ms': np.mean(self.inference_times) * 1000,
            'median_time_ms': np.median(self.inference_times) * 1000,
            'min_time_ms': np.min(self.inference_times) * 1000,
            'max_time_ms': np.max(self.inference_times) * 1000,
            'fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }
    
    def set_confidence_thresholds(self, signature: float, stamp: float):
        """Update class-specific confidence thresholds"""
        self.conf_thresholds['signature'] = signature
        self.conf_thresholds['stamp'] = stamp
        logger.info(f"Updated thresholds: signature={signature}, stamp={stamp}")


# Utility functions
def visualize_detections(
    image: np.ndarray, 
    detections: List[DetectionResult],
    save_path: Optional[str] = None
) -> np.ndarray:
    """Visualize detections on image"""
    vis_img = image.copy()
    
    colors = {
        'signature': (0, 255, 0),  # Green
        'stamp': (255, 0, 0)       # Red
    }
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = colors.get(det.class_name, (255, 255, 0))
        
        # Draw box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(
            vis_img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img


if __name__ == "__main__":
    # Example usage
    detector = SignatureStampDetector(
        train_data_path="train",
        device='auto',
        half_precision=True,
        img_size=1024
    )
    
    print("Detector initialized")
    print(f"Device: {detector.device}")
    
    # Auto-annotate
    annotations = detector.auto_annotate(
        output_file="auto_annotations.json",
        conf_threshold=0.6,
        workers=4
    )
    
    print(f"Auto-annotated {len(annotations)} images")
    
    # Create dataset
    dataset_yaml = detector.create_yolo_dataset("auto_annotations.json")
    
    if dataset_yaml:
        print(f"Dataset created: {dataset_yaml}")
        print("\nTo train: detector.train_on_existing_data(epochs=100)")