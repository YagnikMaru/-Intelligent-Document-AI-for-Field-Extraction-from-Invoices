#!/usr/bin/env python3
"""
Evaluation and Analysis Script for Document AI Pipeline
Generates comprehensive analysis of processing results
"""

import json
import os
import argparse
import statistics
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentEvaluator:
    """
    Comprehensive evaluation and analysis of document processing results
    """
    
    def __init__(self, results_file: str, ground_truth_file: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            results_file: Path to all_results.json
            ground_truth_file: Path to ground truth annotations (optional)
        """
        self.results_file = results_file
        self.ground_truth_file = ground_truth_file
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # Load ground truth if available
        self.ground_truth = None
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                self.ground_truth = json.load(f)
        
        # Initialize metrics
        self.metrics = {
            'overall': {},
            'field_level': {},
            'confidence_analysis': {},
            'performance_metrics': {},
            'error_analysis': {},
            'cost_analysis': {}
        }
    
    def calculate_document_level_accuracy(self) -> Dict[str, Any]:
        """
        Calculate Document-Level Accuracy (DLA)
        
        Returns:
            DLA metrics
        """
        if not self.ground_truth:
            logger.warning("No ground truth provided, skipping DLA calculation")
            return {"error": "Ground truth required"}
        
        total_docs = len(self.results)
        correct_docs = 0
        partially_correct = 0
        
        for result in self.results:
            doc_id = result['doc_id']
            if doc_id in self.ground_truth:
                gt = self.ground_truth[doc_id]
                is_correct = self._is_document_correct(result, gt)
                
                if is_correct:
                    correct_docs += 1
                elif self._is_partially_correct(result, gt):
                    partially_correct += 1
        
        dla = (correct_docs / total_docs) * 100 if total_docs > 0 else 0
        
        return {
            'total_documents': total_docs,
            'correct_documents': correct_docs,
            'partially_correct': partially_correct,
            'incorrect_documents': total_docs - correct_docs - partially_correct,
            'document_level_accuracy': round(dla, 2),
            'target_accuracy': 95.0,
            'achieved_target': dla >= 95.0
        }
    
    def _is_document_correct(self, result: Dict, ground_truth: Dict) -> bool:
        """Check if all fields are correctly extracted"""
        result_fields = result['fields']
        gt_fields = ground_truth['fields']
        
        # Check each field
        checks = []
        
        # Dealer name (fuzzy match >= 90%)
        if result_fields['dealer_name'] and gt_fields['dealer_name']:
            dealer_similarity = self._calculate_similarity(
                result_fields['dealer_name'],
                gt_fields['dealer_name']
            )
            checks.append(dealer_similarity >= 0.9)
        
        # Model name (exact match)
        if result_fields['model_name'] and gt_fields['model_name']:
            checks.append(
                str(result_fields['model_name']).strip().upper() == 
                str(gt_fields['model_name']).strip().upper()
            )
        
        # Horse power (exact match ±5%)
        if result_fields['horse_power'] and gt_fields['horse_power']:
            hp_result = float(result_fields['horse_power'])
            hp_gt = float(gt_fields['horse_power'])
            tolerance = abs(hp_gt * 0.05)
            checks.append(abs(hp_result - hp_gt) <= tolerance)
        
        # Asset cost (exact match ±5%)
        if result_fields['asset_cost'] and gt_fields['asset_cost']:
            cost_result = float(result_fields['asset_cost'])
            cost_gt = float(gt_fields['asset_cost'])
            tolerance = abs(cost_gt * 0.05)
            checks.append(abs(cost_result - cost_gt) <= tolerance)
        
        # Signature presence
        if 'signature' in result_fields and 'signature' in gt_fields:
            checks.append(
                result_fields['signature']['present'] == 
                gt_fields['signature']['present']
            )
        
        # Stamp presence
        if 'stamp' in result_fields and 'stamp' in gt_fields:
            checks.append(
                result_fields['stamp']['present'] == 
                gt_fields['stamp']['present']
            )
        
        # All checks must pass for document to be correct
        return all(checks)
    
    def _is_partially_correct(self, result: Dict, ground_truth: Dict) -> bool:
        """Check if at least 4 out of 6 fields are correct"""
        result_fields = result['fields']
        gt_fields = ground_truth['fields']
        
        correct_fields = 0
        total_fields = 0
        
        # Dealer name
        if result_fields['dealer_name'] and gt_fields['dealer_name']:
            similarity = self._calculate_similarity(
                result_fields['dealer_name'],
                gt_fields['dealer_name']
            )
            if similarity >= 0.9:
                correct_fields += 1
            total_fields += 1
        
        # Model name
        if result_fields['model_name'] and gt_fields['model_name']:
            if str(result_fields['model_name']).upper() == str(gt_fields['model_name']).upper():
                correct_fields += 1
            total_fields += 1
        
        # Horse power
        if result_fields['horse_power'] and gt_fields['horse_power']:
            hp_result = float(result_fields['horse_power'])
            hp_gt = float(gt_fields['horse_power'])
            if abs(hp_result - hp_gt) <= abs(hp_gt * 0.05):
                correct_fields += 1
            total_fields += 1
        
        # Asset cost
        if result_fields['asset_cost'] and gt_fields['asset_cost']:
            cost_result = float(result_fields['asset_cost'])
            cost_gt = float(gt_fields['asset_cost'])
            if abs(cost_result - cost_gt) <= abs(cost_gt * 0.05):
                correct_fields += 1
            total_fields += 1
        
        # Signature
        if 'signature' in result_fields and 'signature' in gt_fields:
            if result_fields['signature']['present'] == gt_fields['signature']['present']:
                correct_fields += 1
            total_fields += 1
        
        # Stamp
        if 'stamp' in result_fields and 'stamp' in gt_fields:
            if result_fields['stamp']['present'] == gt_fields['stamp']['present']:
                correct_fields += 1
            total_fields += 1
        
        return correct_fields >= 4 and total_fields >= 4
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (0-1)"""
        try:
            from thefuzz import fuzz
            return fuzz.ratio(str(str1).lower(), str(str2).lower()) / 100.0
        except:
            # Fallback simple similarity
            str1 = str(str1).lower()
            str2 = str(str2).lower()
            
            if not str1 or not str2:
                return 0.0
            
            # Tokenize
            tokens1 = set(str1.split())
            tokens2 = set(str2.split())
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            return len(intersection) / len(union)
    
    def calculate_field_level_metrics(self) -> Dict[str, Any]:
        """
        Calculate field-level metrics (precision, recall, F1)
        
        Returns:
            Field-level metrics
        """
        if not self.ground_truth:
            logger.warning("No ground truth provided, skipping field-level metrics")
            return {"error": "Ground truth required"}
        
        field_metrics = {
            'dealer_name': {'tp': 0, 'fp': 0, 'fn': 0},
            'model_name': {'tp': 0, 'fp': 0, 'fn': 0},
            'horse_power': {'tp': 0, 'fp': 0, 'fn': 0},
            'asset_cost': {'tp': 0, 'fp': 0, 'fn': 0},
            'signature': {'tp': 0, 'fp': 0, 'fn': 0},
            'stamp': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        
        for result in self.results:
            doc_id = result['doc_id']
            if doc_id not in self.ground_truth:
                continue
            
            result_fields = result['fields']
            gt_fields = self.ground_truth[doc_id]['fields']
            
            # Dealer name (fuzzy match)
            if result_fields['dealer_name'] and gt_fields['dealer_name']:
                similarity = self._calculate_similarity(
                    result_fields['dealer_name'],
                    gt_fields['dealer_name']
                )
                if similarity >= 0.9:
                    field_metrics['dealer_name']['tp'] += 1
                else:
                    field_metrics['dealer_name']['fp'] += 1
            elif result_fields['dealer_name'] and not gt_fields['dealer_name']:
                field_metrics['dealer_name']['fp'] += 1
            elif not result_fields['dealer_name'] and gt_fields['dealer_name']:
                field_metrics['dealer_name']['fn'] += 1
            
            # Model name (exact match)
            if result_fields['model_name'] and gt_fields['model_name']:
                if str(result_fields['model_name']).upper() == str(gt_fields['model_name']).upper():
                    field_metrics['model_name']['tp'] += 1
                else:
                    field_metrics['model_name']['fp'] += 1
            elif result_fields['model_name'] and not gt_fields['model_name']:
                field_metrics['model_name']['fp'] += 1
            elif not result_fields['model_name'] and gt_fields['model_name']:
                field_metrics['model_name']['fn'] += 1
            
            # Horse power (±5%)
            if result_fields['horse_power'] and gt_fields['horse_power']:
                hp_result = float(result_fields['horse_power'])
                hp_gt = float(gt_fields['horse_power'])
                if abs(hp_result - hp_gt) <= abs(hp_gt * 0.05):
                    field_metrics['horse_power']['tp'] += 1
                else:
                    field_metrics['horse_power']['fp'] += 1
            elif result_fields['horse_power'] and not gt_fields['horse_power']:
                field_metrics['horse_power']['fp'] += 1
            elif not result_fields['horse_power'] and gt_fields['horse_power']:
                field_metrics['horse_power']['fn'] += 1
            
            # Asset cost (±5%)
            if result_fields['asset_cost'] and gt_fields['asset_cost']:
                cost_result = float(result_fields['asset_cost'])
                cost_gt = float(gt_fields['asset_cost'])
                if abs(cost_result - cost_gt) <= abs(cost_gt * 0.05):
                    field_metrics['asset_cost']['tp'] += 1
                else:
                    field_metrics['asset_cost']['fp'] += 1
            elif result_fields['asset_cost'] and not gt_fields['asset_cost']:
                field_metrics['asset_cost']['fp'] += 1
            elif not result_fields['asset_cost'] and gt_fields['asset_cost']:
                field_metrics['asset_cost']['fn'] += 1
            
            # Signature (binary)
            if 'signature' in result_fields and 'signature' in gt_fields:
                if result_fields['signature']['present'] == gt_fields['signature']['present']:
                    field_metrics['signature']['tp'] += 1
                else:
                    if result_fields['signature']['present']:
                        field_metrics['signature']['fp'] += 1
                    else:
                        field_metrics['signature']['fn'] += 1
            
            # Stamp (binary)
            if 'stamp' in result_fields and 'stamp' in gt_fields:
                if result_fields['stamp']['present'] == gt_fields['stamp']['present']:
                    field_metrics['stamp']['tp'] += 1
                else:
                    if result_fields['stamp']['present']:
                        field_metrics['stamp']['fp'] += 1
                    else:
                        field_metrics['stamp']['fn'] += 1
        
        # Calculate precision, recall, F1 for each field
        for field, metrics in field_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['precision'] = round(precision, 3)
            metrics['recall'] = round(recall, 3)
            metrics['f1_score'] = round(f1, 3)
        
        return field_metrics
    
    def analyze_confidence_distribution(self) -> Dict[str, Any]:
        """
        Analyze confidence score distribution
        
        Returns:
            Confidence analysis
        """
        confidences = [r.get('confidence', 0) for r in self.results if r.get('status') == 'success']
        
        if not confidences:
            return {"error": "No successful results"}
        
        # Categorize by confidence levels
        high_conf = [c for c in confidences if c >= 0.9]
        medium_conf = [c for c in confidences if 0.7 <= c < 0.9]
        low_conf = [c for c in confidences if c < 0.7]
        
        return {
            'total_documents': len(confidences),
            'mean_confidence': round(statistics.mean(confidences), 3),
            'median_confidence': round(statistics.median(confidences), 3),
            'std_confidence': round(statistics.stdev(confidences) if len(confidences) > 1 else 0, 3),
            'min_confidence': round(min(confidences), 3),
            'max_confidence': round(max(confidences), 3),
            'confidence_distribution': {
                'high': {
                    'count': len(high_conf),
                    'percentage': round(len(high_conf) / len(confidences) * 100, 1)
                },
                'medium': {
                    'count': len(medium_conf),
                    'percentage': round(len(medium_conf) / len(confidences) * 100, 1)
                },
                'low': {
                    'count': len(low_conf),
                    'percentage': round(len(low_conf) / len(confidences) * 100, 1)
                }
            },
            'threshold_analysis': {
                'above_95': sum(1 for c in confidences if c >= 0.95),
                'above_90': sum(1 for c in confidences if c >= 0.90),
                'above_80': sum(1 for c in confidences if c >= 0.80),
                'above_70': sum(1 for c in confidences if c >= 0.70)
            }
        }
    
    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """
        Analyze processing performance metrics
        
        Returns:
            Performance analysis
        """
        successful_results = [r for r in self.results if r.get('status') == 'success']
        
        if not successful_results:
            return {"error": "No successful results"}
        
        processing_times = [r.get('processing_time_sec', 0) for r in successful_results]
        ocr_times = [r.get('processing_details', {}).get('ocr_time_sec', 0) for r in successful_results]
        extraction_times = [r.get('processing_details', {}).get('extraction_time_sec', 0) for r in successful_results]
        detection_times = [r.get('processing_details', {}).get('detection_time_sec', 0) for r in successful_results]
        vlm_times = [r.get('processing_details', {}).get('vlm_time_sec', 0) for r in successful_results]
        
        return {
            'total_documents': len(successful_results),
            'processing_time': {
                'mean': round(statistics.mean(processing_times), 3),
                'median': round(statistics.median(processing_times), 3),
                'std': round(statistics.stdev(processing_times) if len(processing_times) > 1 else 0, 3),
                'min': round(min(processing_times), 3),
                'max': round(max(processing_times), 3),
                'total': round(sum(processing_times), 3),
                'target_30s': sum(1 for t in processing_times if t <= 30)
            },
            'component_times': {
                'ocr': {
                    'mean': round(statistics.mean(ocr_times), 3),
                    'percentage': round(statistics.mean(ocr_times) / statistics.mean(processing_times) * 100, 1)
                },
                'extraction': {
                    'mean': round(statistics.mean(extraction_times), 3),
                    'percentage': round(statistics.mean(extraction_times) / statistics.mean(processing_times) * 100, 1)
                },
                'detection': {
                    'mean': round(statistics.mean(detection_times), 3),
                    'percentage': round(statistics.mean(detection_times) / statistics.mean(processing_times) * 100, 1)
                },
                'vlm': {
                    'mean': round(statistics.mean(vlm_times), 3),
                    'percentage': round(statistics.mean(vlm_times) / statistics.mean(processing_times) * 100, 1),
                    'used_count': sum(1 for r in successful_results if r.get('processing_details', {}).get('vlm_used', False))
                }
            },
            'throughput': {
                'documents_per_hour': round(len(successful_results) / (sum(processing_times) / 3600), 1),
                'average_time_per_doc': round(statistics.mean(processing_times), 3)
            }
        }
    
    def analyze_cost_metrics(self) -> Dict[str, Any]:
        """
        Analyze cost metrics
        
        Returns:
            Cost analysis
        """
        successful_results = [r for r in self.results if r.get('status') == 'success']
        
        if not successful_results:
            return {"error": "No successful results"}
        
        costs = [r.get('cost_estimate_usd', 0) for r in successful_results]
        
        return {
            'total_documents': len(successful_results),
            'cost_metrics': {
                'mean': round(statistics.mean(costs), 6),
                'median': round(statistics.median(costs), 6),
                'std': round(statistics.stdev(costs) if len(costs) > 1 else 0, 6),
                'min': round(min(costs), 6),
                'max': round(max(costs), 6),
                'total': round(sum(costs), 6)
            },
            'cost_distribution': {
                'below_001': sum(1 for c in costs if c < 0.001),
                '001_to_005': sum(1 for c in costs if 0.001 <= c < 0.005),
                '005_to_01': sum(1 for c in costs if 0.005 <= c < 0.01),
                'above_01': sum(1 for c in costs if c >= 0.01)
            },
            'target_analysis': {
                'target_cost': 0.01,
                'below_target': sum(1 for c in costs if c < 0.01),
                'above_target': sum(1 for c in costs if c >= 0.01),
                'percentage_below_target': round(sum(1 for c in costs if c < 0.01) / len(costs) * 100, 1)
            }
        }
    
    def analyze_errors(self) -> Dict[str, Any]:
        """
        Analyze errors and failure cases
        
        Returns:
            Error analysis
        """
        failed_results = [r for r in self.results if r.get('status') == 'failed']
        successful_results = [r for r in self.results if r.get('status') == 'success']
        
        error_categories = defaultdict(int)
        field_extraction_stats = defaultdict(lambda: {'extracted': 0, 'missing': 0})
        
        # Analyze failed results
        for result in failed_results:
            error_msg = result.get('error', 'Unknown error').lower()
            
            # Categorize errors
            if 'ocr' in error_msg:
                error_categories['ocr_failure'] += 1
            elif 'memory' in error_msg or 'out of memory' in error_msg:
                error_categories['memory_error'] += 1
            elif 'timeout' in error_msg:
                error_categories['timeout'] += 1
            elif 'file' in error_msg or 'not found' in error_msg:
                error_categories['file_error'] += 1
            else:
                error_categories['other_errors'] += 1
        
        # Analyze field extraction in successful results
        for result in successful_results:
            fields = result.get('fields', {})
            
            for field_name in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                if fields.get(field_name):
                    field_extraction_stats[field_name]['extracted'] += 1
                else:
                    field_extraction_stats[field_name]['missing'] += 1
        
        # Calculate field extraction rates
        field_rates = {}
        for field, stats in field_extraction_stats.items():
            total = stats['extracted'] + stats['missing']
            if total > 0:
                field_rates[field] = {
                    'extraction_rate': round(stats['extracted'] / total * 100, 1),
                    'extracted': stats['extracted'],
                    'missing': stats['missing']
                }
        
        return {
            'total_documents': len(self.results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': round(len(successful_results) / len(self.results) * 100, 1),
            'error_categories': dict(error_categories),
            'field_extraction_rates': field_rates,
            'common_issues': {
                'low_confidence': sum(1 for r in successful_results if r.get('confidence', 0) < 0.7),
                'high_processing_time': sum(1 for r in successful_results if r.get('processing_time_sec', 0) > 30),
                'high_cost': sum(1 for r in successful_results if r.get('cost_estimate_usd', 0) > 0.01)
            }
        }
    
    def identify_error_samples(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Identify error samples for analysis
        
        Args:
            count: Number of error samples to identify
            
        Returns:
            List of error samples
        """
        error_samples = []
        
        # Sort by confidence (lowest first)
        sorted_results = sorted(self.results, key=lambda x: x.get('confidence', 0))
        
        for result in sorted_results[:count]:
            sample = {
                'doc_id': result.get('doc_id'),
                'confidence': result.get('confidence', 0),
                'processing_time': result.get('processing_time_sec', 0),
                'status': result.get('status', 'unknown'),
                'fields': result.get('fields', {}),
                'error': result.get('error', None),
                'issues': self._identify_field_issues(result.get('fields', {}))
            }
            error_samples.append(sample)
        
        return error_samples
    
    def _identify_field_issues(self, fields: Dict[str, Any]) -> List[str]:
        """Identify issues with extracted fields"""
        issues = []
        
        # Check for missing fields
        required_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        for field in required_fields:
            if not fields.get(field):
                issues.append(f'missing_{field}')
        
        # Check for invalid values
        if fields.get('horse_power'):
            hp = fields['horse_power']
            if not (10 <= hp <= 200):
                issues.append('invalid_horse_power')
        
        if fields.get('asset_cost'):
            cost = fields['asset_cost']
            if cost < 100000 or cost > 5000000:
                issues.append('invalid_asset_cost')
        
        # Check signature/stamp consistency
        if fields.get('signature', {}).get('present') and not fields.get('signature', {}).get('bbox'):
            issues.append('signature_no_bbox')
        
        if fields.get('stamp', {}).get('present') and not fields.get('stamp', {}).get('bbox'):
            issues.append('stamp_no_bbox')
        
        return issues
    
    def generate_visualizations(self, output_dir: str):
        """
        Generate visualization plots
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Data for plots
        successful_results = [r for r in self.results if r.get('status') == 'success']
        
        if not successful_results:
            logger.warning("No successful results for visualizations")
            return
        
        # 1. Confidence Distribution Histogram
        confidences = [r.get('confidence', 0) for r in successful_results]
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.95, color='red', linestyle='--', label='Target (95%)')
        plt.axvline(x=0.9, color='orange', linestyle='--', label='Good (90%)')
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Documents')
        plt.title('Confidence Distribution Across Documents')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Processing Time vs Confidence
        processing_times = [r.get('processing_time_sec', 0) for r in successful_results]
        plt.figure(figsize=(10, 6))
        plt.scatter(processing_times, confidences, alpha=0.6, c='green')
        plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% Confidence')
        plt.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='30s Target')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Confidence Score')
        plt.title('Processing Time vs Confidence Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'time_vs_confidence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Field Extraction Success Rate
        field_stats = self.analyze_errors()['field_extraction_rates']
        fields = list(field_stats.keys())
        extraction_rates = [field_stats[f]['extraction_rate'] for f in fields]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(fields, extraction_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.axhline(y=90, color='red', linestyle='--', label='90% Target')
        plt.xlabel('Field')
        plt.ylabel('Extraction Rate (%)')
        plt.title('Field Extraction Success Rates')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(os.path.join(output_dir, 'field_extraction_rates.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Component Time Breakdown
        component_times = self.analyze_performance_metrics()['component_times']
        components = list(component_times.keys())
        times = [component_times[c]['mean'] for c in components]
        
        plt.figure(figsize=(10, 6))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        plt.pie(times, labels=components, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Average Processing Time Breakdown by Component')
        plt.savefig(os.path.join(output_dir, 'time_breakdown.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_comprehensive_report(self, output_dir: str = "analysis_output"):
        """
        Generate comprehensive evaluation report
        
        Args:
            output_dir: Directory to save reports
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating comprehensive evaluation report...")
        
        # Calculate all metrics
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_file': self.results_file,
            'total_documents': len(self.results),
            'document_level_accuracy': self.calculate_document_level_accuracy(),
            'field_level_metrics': self.calculate_field_level_metrics(),
            'confidence_analysis': self.analyze_confidence_distribution(),
            'performance_analysis': self.analyze_performance_metrics(),
            'cost_analysis': self.analyze_cost_metrics(),
            'error_analysis': self.analyze_errors(),
            'error_samples': self.identify_error_samples(20),
            'summary': self._generate_summary()
        }
        
        # Save analysis report
        analysis_file = os.path.join(output_dir, "analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save error samples separately
        error_samples_file = os.path.join(output_dir, "error_samples.json")
        with open(error_samples_file, 'w', encoding='utf-8') as f:
            json.dump(report['error_samples'], f, indent=2, ensure_ascii=False)
        
        # Generate visualizations
        self.generate_visualizations(output_dir)
        
        # Generate markdown summary
        self._generate_markdown_summary(report, output_dir)
        
        logger.info(f"✅ Comprehensive report generated in {output_dir}")
        logger.info(f"   - analysis.json")
        logger.info(f"   - error_samples.json")
        logger.info(f"   - markdown_report.md")
        logger.info(f"   - Visualizations (.png files)")
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        conf_analysis = self.analyze_confidence_distribution()
        perf_analysis = self.analyze_performance_metrics()
        cost_analysis = self.analyze_cost_metrics()
        error_analysis = self.analyze_errors()
        
        # Calculate key metrics
        avg_confidence = conf_analysis.get('mean_confidence', 0)
        avg_processing_time = perf_analysis.get('processing_time', {}).get('mean', 0)
        avg_cost = cost_analysis.get('cost_metrics', {}).get('mean', 0)
        success_rate = error_analysis.get('success_rate', 0)
        
        # Check targets
        targets = {
            'confidence_target': avg_confidence >= 0.95,
            'processing_time_target': avg_processing_time <= 30,
            'cost_target': avg_cost <= 0.01,
            'success_rate_target': success_rate >= 95
        }
        
        summary = {
            'key_metrics': {
                'average_confidence': round(avg_confidence, 3),
                'average_processing_time_sec': round(avg_processing_time, 3),
                'average_cost_usd': round(avg_cost, 6),
                'success_rate_percent': round(success_rate, 1)
            },
            'target_achievement': targets,
            'overall_assessment': self._get_overall_assessment(targets),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _get_overall_assessment(self, targets: Dict[str, bool]) -> str:
        """Generate overall assessment based on targets"""
        achieved = sum(targets.values())
        total = len(targets)
        
        if achieved == total:
            return "EXCELLENT - All targets achieved"
        elif achieved >= total * 0.75:
            return "GOOD - Most targets achieved"
        elif achieved >= total * 0.5:
            return "FAIR - Some targets achieved"
        else:
            return "NEEDS IMPROVEMENT - Few targets achieved"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        # Analyze data to generate specific recommendations
        conf_analysis = self.analyze_confidence_distribution()
        perf_analysis = self.analyze_performance_metrics()
        error_analysis = self.analyze_errors()
        
        # Confidence-related recommendations
        if conf_analysis.get('confidence_distribution', {}).get('low', {}).get('percentage', 0) > 20:
            recommendations.append(
                "Improve OCR accuracy for low-quality scans using better preprocessing"
            )
        
        # Performance-related recommendations
        if perf_analysis.get('processing_time', {}).get('target_30s', 0) < len(self.results) * 0.9:
            recommendations.append(
                "Optimize VLM usage - only use when confidence is very low to reduce processing time"
            )
        
        # Error-related recommendations
        if error_analysis.get('failed', 0) > len(self.results) * 0.1:
            recommendations.append(
                "Add retry logic for failed OCR attempts with different preprocessing parameters"
            )
        
        # Cost-related recommendations
        vlm_used = perf_analysis.get('component_times', {}).get('vlm', {}).get('used_count', 0)
        if vlm_used > len(self.results) * 0.3:
            recommendations.append(
                "Reduce VLM dependency by improving rule-based extraction patterns"
            )
        
        # Always include these
        recommendations.extend([
            "Implement active learning to improve model with difficult cases",
            "Add more language-specific patterns for Hindi and Gujarati documents",
            "Optimize YOLO model with more signature/stamp training data",
            "Implement caching for frequently seen document templates"
        ])
        
        return recommendations
    
    def _generate_markdown_summary(self, report: Dict[str, Any], output_dir: str):
        """Generate markdown summary report"""
        md_file = os.path.join(output_dir, "markdown_report.md")
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Document AI Pipeline Evaluation Report\n\n")
            
            f.write("## 📊 Executive Summary\n\n")
            summary = report['summary']
            
            f.write(f"**Overall Assessment:** {summary['overall_assessment']}\n\n")
            
            f.write("### Key Metrics\n")
            f.write("| Metric | Value | Target | Status |\n")
            f.write("|--------|-------|--------|--------|\n")
            
            metrics = summary['key_metrics']
            targets = summary['target_achievement']
            
            f.write(f"| Average Confidence | {metrics['average_confidence']:.3f} | ≥0.95 | {'✅' if targets['confidence_target'] else '❌'} |\n")
            f.write(f"| Avg Processing Time | {metrics['average_processing_time_sec']:.2f}s | ≤30s | {'✅' if targets['processing_time_target'] else '❌'} |\n")
            f.write(f"| Avg Cost/Document | ${metrics['average_cost_usd']:.6f} | ≤$0.01 | {'✅' if targets['cost_target'] else '❌'} |\n")
            f.write(f"| Success Rate | {metrics['success_rate_percent']:.1f}% | ≥95% | {'✅' if targets['success_rate_target'] else '❌'} |\n")
            
            f.write("\n## 📈 Detailed Analysis\n\n")
            
            # Document Level Accuracy
            if 'error' not in report['document_level_accuracy']:
                dla = report['document_level_accuracy']
                f.write(f"### Document-Level Accuracy: {dla['document_level_accuracy']}%\n")
                f.write(f"- Target: {dla['target_accuracy']}%\n")
                f.write(f"- Achieved Target: {'✅ Yes' if dla['achieved_target'] else '❌ No'}\n")
                f.write(f"- Correct Documents: {dla['correct_documents']}/{dla['total_documents']}\n\n")
            
            # Field Level Metrics
            if 'error' not in report['field_level_metrics']:
                f.write("### Field-Level Performance\n")
                f.write("| Field | Precision | Recall | F1 Score |\n")
                f.write("|-------|-----------|--------|----------|\n")
                
                for field, metrics in report['field_level_metrics'].items():
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        f.write(f"| {field.replace('_', ' ').title()} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1_score']:.3f} |\n")
                f.write("\n")
            
            # Performance Metrics
            perf = report['performance_analysis']
            if 'error' not in perf:
                f.write("### Performance Metrics\n")
                f.write(f"- Average Processing Time: {perf['processing_time']['mean']:.2f}s\n")
                f.write(f"- Documents within 30s target: {perf['processing_time']['target_30s']}/{perf['total_documents']}\n")
                f.write(f"- Throughput: {perf['throughput']['documents_per_hour']:.0f} documents/hour\n\n")
            
            # Cost Analysis
            cost = report['cost_analysis']
            if 'error' not in cost:
                f.write("### Cost Analysis\n")
                f.write(f"- Average Cost/Document: ${cost['cost_metrics']['mean']:.6f}\n")
                target_analysis = cost['target_analysis']
                f.write(f"- Documents below $0.01 target: {target_analysis['below_target']}/{cost['total_documents']} ({target_analysis['percentage_below_target']}%)\n\n")
            
            # Recommendations
            f.write("## 🎯 Recommendations\n\n")
            for i, rec in enumerate(summary['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n## 📁 Generated Files\n\n")
            f.write("- `analysis.json`: Complete analysis data\n")
            f.write("- `error_samples.json`: Detailed error cases\n")
            f.write("- `markdown_report.md`: This report\n")
            f.write("- `*.png`: Visualization plots\n")
            
            f.write(f"\n*Report generated on {report['timestamp']}*\n")
        
        logger.info(f"Markdown report saved to {md_file}")

def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate Document AI Pipeline Results")
    parser.add_argument("--results", "-r", type=str, default="sample_output/all_results.json",
                       help="Path to all_results.json")
    parser.add_argument("--ground_truth", "-g", type=str,
                       help="Path to ground truth annotations (optional)")
    parser.add_argument("--output", "-o", type=str, default="analysis_output",
                       help="Output directory for analysis results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    print("=" * 60)
    print("Document AI Pipeline Evaluation")
    print("=" * 60)
    print(f"Results file: {args.results}")
    print(f"Ground truth: {args.ground_truth or 'Not provided'}")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = DocumentEvaluator(args.results, args.ground_truth)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report(args.output)
    
    # Display summary
    summary = report['summary']
    print("\n📋 EXECUTIVE SUMMARY")
    print("=" * 40)
    print(f"Overall Assessment: {summary['overall_assessment']}")
    print("\nKey Metrics:")
    for metric, value in summary['key_metrics'].items():
        print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print("\nTarget Achievement:")
    for target, achieved in summary['target_achievement'].items():
        status = "✅" if achieved else "❌"
        print(f"  {target.replace('_', ' ').title()}: {status}")
    
    # Display top recommendations
    print("\nTop Recommendations:")
    for i, rec in enumerate(summary['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n✅ Evaluation complete! Results saved to {args.output}/")
    
    return 0

if __name__ == "__main__":
    exit(main())