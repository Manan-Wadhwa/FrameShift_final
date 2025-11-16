"""
Heatmap metrics calculation for anomaly detection.
Computes union, intersection, and various statistical metrics from heatmaps.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional


class HeatmapMetrics:
    """Calculate comprehensive metrics from anomaly heatmaps."""
    
    @staticmethod
    def compute_union_heatmap(
        heatmap_a2b: np.ndarray,
        heatmap_b2a: np.ndarray,
        method: str = 'max'
    ) -> np.ndarray:
        """
        Compute union of two heatmaps.
        
        Args:
            heatmap_a2b: Heatmap from A to B comparison
            heatmap_b2a: Heatmap from B to A comparison
            method: Union method ('max', 'mean', 'weighted')
                - 'max': Take maximum value at each pixel
                - 'mean': Take average value
                - 'weighted': Weighted combination (0.6 * max + 0.4 * mean)
        
        Returns:
            Union heatmap
        """
        if heatmap_a2b.shape != heatmap_b2a.shape:
            raise ValueError(f"Heatmap shapes must match: {heatmap_a2b.shape} vs {heatmap_b2a.shape}")
        
        if method == 'max':
            union = np.maximum(heatmap_a2b, heatmap_b2a)
        elif method == 'mean':
            union = (heatmap_a2b + heatmap_b2a) / 2.0
        elif method == 'weighted':
            max_map = np.maximum(heatmap_a2b, heatmap_b2a)
            mean_map = (heatmap_a2b + heatmap_b2a) / 2.0
            union = 0.6 * max_map + 0.4 * mean_map
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return union.astype(np.float32)
    
    @staticmethod
    def compute_intersection_heatmap(
        heatmap_a2b: np.ndarray,
        heatmap_b2a: np.ndarray
    ) -> np.ndarray:
        """
        Compute intersection of two heatmaps (minimum values).
        
        Args:
            heatmap_a2b: Heatmap from A to B comparison
            heatmap_b2a: Heatmap from B to A comparison
        
        Returns:
            Intersection heatmap
        """
        return np.minimum(heatmap_a2b, heatmap_b2a).astype(np.float32)
    
    @staticmethod
    def compute_difference_heatmap(
        heatmap_a2b: np.ndarray,
        heatmap_b2a: np.ndarray
    ) -> np.ndarray:
        """
        Compute absolute difference between heatmaps.
        
        Args:
            heatmap_a2b: Heatmap from A to B comparison
            heatmap_b2a: Heatmap from B to A comparison
        
        Returns:
            Difference heatmap
        """
        return np.abs(heatmap_a2b - heatmap_b2a).astype(np.float32)
    
    @staticmethod
    def compute_severity_levels(
        heatmap: np.ndarray,
        low_threshold: float = 0.3,
        medium_threshold: float = 0.6,
        high_threshold: float = 0.8
    ) -> Dict[str, int]:
        """
        Compute pixel counts for different severity levels.
        
        Args:
            heatmap: Anomaly heatmap (normalized 0-1)
            low_threshold: Threshold for low severity
            medium_threshold: Threshold for medium severity
            high_threshold: Threshold for high severity
        
        Returns:
            Dictionary with pixel counts for each severity level
        """
        # Normalize if needed
        if heatmap.max() > 1.0:
            heatmap = heatmap / 255.0
        
        low_pixels = np.sum((heatmap >= low_threshold) & (heatmap < medium_threshold))
        medium_pixels = np.sum((heatmap >= medium_threshold) & (heatmap < high_threshold))
        high_pixels = np.sum(heatmap >= high_threshold)
        
        return {
            'low_severity_pixels': int(low_pixels),
            'medium_severity_pixels': int(medium_pixels),
            'high_severity_pixels': int(high_pixels),
            'total_anomaly_pixels': int(low_pixels + medium_pixels + high_pixels)
        }
    
    @staticmethod
    def compute_connected_components(
        heatmap: np.ndarray,
        threshold: float = 0.3,
        min_area: int = 50
    ) -> Tuple[int, List[Dict]]:
        """
        Find connected anomaly regions in heatmap.
        
        Args:
            heatmap: Anomaly heatmap
            threshold: Threshold for binarization
            min_area: Minimum area for valid regions
        
        Returns:
            Tuple of (number of regions, list of region info dicts)
        """
        # Normalize and threshold
        if heatmap.max() > 1.0:
            heatmap = heatmap / 255.0
        
        binary = (heatmap >= threshold).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                
                # Get max value in this region
                mask = (labels == i).astype(np.uint8)
                max_val = np.max(heatmap[mask == 1])
                mean_val = np.mean(heatmap[mask == 1])
                
                regions.append({
                    'id': i,
                    'area': int(area),
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'centroid': (float(cx), float(cy)),
                    'max_intensity': float(max_val),
                    'mean_intensity': float(mean_val)
                })
        
        return len(regions), regions
    
    @staticmethod
    def compute_spatial_distribution(
        heatmap: np.ndarray,
        grid_size: Tuple[int, int] = (3, 3)
    ) -> Dict[str, float]:
        """
        Compute spatial distribution of anomalies across image grid.
        
        Args:
            heatmap: Anomaly heatmap
            grid_size: Grid dimensions (rows, cols)
        
        Returns:
            Dictionary with anomaly percentage for each grid cell
        """
        h, w = heatmap.shape[:2]
        rows, cols = grid_size
        
        cell_h = h // rows
        cell_w = w // cols
        
        distribution = {}
        for i in range(rows):
            for j in range(cols):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i < rows - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j < cols - 1 else w
                
                cell = heatmap[y1:y2, x1:x2]
                
                # Normalize if needed
                if cell.max() > 1.0:
                    cell = cell / 255.0
                
                # Calculate anomaly percentage in this cell
                anomaly_ratio = np.mean(cell > 0.3)
                distribution[f'cell_{i}_{j}'] = float(anomaly_ratio)
        
        return distribution
    
    @staticmethod
    def compute_comprehensive_metrics(
        heatmap_a2b: np.ndarray,
        heatmap_b2a: np.ndarray,
        image_a: Optional[np.ndarray] = None,
        image_b: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute all metrics for a pair of heatmaps.
        
        Args:
            heatmap_a2b: Heatmap from A to B comparison
            heatmap_b2a: Heatmap from B to A comparison
            image_a: Original image A (optional, for visualization)
            image_b: Original image B (optional, for visualization)
        
        Returns:
            Comprehensive metrics dictionary
        """
        print("📊 Computing comprehensive heatmap metrics...")
        
        # Compute union heatmap
        union_heatmap = HeatmapMetrics.compute_union_heatmap(heatmap_a2b, heatmap_b2a, method='max')
        intersection_heatmap = HeatmapMetrics.compute_intersection_heatmap(heatmap_a2b, heatmap_b2a)
        difference_heatmap = HeatmapMetrics.compute_difference_heatmap(heatmap_a2b, heatmap_b2a)
        
        # Normalize heatmaps
        union_norm = union_heatmap / 255.0 if union_heatmap.max() > 1.0 else union_heatmap
        
        # Severity levels
        severity = HeatmapMetrics.compute_severity_levels(union_norm)
        
        # Connected components
        num_regions, regions = HeatmapMetrics.compute_connected_components(union_norm)
        
        # Spatial distribution
        spatial_dist = HeatmapMetrics.compute_spatial_distribution(union_norm)
        
        # Overall statistics
        total_pixels = union_norm.shape[0] * union_norm.shape[1]
        anomaly_percentage = (severity['total_anomaly_pixels'] / total_pixels) * 100
        
        # Mean and max anomaly scores
        mean_anomaly_score = float(np.mean(union_norm))
        max_anomaly_score = float(np.max(union_norm))
        
        # Coverage metrics
        coverage_a2b = np.sum(heatmap_a2b > 0.3 * 255) / total_pixels * 100
        coverage_b2a = np.sum(heatmap_b2a > 0.3 * 255) / total_pixels * 100
        
        metrics = {
            # Basic stats
            'total_pixels': int(total_pixels),
            'total_anomaly_pixels': severity['total_anomaly_pixels'],
            'anomaly_percentage': float(anomaly_percentage),
            'mean_anomaly_score': mean_anomaly_score,
            'max_anomaly_score': max_anomaly_score,
            
            # Severity breakdown
            'low_severity_pixels': severity['low_severity_pixels'],
            'medium_severity_pixels': severity['medium_severity_pixels'],
            'high_severity_pixels': severity['high_severity_pixels'],
            
            # Region analysis
            'num_regions': num_regions,
            'regions': regions,
            'largest_region_area': max([r['area'] for r in regions]) if regions else 0,
            'mean_region_area': np.mean([r['area'] for r in regions]) if regions else 0,
            
            # Spatial distribution
            'spatial_distribution': spatial_dist,
            
            # Coverage comparison
            'coverage_a_to_b': float(coverage_a2b),
            'coverage_b_to_a': float(coverage_b2a),
            'coverage_difference': float(abs(coverage_a2b - coverage_b2a)),
            
            # Heatmaps
            'union_heatmap': union_heatmap,
            'intersection_heatmap': intersection_heatmap,
            'difference_heatmap': difference_heatmap
        }
        
        print(f"✓ Metrics computed: {anomaly_percentage:.2f}% anomaly coverage, {num_regions} regions")
        
        return metrics
    
    @staticmethod
    def visualize_metrics(
        metrics: Dict,
        image_a: np.ndarray,
        image_b: np.ndarray,
        output_path: str
    ) -> str:
        """
        Create visualization of metrics and heatmaps.
        
        Args:
            metrics: Metrics dictionary from compute_comprehensive_metrics
            image_a: Original image A
            image_b: Original image B
            output_path: Path to save visualization
        
        Returns:
            Path to saved visualization
        """
        # Create figure with multiple subplots
        union_heatmap = metrics['union_heatmap']
        intersection_heatmap = metrics['intersection_heatmap']
        difference_heatmap = metrics['difference_heatmap']
        
        # Apply colormap
        union_colored = cv2.applyColorMap((union_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        intersection_colored = cv2.applyColorMap((intersection_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        difference_colored = cv2.applyColorMap((difference_heatmap * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        # Resize images to same size
        h, w = union_colored.shape[:2]
        image_a_resized = cv2.resize(image_a, (w, h))
        image_b_resized = cv2.resize(image_b, (w, h))
        
        # Create overlay visualizations
        union_overlay = cv2.addWeighted(image_a_resized, 0.5, union_colored, 0.5, 0)
        
        # Draw regions on union overlay
        regions = metrics.get('regions', [])
        for region in regions[:10]:  # Draw top 10 regions
            x, y, rw, rh = region['bbox']
            cv2.rectangle(union_overlay, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
            # Add region ID
            cv2.putText(union_overlay, f"R{region['id']}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Create grid layout
        row1 = np.hstack([image_a_resized, image_b_resized])
        row2 = np.hstack([union_colored, union_overlay])
        row3 = np.hstack([intersection_colored, difference_colored])
        
        # Stack rows
        visualization = np.vstack([row1, row2, row3])
        
        # Add text labels
        label_height = 30
        labeled_viz = np.zeros((visualization.shape[0] + label_height * 3, visualization.shape[1], 3), dtype=np.uint8)
        labeled_viz[label_height * 3:] = visualization
        
        # Add labels
        labels = [
            ('Image A (Reference)', 'Image B (Current)'),
            ('Union Heatmap', 'Union Overlay + Regions'),
            ('Intersection Heatmap', 'Difference Heatmap')
        ]
        
        y_offset = 0
        for i, (left_label, right_label) in enumerate(labels):
            cv2.putText(labeled_viz, left_label, (10, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(labeled_viz, right_label, (w + 10, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += label_height
        
        # Save
        cv2.imwrite(output_path, labeled_viz)
        print(f"✓ Metrics visualization saved to: {output_path}")
        
        return output_path
