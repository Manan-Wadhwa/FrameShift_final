"""
Basic metrics report generator for anomaly detection.
Generates simple JSON and text reports without AI analysis.
"""

import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


class BasicReportGenerator:
    """Generate basic metrics reports in various formats."""
    
    @staticmethod
    def generate_json_report(
        metrics: Dict,
        output_path: str
    ) -> str:
        """
        Generate JSON report with basic metrics.
        
        Args:
            metrics: Metrics dictionary
            output_path: Path to save JSON report
        
        Returns:
            Path to saved report
        """
        # Create report structure
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_pixels': metrics.get('total_pixels', 0),
                'anomaly_pixels': metrics.get('total_anomaly_pixels', 0),
                'anomaly_percentage': metrics.get('anomaly_percentage', 0),
                'mean_anomaly_score': metrics.get('mean_anomaly_score', 0),
                'max_anomaly_score': metrics.get('max_anomaly_score', 0),
            },
            'severity_breakdown': {
                'low': metrics.get('low_severity_pixels', 0),
                'medium': metrics.get('medium_severity_pixels', 0),
                'high': metrics.get('high_severity_pixels', 0),
            },
            'regions': {
                'count': metrics.get('num_regions', 0),
                'largest_area': metrics.get('largest_region_area', 0),
                'mean_area': float(metrics.get('mean_region_area', 0)),
                'details': metrics.get('regions', [])
            },
            'coverage': {
                'a_to_b': metrics.get('coverage_a_to_b', 0),
                'b_to_a': metrics.get('coverage_b_to_a', 0),
                'difference': metrics.get('coverage_difference', 0)
            },
            'spatial_distribution': metrics.get('spatial_distribution', {})
        }
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ JSON report saved to: {output_path}")
        return output_path
    
    @staticmethod
    def generate_text_report(
        metrics: Dict,
        output_path: str
    ) -> str:
        """
        Generate human-readable text report.
        
        Args:
            metrics: Metrics dictionary
            output_path: Path to save text report
        
        Returns:
            Path to saved report
        """
        lines = [
            "=" * 80,
            "F1 ANOMALY DETECTION - BASIC METRICS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "OVERALL SUMMARY",
            "-" * 80,
            f"Total Pixels:              {metrics.get('total_pixels', 0):,}",
            f"Anomaly Pixels:            {metrics.get('total_anomaly_pixels', 0):,}",
            f"Anomaly Coverage:          {metrics.get('anomaly_percentage', 0):.2f}%",
            f"Mean Anomaly Score:        {metrics.get('mean_anomaly_score', 0):.4f}",
            f"Max Anomaly Score:         {metrics.get('max_anomaly_score', 0):.4f}",
            "",
            "SEVERITY BREAKDOWN",
            "-" * 80,
            f"Low Severity Pixels:       {metrics.get('low_severity_pixels', 0):,}",
            f"Medium Severity Pixels:    {metrics.get('medium_severity_pixels', 0):,}",
            f"High Severity Pixels:      {metrics.get('high_severity_pixels', 0):,}",
            "",
            "REGION ANALYSIS",
            "-" * 80,
            f"Number of Regions:         {metrics.get('num_regions', 0)}",
            f"Largest Region Area:       {metrics.get('largest_region_area', 0):,} pixels",
            f"Mean Region Area:          {metrics.get('mean_region_area', 0):.1f} pixels",
            "",
        ]
        
        # Add top regions
        regions = metrics.get('regions', [])
        if regions:
            lines.append("TOP ANOMALY REGIONS")
            lines.append("-" * 80)
            # Sort by area descending
            sorted_regions = sorted(regions, key=lambda r: r['area'], reverse=True)
            for i, region in enumerate(sorted_regions[:10], 1):
                x, y, w, h = region['bbox']
                lines.append(
                    f"Region {i:2d}: Area={region['area']:6,} px, "
                    f"BBox=({x:4d},{y:4d},{w:4d},{h:4d}), "
                    f"MaxIntensity={region['max_intensity']:.3f}, "
                    f"MeanIntensity={region['mean_intensity']:.3f}"
                )
            lines.append("")
        
        # Add coverage comparison
        lines.extend([
            "COVERAGE COMPARISON",
            "-" * 80,
            f"A → B Coverage:            {metrics.get('coverage_a_to_b', 0):.2f}%",
            f"B → A Coverage:            {metrics.get('coverage_b_to_a', 0):.2f}%",
            f"Coverage Difference:       {metrics.get('coverage_difference', 0):.2f}%",
            "",
        ])
        
        # Add spatial distribution
        spatial_dist = metrics.get('spatial_distribution', {})
        if spatial_dist:
            lines.extend([
                "SPATIAL DISTRIBUTION (3x3 Grid)",
                "-" * 80,
            ])
            # Arrange in 3x3 grid
            for i in range(3):
                row_values = []
                for j in range(3):
                    key = f'cell_{i}_{j}'
                    value = spatial_dist.get(key, 0) * 100
                    row_values.append(f"{value:5.1f}%")
                lines.append("  ".join(row_values))
            lines.append("")
        
        lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        # Save text report
        report_text = "\n".join(lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Text report saved to: {output_path}")
        return output_path
    
    @staticmethod
    def generate_html_report(
        metrics: Dict,
        output_path: str
    ) -> str:
        """
        Generate basic HTML report (without AI analysis).
        
        Args:
            metrics: Metrics dictionary
            output_path: Path to save HTML report
        
        Returns:
            Path to saved report
        """
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>F1 Anomaly Detection - Basic Metrics Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #e10600;
            text-align: center;
            font-size: 32px;
            margin-bottom: 10px;
            border-bottom: 3px solid #e10600;
            padding-bottom: 15px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section-title {{
            font-size: 24px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .severity-bar {{
            display: flex;
            height: 40px;
            border-radius: 5px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .severity-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }}
        .severity-low {{
            background-color: #28a745;
        }}
        .severity-medium {{
            background-color: #ffc107;
            color: black;
        }}
        .severity-high {{
            background-color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .grid-viz {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 20px 0;
        }}
        .grid-cell {{
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 5px;
            font-weight: bold;
            color: white;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏎️ F1 Anomaly Detection</h1>
        <div class="subtitle">Basic Metrics Report</div>
        
        <div class="section">
            <div class="section-title">📊 Key Metrics</div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Anomaly Coverage</div>
                    <div class="metric-value">{metrics.get('anomaly_percentage', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Affected Pixels</div>
                    <div class="metric-value">{metrics.get('total_anomaly_pixels', 0):,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Anomaly Regions</div>
                    <div class="metric-value">{metrics.get('num_regions', 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Intensity</div>
                    <div class="metric-value">{metrics.get('max_anomaly_score', 0):.2f}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">⚠️ Severity Distribution</div>
"""
        
        # Calculate percentages for severity bar
        total_severity = (metrics.get('low_severity_pixels', 0) + 
                         metrics.get('medium_severity_pixels', 0) + 
                         metrics.get('high_severity_pixels', 0))
        
        if total_severity > 0:
            low_pct = (metrics.get('low_severity_pixels', 0) / total_severity) * 100
            med_pct = (metrics.get('medium_severity_pixels', 0) / total_severity) * 100
            high_pct = (metrics.get('high_severity_pixels', 0) / total_severity) * 100
            
            html_content += f"""
            <div class="severity-bar">
                <div class="severity-segment severity-low" style="width: {low_pct}%">
                    Low: {metrics.get('low_severity_pixels', 0):,} ({low_pct:.1f}%)
                </div>
                <div class="severity-segment severity-medium" style="width: {med_pct}%">
                    Medium: {metrics.get('medium_severity_pixels', 0):,} ({med_pct:.1f}%)
                </div>
                <div class="severity-segment severity-high" style="width: {high_pct}%">
                    High: {metrics.get('high_severity_pixels', 0):,} ({high_pct:.1f}%)
                </div>
            </div>
"""
        
        # Add regions table
        regions = metrics.get('regions', [])
        if regions:
            sorted_regions = sorted(regions, key=lambda r: r['area'], reverse=True)
            html_content += """
        </div>
        
        <div class="section">
            <div class="section-title">🎯 Top Anomaly Regions</div>
            <table>
                <tr>
                    <th>Region</th>
                    <th>Area (pixels)</th>
                    <th>Bounding Box</th>
                    <th>Max Intensity</th>
                    <th>Mean Intensity</th>
                </tr>
"""
            for i, region in enumerate(sorted_regions[:10], 1):
                x, y, w, h = region['bbox']
                html_content += f"""
                <tr>
                    <td>Region {i}</td>
                    <td>{region['area']:,}</td>
                    <td>({x}, {y}, {w}, {h})</td>
                    <td>{region['max_intensity']:.3f}</td>
                    <td>{region['mean_intensity']:.3f}</td>
                </tr>
"""
            html_content += """
            </table>
"""
        
        # Add spatial distribution
        spatial_dist = metrics.get('spatial_distribution', {})
        if spatial_dist:
            html_content += """
        </div>
        
        <div class="section">
            <div class="section-title">🗺️ Spatial Distribution</div>
            <div class="grid-viz">
"""
            for i in range(3):
                for j in range(3):
                    key = f'cell_{i}_{j}'
                    value = spatial_dist.get(key, 0) * 100
                    # Color based on intensity
                    if value < 10:
                        color = '#28a745'
                    elif value < 30:
                        color = '#ffc107'
                    else:
                        color = '#dc3545'
                    
                    html_content += f"""
                <div class="grid-cell" style="background-color: {color}">
                    {value:.1f}%
                </div>
"""
            html_content += """
            </div>
"""
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content += f"""
        </div>
        
        <div class="timestamp">
            Report generated: {timestamp}
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved to: {output_path}")
        return output_path
