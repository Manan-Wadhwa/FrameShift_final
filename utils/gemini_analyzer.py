"""
Gemini API integration for analyzing anomaly heatmaps and generating AI-powered reports.
Focuses on structural changes (cracks, damage, deformation) rather than cosmetic changes.
"""

import os
import base64
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

class GeminiAnalyzer:
    """Analyzes anomaly heatmaps using Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini analyzer.
        
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )
        
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✓ Gemini API initialized successfully")
        except ImportError:
            raise ImportError(
                "google-generativeai package not found. Install with: pip install google-generativeai"
            )
    
    def encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Image array (BGR or grayscale)
            
        Returns:
            Base64 encoded image string
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        
        return base64.b64encode(buffer).decode('utf-8')
    
    def create_structural_analysis_prompt(self, metrics: Dict) -> str:
        """
        Create prompt for Gemini focusing on structural changes.
        
        Args:
            metrics: Dictionary containing basic metrics
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are analyzing anomaly detection heatmaps from an F1 race c ar inspection system.

**CRITICAL INSTRUCTIONS:**
- Focus ONLY on structural and mechanical changes that affect safety/performance
- IGNORE cosmetic changes like logos, stickers, paint, sponsor decals
- Report tire cracks, rubber degradation, surface damage
- Report wing damage, deformation, cracks in aerodynamic components
- Report suspension damage, brake disc wear, structural cracks
- Report bodywork damage, carbon fiber cracks, panel deformation
- Use the heatmap colors: RED = high anomaly (critical damage), YELLOW = medium anomaly (wear), BLUE = low anomaly

**Images Provided:**
1. Image A (reference/before)
2. Image B (current/after)
3. Union Heatmap (combined anomalies from both comparisons)

**Basic Metrics:**
- Total anomaly area: {metrics.get('total_anomaly_pixels', 0)} pixels ({metrics.get('anomaly_percentage', 0):.2f}%)
- High severity anomalies: {metrics.get('high_severity_pixels', 0)} pixels
- Medium severity anomalies: {metrics.get('medium_severity_pixels', 0)} pixels
- Low severity anomalies: {metrics.get('low_severity_pixels', 0)} pixels
- Number of anomaly regions: {metrics.get('num_regions', 0)}

**Your Task:**
1. Analyze the heatmap and identify specific structural issues (NOT cosmetic)
2. For each issue, specify:
   - Component affected (tire, wing, suspension, etc.)
   - Type of damage (crack, deformation, wear, etc.)
   - Severity (critical/high/medium/low based on heatmap color)
   - Location (front/rear/left/right, specific component)
3. Prioritize issues by safety/performance impact
4. Provide actionable recommendations

**Output Format (JSON):**
{{
    "structural_issues": [
        {{
            "component": "Front left tire",
            "issue_type": "Surface cracking",
            "severity": "high",
            "location": "Outer shoulder region",
            "description": "Multiple radial cracks visible in tire rubber",
            "safety_impact": "Critical - potential tire failure",
            "recommendation": "Immediate replacement required"
        }}
    ],
    "summary": "Brief overall assessment",
    "critical_actions": ["List of immediate actions needed"]
}}

Analyze the images and provide your assessment in the JSON format above.
"""
        return prompt
    
    def analyze_heatmaps(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        union_heatmap: np.ndarray,
        metrics: Dict
    ) -> Dict:
        """
        Send heatmaps to Gemini for AI-powered structural analysis.
        
        Args:
            image_a: Reference image
            image_b: Current image
            union_heatmap: Combined anomaly heatmap
            metrics: Basic metrics dictionary
            
        Returns:
            Dictionary containing AI analysis results
        """
        print("📤 Sending heatmaps to Gemini for analysis...")
        
        try:
            # Create prompt
            prompt = self.create_structural_analysis_prompt(metrics)
            
            # Prepare images for Gemini
            from PIL import Image
            import io
            
            def np_to_pil(img):
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img)
            
            image_a_pil = np_to_pil(image_a)
            image_b_pil = np_to_pil(image_b)
            union_heatmap_pil = np_to_pil(union_heatmap)
            
            # Send to Gemini
            response = self.model.generate_content([
                prompt,
                image_a_pil,
                image_b_pil,
                union_heatmap_pil
            ])
            
            print("✓ Received response from Gemini")
            
            # Parse response
            response_text = response.text
            
            # Try to extract JSON from response
            try:
                # Look for JSON block in response
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif '{' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_text = response_text[json_start:json_end]
                else:
                    json_text = response_text
                
                analysis = json.loads(json_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
                analysis = {
                    'raw_response': response_text,
                    'structural_issues': [],
                    'summary': response_text,
                    'critical_actions': []
                }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error during Gemini analysis: {e}")
            return {
                'error': str(e),
                'structural_issues': [],
                'summary': 'Analysis failed',
                'critical_actions': []
            }
    
    def generate_report(
        self,
        metrics: Dict,
        ai_analysis: Dict,
        output_path: str
    ) -> str:
        """
        Generate comprehensive HTML report combining metrics and AI analysis.
        
        Args:
            metrics: Basic metrics dictionary
            ai_analysis: AI analysis results from Gemini
            output_path: Path to save HTML report
            
        Returns:
            Path to generated report
        """
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>F1 Anomaly Detection Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #e10600;
            border-bottom: 3px solid #e10600;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #333;
            margin-top: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .issue {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .issue.critical {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }}
        .issue.high {{
            background-color: #fff3cd;
            border-left-color: #ff6b6b;
        }}
        .issue.medium {{
            background-color: #d1ecf1;
            border-left-color: #17a2b8;
        }}
        .issue-header {{
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
        }}
        .severity-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .severity-critical {{
            background-color: #dc3545;
            color: white;
        }}
        .severity-high {{
            background-color: #ff6b6b;
            color: white;
        }}
        .severity-medium {{
            background-color: #ffc107;
            color: black;
        }}
        .severity-low {{
            background-color: #28a745;
            color: white;
        }}
        .actions {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary {{
            background-color: #e7f3ff;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #2196F3;
        }}
        .timestamp {{
            color: #666;
            font-size: 14px;
            margin-top: 30px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏎️ F1 Anomaly Detection Report</h1>
        
        <h2>📊 Basic Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Anomaly Coverage</div>
                <div class="metric-value">{metrics.get('anomaly_percentage', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">High Severity Pixels</div>
                <div class="metric-value">{metrics.get('high_severity_pixels', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Medium Severity Pixels</div>
                <div class="metric-value">{metrics.get('medium_severity_pixels', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Anomaly Regions</div>
                <div class="metric-value">{metrics.get('num_regions', 0)}</div>
            </div>
        </div>
        
        <h2>🤖 AI-Powered Structural Analysis</h2>
        <div class="summary">
            <strong>Summary:</strong> {ai_analysis.get('summary', 'No summary available')}
        </div>
"""
        
        # Add structural issues
        issues = ai_analysis.get('structural_issues', [])
        if issues:
            html_content += "<h3>⚠️ Detected Structural Issues</h3>\n"
            for issue in issues:
                severity = issue.get('severity', 'low').lower()
                html_content += f"""
        <div class="issue {severity}">
            <div class="issue-header">
                {issue.get('component', 'Unknown Component')}
                <span class="severity-badge severity-{severity}">{severity.upper()}</span>
            </div>
            <div><strong>Issue:</strong> {issue.get('issue_type', 'Unknown')}</div>
            <div><strong>Location:</strong> {issue.get('location', 'Not specified')}</div>
            <div><strong>Description:</strong> {issue.get('description', 'No description')}</div>
            <div><strong>Safety Impact:</strong> {issue.get('safety_impact', 'Unknown')}</div>
            <div><strong>Recommendation:</strong> {issue.get('recommendation', 'No recommendation')}</div>
        </div>
"""
        else:
            html_content += "<p>✓ No significant structural issues detected.</p>\n"
        
        # Add critical actions
        actions = ai_analysis.get('critical_actions', [])
        if actions:
            html_content += "<h3>🔧 Critical Actions Required</h3>\n<div class='actions'><ul>\n"
            for action in actions:
                html_content += f"<li>{action}</li>\n"
            html_content += "</ul></div>\n"
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content += f"""
        <div class="timestamp">Report generated: {timestamp}</div>
    </div>
</body>
</html>
"""
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Report saved to: {output_path}")
        return output_path
