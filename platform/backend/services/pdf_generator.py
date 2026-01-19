"""
PDF Report Generator Service
Generates professional PDF reports with embedded charts
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import io
import tempfile
import os

# For chart generation
import plotly.graph_objects as go


class PDFReportGenerator:
    """Generate PDF reports from analysis results with embedded charts"""
    
    # Colors for charts
    COLORS = {
        "primary": "#6366f1",
        "success": "#10b981",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "info": "#3b82f6",
        "purple": "#8b5cf6"
    }
    
    LEVEL_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]  # LOW, MODERATE, HIGH, SEVERE
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.temp_dir = tempfile.mkdtemp()
    
    def _setup_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#6366f1'),
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#4f46e5')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#6366f1')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='Finding',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=5,
            bulletIndent=10
        ))
    
    def _generate_chart_image(self, fig, filename: str, width: int = 500, height: int = 300) -> str:
        """Generate a chart image and return the file path"""
        filepath = os.path.join(self.temp_dir, filename)
        try:
            fig.write_image(filepath, width=width, height=height, scale=2)
            return filepath
        except Exception as e:
            print(f"Error generating chart image: {e}")
            return None
    
    def _create_locations_chart(self, top_locations: Dict[str, int]) -> Optional[str]:
        """Create a horizontal bar chart for top locations"""
        sorted_items = sorted(top_locations.items(), key=lambda x: x[1], reverse=True)[:8]
        locations = [item[0][:25] for item in reversed(sorted_items)]
        values = [item[1] for item in reversed(sorted_items)]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values,
            y=locations,
            orientation='h',
            marker=dict(color=self.COLORS["primary"]),
            text=[f"{v:,}" for v in values],
            textposition='outside',
            textfont=dict(size=10)
        ))
        
        fig.update_layout(
            title=dict(text="Top Locations by Activity", font=dict(size=14)),
            xaxis_title="Count",
            margin=dict(l=120, r=40, t=50, b=40),
            height=300,
            width=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return self._generate_chart_image(fig, "locations_chart.png")
    
    def _create_classification_chart(self, distribution: Dict[str, int]) -> Optional[str]:
        """Create a donut chart for classification distribution"""
        level_order = ["LOW", "MODERATE", "HIGH", "SEVERE"]
        labels = []
        values = []
        chart_colors = []
        
        for i, level in enumerate(level_order):
            if level in distribution:
                labels.append(level)
                values.append(distribution[level])
                chart_colors.append(self.LEVEL_COLORS[i])
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=chart_colors, line=dict(color='white', width=2)),
            hole=0.4,
            textinfo='label+percent',
            textfont=dict(size=11)
        ))
        
        fig.update_layout(
            title=dict(text="Activity Level Distribution", font=dict(size=14)),
            margin=dict(l=20, r=20, t=50, b=20),
            height=300,
            width=400,
            paper_bgcolor='white',
            showlegend=False
        )
        
        return self._generate_chart_image(fig, "classification_chart.png", width=400)
    
    def _create_outliers_chart(self, outlier_counts: Dict[str, int]) -> Optional[str]:
        """Create a bar chart for outlier counts"""
        # Filter only columns with outliers
        filtered = {k: v for k, v in outlier_counts.items() if v > 0}
        if not filtered:
            return None
        
        columns = list(filtered.keys())
        counts = list(filtered.values())
        
        # Truncate long column names
        columns = [c[:15] + "..." if len(c) > 18 else c for c in columns]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=columns,
            y=counts,
            marker=dict(color=self.COLORS["danger"]),
            text=counts,
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(text="Outliers Detected by Column", font=dict(size=14)),
            xaxis_title="Column",
            yaxis_title="Count",
            margin=dict(l=40, r=20, t=50, b=80),
            height=300,
            width=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_tickangle=-45
        )
        
        return self._generate_chart_image(fig, "outliers_chart.png")
    
    def generate_to_bytes(self, analysis_result: Dict[str, Any], dataset_info: Dict[str, Any]) -> bytes:
        """Generate PDF and return as bytes (for HTTP response)"""
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        story = self._build_story_with_charts(analysis_result, dataset_info)
        doc.build(story)
        
        # Cleanup temp files
        self._cleanup_temp_files()
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _cleanup_temp_files(self):
        """Clean up temporary chart images"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            self.temp_dir = tempfile.mkdtemp()
        except:
            pass
    
    def _build_story_with_charts(self, analysis_result: Dict[str, Any], dataset_info: Dict[str, Any]) -> List:
        """Build PDF content with embedded charts"""
        story = []
        
        # Title
        story.append(Paragraph("Traffic Analytics Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.1 * inch))
        
        # Report metadata
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                               self.styles['CustomBody']))
        story.append(Paragraph(f"Dataset: {dataset_info.get('filename', 'Unknown')}", 
                               self.styles['CustomBody']))
        story.append(Spacer(1, 0.2 * inch))
        
        results = analysis_result.get('results', {})
        dataset_results = results.get('dataset_info', {})
        analyses = results.get('analyses', {})
        
        # Dataset Overview Table
        story.append(Paragraph("Dataset Overview", self.styles['SectionTitle']))
        
        overview_data = [
            ['Metric', 'Value'],
            ['Total Rows', str(dataset_results.get('rows', 'N/A'))],
            ['Total Columns', str(len(dataset_results.get('columns', [])))],
            ['Dataset Type', dataset_results.get('detected_type', 'Generic').replace('_', ' ').title()],
            ['Status', analysis_result.get('status', 'Unknown').title()]
        ]
        
        overview_table = Table(overview_data, colWidths=[2.5*inch, 3*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.2 * inch))
        
        # Key Findings
        story.append(Paragraph("Key Findings", self.styles['SectionTitle']))
        summary = results.get('summary', {})
        findings = summary.get('key_findings', [])
        
        if findings:
            for finding in findings[:6]:
                clean_finding = str(finding).replace('**', '').replace('*', '')
                story.append(Paragraph(f"• {clean_finding}", self.styles['Finding']))
        else:
            story.append(Paragraph("No significant findings detected.", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.2 * inch))
        
        # ===== CHARTS SECTION =====
        
        # Top Locations Chart
        if 'patterns' in analyses and analyses['patterns'].get('top_locations'):
            story.append(Paragraph("Top Locations Analysis", self.styles['SectionTitle']))
            
            chart_path = self._create_locations_chart(analyses['patterns']['top_locations'])
            if chart_path and os.path.exists(chart_path):
                img = Image(chart_path, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2 * inch))
        
        # Classification Distribution Chart
        if 'classification' in analyses and analyses['classification'].get('distribution'):
            story.append(Paragraph("Activity Level Classification", self.styles['SectionTitle']))
            
            chart_path = self._create_classification_chart(analyses['classification']['distribution'])
            if chart_path and os.path.exists(chart_path):
                img = Image(chart_path, width=4*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.1 * inch))
            
            # Also add a table with the distribution
            dist = analyses['classification']['distribution']
            total = sum(dist.values())
            dist_data = [['Level', 'Count', 'Percentage']]
            for level in ['LOW', 'MODERATE', 'HIGH', 'SEVERE']:
                if level in dist:
                    count = dist[level]
                    pct = (count / total * 100) if total > 0 else 0
                    dist_data.append([level, str(count), f"{pct:.1f}%"])
            
            dist_table = Table(dist_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
            dist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(dist_table)
            story.append(Spacer(1, 0.2 * inch))
        
        # Outliers Chart
        if 'anomaly' in analyses:
            outlier_counts = analyses['anomaly'].get('outlier_counts', {})
            if outlier_counts and sum(outlier_counts.values()) > 0:
                story.append(Paragraph("Anomaly Detection", self.styles['SectionTitle']))
                
                chart_path = self._create_outliers_chart(outlier_counts)
                if chart_path and os.path.exists(chart_path):
                    img = Image(chart_path, width=5*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2 * inch))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.styles['SectionTitle']))
        recommendations = results.get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                clean_rec = str(rec).replace('**', '').replace('*', '')
                story.append(Paragraph(f"{i}. {clean_rec}", self.styles['Finding']))
        else:
            story.append(Paragraph("No specific recommendations at this time.", self.styles['CustomBody']))
        
        # Footer
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("—" * 60, self.styles['CustomBody']))
        story.append(Paragraph("Generated by Traffic Analytics Platform", 
                               ParagraphStyle('Footer', parent=self.styles['Normal'], 
                                             fontSize=8, textColor=colors.gray, alignment=TA_CENTER)))
        
        return story
