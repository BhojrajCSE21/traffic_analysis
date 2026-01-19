"""
Data Validation Service
Validates uploaded files, auto-detects schema, and provides user guidance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class DatasetTemplate:
    """Template defining required and optional columns for a dataset type"""
    name: str
    display_name: str
    description: str
    required_columns: List[str]  # Column patterns (any must match)
    optional_columns: List[str]
    example_columns: List[str]
    icon: str


# Dataset type templates
DATASET_TEMPLATES = {
    "traffic": DatasetTemplate(
        name="traffic",
        display_name="Traffic & Accidents",
        description="Traffic accident data with locations, dates, and incident counts",
        required_columns=["state|city|location|region", "accidents|incidents|count|total"],
        optional_columns=["date|month|year|time", "severity|fatality|deaths|injured"],
        example_columns=["State", "Month", "Accidents", "Fatalities"],
        icon="🚗"
    ),
    "healthcare": DatasetTemplate(
        name="healthcare",
        display_name="Healthcare",
        description="Healthcare data with patient counts, diagnoses, or medical metrics",
        required_columns=["patient|cases|admission|diagnosis", "hospital|facility|region|state"],
        optional_columns=["date|month|year", "age|gender|department"],
        example_columns=["Hospital", "Date", "Admissions", "Department"],
        icon="🏥"
    ),
    "aadhaar": DatasetTemplate(
        name="aadhaar",
        display_name="Aadhaar/ID Data",
        description="Identity enrollment or verification data",
        required_columns=["enrollment|aadhaar|uid|id", "state|district|region"],
        optional_columns=["date|month|year", "gender|age|status"],
        example_columns=["State", "Enrollments", "Date", "Status"],
        icon="🪪"
    ),
    "generic": DatasetTemplate(
        name="generic",
        display_name="Generic Dataset",
        description="Any tabular data - we'll analyze what we can find",
        required_columns=[],
        optional_columns=[],
        example_columns=["Any columns with numeric data"],
        icon="📊"
    )
}


class ValidationWarning:
    """Represents a validation warning/suggestion"""
    def __init__(self, level: str, message: str, suggestion: str = None):
        self.level = level  # 'error', 'warning', 'info'
        self.message = message
        self.suggestion = suggestion
    
    def to_dict(self):
        return {
            "level": self.level,
            "message": self.message,
            "suggestion": self.suggestion
        }


class DataValidator:
    """Validate and analyze uploaded datasets with user guidance"""
    
    # Common column name patterns for auto-detection
    DATE_PATTERNS = ['date', 'datetime', 'time', 'timestamp', 'period', 'month', 'year']
    LOCATION_PATTERNS = ['state', 'city', 'location', 'region', 'area', 'district', 'zone']
    COUNT_PATTERNS = ['count', 'total', 'accidents', 'incidents', 'cases', 'number', 'volume', 'enrollments', 'admissions']
    SEVERITY_PATTERNS = ['severity', 'fatality', 'fatal', 'injury', 'deaths', 'killed', 'injured']
    
    def __init__(self):
        self.supported_formats = {'.csv', '.xlsx', '.xls'}
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Return list of available dataset templates for UI"""
        return [
            {
                "name": t.name,
                "display_name": t.display_name,
                "description": t.description,
                "icon": t.icon,
                "example_columns": t.example_columns
            }
            for t in DATASET_TEMPLATES.values()
        ]
    
    def validate_and_detect(self, filepath: str, expected_type: str = None) -> Dict[str, Any]:
        """
        Validate file and auto-detect schema
        Returns validation result with column info, detected types, and warnings
        """
        path = Path(filepath)
        warnings = []
        
        # Check file exists
        if not path.exists():
            raise ValueError(f"File not found: {filepath}")
        
        # Check extension
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Load data
        df = self._load_file(filepath)
        
        # Basic validation
        if df.empty:
            raise ValueError("File is empty")
        
        if len(df.columns) < 2:
            raise ValueError("File must have at least 2 columns")
        
        # Detect schema
        schema = self._detect_schema(df)
        
        # Detect best matching template
        detected_template, confidence, template_warnings = self._match_template(df, schema)
        warnings.extend(template_warnings)
        
        # If user specified expected type, validate against it
        if expected_type and expected_type != "generic":
            type_warnings = self._validate_against_template(df, expected_type, schema)
            warnings.extend(type_warnings)
        
        # Get column info
        column_info = self._get_column_info(df)
        
        # Generate human-readable explanation
        explanation = self._generate_explanation(schema, detected_template, df)
        
        return {
            "valid": True,
            "rows": len(df),
            "columns": list(df.columns),
            "schema": schema,
            "column_info": column_info,
            "preview": df.head(5).to_dict(orient='records'),
            # New fields for user clarity
            "detected_template": detected_template,
            "template_confidence": confidence,
            "warnings": [w.to_dict() for w in warnings],
            "explanation": explanation,
            "column_mapping_suggestions": self._get_mapping_suggestions(df, schema)
        }
    
    def _match_template(self, df: pd.DataFrame, schema: Dict) -> Tuple[str, float, List[ValidationWarning]]:
        """Match dataset against templates and return best match with confidence"""
        warnings = []
        best_match = "generic"
        best_score = 0
        
        columns_lower = [c.lower() for c in df.columns]
        
        for template_name, template in DATASET_TEMPLATES.items():
            if template_name == "generic":
                continue
            
            score = 0
            required_matched = 0
            required_total = len(template.required_columns)
            
            # Check required columns
            for pattern_group in template.required_columns:
                patterns = pattern_group.split("|")
                if any(any(p in col for p in patterns) for col in columns_lower):
                    required_matched += 1
                    score += 2
            
            # Check optional columns
            for pattern_group in template.optional_columns:
                patterns = pattern_group.split("|")
                if any(any(p in col for p in patterns) for col in columns_lower):
                    score += 1
            
            # Calculate confidence
            if required_total > 0:
                confidence = required_matched / required_total
            else:
                confidence = 0.5
            
            if score > best_score:
                best_score = score
                best_match = template_name
        
        # Calculate final confidence
        if best_match != "generic":
            template = DATASET_TEMPLATES[best_match]
            confidence = min(1.0, best_score / (len(template.required_columns) * 2 + len(template.optional_columns)))
            
            if confidence < 0.5:
                warnings.append(ValidationWarning(
                    level="warning",
                    message=f"Dataset partially matches '{template.display_name}' template ({int(confidence*100)}% confidence)",
                    suggestion="Some expected columns may be missing. Analysis will proceed with available data."
                ))
        else:
            confidence = 0.3
            warnings.append(ValidationWarning(
                level="info",
                message="Dataset type could not be determined automatically",
                suggestion="We'll analyze this as a generic dataset. For best results, ensure your data has clear column names."
            ))
        
        return best_match, confidence, warnings
    
    def _validate_against_template(self, df: pd.DataFrame, expected_type: str, schema: Dict) -> List[ValidationWarning]:
        """Validate dataset against user's expected template"""
        warnings = []
        
        if expected_type not in DATASET_TEMPLATES:
            return warnings
        
        template = DATASET_TEMPLATES[expected_type]
        columns_lower = [c.lower() for c in df.columns]
        
        # Check required columns
        missing_required = []
        for pattern_group in template.required_columns:
            patterns = pattern_group.split("|")
            if not any(any(p in col for p in patterns) for col in columns_lower):
                missing_required.append(pattern_group.replace("|", " or "))
        
        if missing_required:
            warnings.append(ValidationWarning(
                level="error",
                message=f"Missing required columns for {template.display_name}",
                suggestion=f"Expected columns matching: {', '.join(missing_required)}"
            ))
        
        return warnings
    
    def _generate_explanation(self, schema: Dict, detected_type: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate human-readable explanation of what was detected"""
        template = DATASET_TEMPLATES.get(detected_type, DATASET_TEMPLATES["generic"])
        
        # Describe what we found
        found_items = []
        if schema["date_columns"]:
            found_items.append(f"📅 Time/Date columns: {', '.join(schema['date_columns'])}")
        if schema["location_columns"]:
            found_items.append(f"📍 Location columns: {', '.join(schema['location_columns'])}")
        if schema["count_columns"]:
            found_items.append(f"🔢 Count/Metric columns: {', '.join(schema['count_columns'])}")
        if schema["severity_columns"]:
            found_items.append(f"⚠️ Severity columns: {', '.join(schema['severity_columns'])}")
        
        # What analyses will run
        analyses = []
        if schema["location_columns"] or schema["categorical_columns"]:
            analyses.append("📊 **Pattern Analysis** - Find trends by location/category")
        if schema["date_columns"]:
            analyses.append("📈 **Forecasting** - Predict future trends")
        if schema["count_columns"] or schema["numeric_columns"]:
            analyses.append("🔍 **Anomaly Detection** - Find unusual values")
            analyses.append("🏷️ **Classification** - Categorize into levels (Low/Medium/High/Severe)")
        
        return {
            "detected_as": template.display_name,
            "icon": template.icon,
            "what_we_found": found_items,
            "what_we_will_do": analyses,
            "summary": f"Your dataset has {len(df)} rows and {len(df.columns)} columns. " +
                      f"Detected as {template.display_name} data with {len(found_items)} key column types identified."
        }
    
    def _get_mapping_suggestions(self, df: pd.DataFrame, schema: Dict) -> List[Dict[str, Any]]:
        """Get column mapping suggestions for user confirmation"""
        suggestions = []
        
        mapping = schema.get("column_mapping", {})
        
        if mapping.get("date"):
            suggestions.append({
                "role": "Date/Time",
                "column": mapping["date"],
                "icon": "📅",
                "confirmed": False
            })
        
        if mapping.get("location"):
            suggestions.append({
                "role": "Location",
                "column": mapping["location"],
                "icon": "📍",
                "confirmed": False
            })
        
        if mapping.get("count"):
            suggestions.append({
                "role": "Count/Metric",
                "column": mapping["count"],
                "icon": "🔢",
                "confirmed": False
            })
        
        if mapping.get("severity"):
            suggestions.append({
                "role": "Severity",
                "column": mapping["severity"],
                "icon": "⚠️",
                "confirmed": False
            })
        
        return suggestions
    
    def _load_file(self, filepath: str) -> pd.DataFrame:
        """Load CSV or Excel file into DataFrame"""
        path = Path(filepath)
        
        try:
            if path.suffix.lower() == '.csv':
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode file with common encodings")
            else:
                df = pd.read_excel(filepath)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to read file: {str(e)}")
    
    def _detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Auto-detect the schema based on column names and data types"""
        schema = {
            "date_columns": [],
            "location_columns": [],
            "count_columns": [],
            "severity_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "detected_type": "generic"
        }
        
        for col in df.columns:
            col_lower = col.lower()
            dtype = df[col].dtype
            
            if any(pattern in col_lower for pattern in self.DATE_PATTERNS):
                schema["date_columns"].append(col)
            elif any(pattern in col_lower for pattern in self.LOCATION_PATTERNS):
                schema["location_columns"].append(col)
            elif any(pattern in col_lower for pattern in self.COUNT_PATTERNS):
                schema["count_columns"].append(col)
            elif any(pattern in col_lower for pattern in self.SEVERITY_PATTERNS):
                schema["severity_columns"].append(col)
            elif np.issubdtype(dtype, np.number):
                schema["numeric_columns"].append(col)
            else:
                schema["categorical_columns"].append(col)
        
        # Determine dataset type
        if schema["count_columns"] and (schema["date_columns"] or schema["location_columns"]):
            if schema["severity_columns"]:
                schema["detected_type"] = "traffic_accidents"
            else:
                schema["detected_type"] = "traffic_counts"
        elif schema["date_columns"] and schema["numeric_columns"]:
            schema["detected_type"] = "time_series"
        else:
            schema["detected_type"] = "generic"
        
        schema["column_mapping"] = self._create_column_mapping(schema)
        
        return schema
    
    def _create_column_mapping(self, schema: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Create mapping from expected column names to actual column names"""
        mapping = {
            "date": schema["date_columns"][0] if schema["date_columns"] else None,
            "location": schema["location_columns"][0] if schema["location_columns"] else None,
            "count": schema["count_columns"][0] if schema["count_columns"] else None,
            "severity": schema["severity_columns"][0] if schema["severity_columns"] else None,
        }
        
        if not mapping["count"] and schema["numeric_columns"]:
            mapping["count"] = schema["numeric_columns"][0]
        
        return mapping
    
    def _get_column_info(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get detailed info for each column"""
        info_list = []
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique())
            }
            
            if np.issubdtype(df[col].dtype, np.number):
                col_info["min"] = float(df[col].min()) if pd.notna(df[col].min()) else None
                col_info["max"] = float(df[col].max()) if pd.notna(df[col].max()) else None
                col_info["mean"] = float(df[col].mean()) if pd.notna(df[col].mean()) else None
            else:
                col_info["sample_values"] = df[col].dropna().head(5).tolist()
            
            info_list.append(col_info)
        
        return info_list
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load and return the dataset as DataFrame"""
        return self._load_file(filepath)
