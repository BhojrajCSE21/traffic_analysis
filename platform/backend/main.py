# Traffic Analytics Platform - Backend
# FastAPI-based backend for processing user-uploaded datasets

"""
Main FastAPI Application
Traffic Analytics Platform API
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Analytics Platform",
    description="Upload datasets, analyze with ML, and get insights",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory storage for demo (would use database in production)
datasets: Dict[str, Dict[str, Any]] = {}
analyses: Dict[str, Dict[str, Any]] = {}


# Pydantic models
class DatasetInfo(BaseModel):
    id: str
    filename: str
    upload_time: str
    status: str
    rows: Optional[int] = None
    columns: Optional[List[str]] = None


class AnalysisResult(BaseModel):
    id: str
    dataset_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class AnalysisRequest(BaseModel):
    analysis_types: Optional[List[str]] = None  # None = run all
    expected_type: Optional[str] = None  # User-specified dataset type


# Import services (will be created next)
from services.validator import DataValidator
from services.orchestrator import AnalysisOrchestrator
from services.visualization import VisualizationService
from services.pdf_generator import PDFReportGenerator


# Initialize services
validator = DataValidator()
orchestrator = AnalysisOrchestrator()
viz_service = VisualizationService()
pdf_generator = PDFReportGenerator()


# ============ TEMPLATES ENDPOINT ============

@app.get("/api/templates")
async def get_templates():
    """Get available dataset templates for user selection"""
    return {
        "templates": validator.get_available_templates(),
        "message": "Select a template that best matches your data, or choose 'Generic' for any dataset"
    }


# ============ UPLOAD ENDPOINTS ============

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or Excel file for analysis"""
    
    # Validate file extension
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (max 50MB)
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB allowed.")
    
    # Generate unique ID and save file
    dataset_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{dataset_id}_{file.filename}"
    
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Validate and get schema info
    try:
        validation_result = validator.validate_and_detect(str(save_path))
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Invalid data file: {str(e)}")
    
    # Store dataset info with full validation details
    dataset_info = {
        "id": dataset_id,
        "filename": file.filename,
        "filepath": str(save_path),
        "upload_time": datetime.now().isoformat(),
        "status": "pending_confirmation",  # Needs user confirmation
        "rows": validation_result.get("rows", 0),
        "columns": validation_result.get("columns", []),
        "schema": validation_result.get("schema", {}),
        "detected_template": validation_result.get("detected_template", "generic"),
        "template_confidence": validation_result.get("template_confidence", 0),
        "warnings": validation_result.get("warnings", []),
        "explanation": validation_result.get("explanation", {}),
        "column_mapping": validation_result.get("column_mapping_suggestions", [])
    }
    datasets[dataset_id] = dataset_info
    
    # Return full validation info for user confirmation
    return {
        "id": dataset_id,
        "filename": file.filename,
        "upload_time": dataset_info["upload_time"],
        "status": "pending_confirmation",
        "rows": dataset_info["rows"],
        "columns": dataset_info["columns"],
        # New fields for user clarity
        "detected_template": dataset_info["detected_template"],
        "template_confidence": dataset_info["template_confidence"],
        "warnings": dataset_info["warnings"],
        "explanation": dataset_info["explanation"],
        "column_mapping": dataset_info["column_mapping"],
        "preview": validation_result.get("preview", [])
    }


@app.get("/api/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List all uploaded datasets"""
    return [
        DatasetInfo(
            id=d["id"],
            filename=d["filename"],
            upload_time=d["upload_time"],
            status=d["status"],
            rows=d.get("rows"),
            columns=d.get("columns")
        )
        for d in datasets.values()
    ]


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset details including schema"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return datasets[dataset_id]


class ConfirmSchemaRequest(BaseModel):
    confirmed_template: Optional[str] = None
    column_overrides: Optional[Dict[str, str]] = None  # {"role": "column_name"}


@app.post("/api/datasets/{dataset_id}/confirm")
async def confirm_dataset_schema(dataset_id: str, request: ConfirmSchemaRequest = None):
    """Confirm dataset schema and mark ready for analysis"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Update with user confirmations
    if request and request.confirmed_template:
        datasets[dataset_id]["confirmed_template"] = request.confirmed_template
    
    if request and request.column_overrides:
        # Update schema with user's column mappings
        schema = datasets[dataset_id].get("schema", {})
        mapping = schema.get("column_mapping", {})
        mapping.update(request.column_overrides)
        schema["column_mapping"] = mapping
        datasets[dataset_id]["schema"] = schema
    
    # Mark as ready
    datasets[dataset_id]["status"] = "ready"
    
    return {
        "message": "Schema confirmed! Dataset is ready for analysis.",
        "dataset_id": dataset_id,
        "status": "ready"
    }


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset and its files"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Remove file
    filepath = datasets[dataset_id].get("filepath")
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
    
    # Remove from storage
    del datasets[dataset_id]
    return {"message": "Dataset deleted successfully"}


# ============ ANALYSIS ENDPOINTS ============

@app.post("/api/analyze/{dataset_id}")
async def run_analysis(
    dataset_id: str, 
    background_tasks: BackgroundTasks,
    request: AnalysisRequest = None
):
    """Run ML analysis on a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    analysis_id = str(uuid.uuid4())[:8]
    
    # Create analysis record
    analysis_info = {
        "id": analysis_id,
        "dataset_id": dataset_id,
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "results": None,
        "charts": []
    }
    analyses[analysis_id] = analysis_info
    
    # Run analysis in background
    background_tasks.add_task(
        run_analysis_pipeline,
        analysis_id,
        datasets[dataset_id],
        request.analysis_types if request else None
    )
    
    return {
        "analysis_id": analysis_id,
        "status": "processing",
        "message": "Analysis started. Poll /api/results/{analysis_id} for progress."
    }


async def run_analysis_pipeline(
    analysis_id: str,
    dataset_info: Dict[str, Any],
    analysis_types: Optional[List[str]] = None
):
    """Background task to run the full analysis pipeline"""
    try:
        # Run orchestrator
        results = orchestrator.run_analysis(
            filepath=dataset_info["filepath"],
            schema=dataset_info["schema"],
            analysis_types=analysis_types
        )
        
        # Generate visualizations
        charts = viz_service.generate_charts(
            results=results,
            analysis_id=analysis_id,
            output_dir=str(RESULTS_DIR)
        )
        
        # Update analysis record
        analyses[analysis_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "results": results,
            "charts": charts
        })
        
    except Exception as e:
        analyses[analysis_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


@app.get("/api/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Get analysis results"""
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analyses[analysis_id]


@app.get("/api/charts/{analysis_id}")
async def get_charts(analysis_id: str):
    """Get list of generated chart files"""
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"charts": analyses[analysis_id].get("charts", [])}


@app.get("/api/charts/{analysis_id}/{chart_name}")
async def get_chart_file(analysis_id: str, chart_name: str):
    """Get a specific chart file"""
    chart_path = RESULTS_DIR / analysis_id / chart_name
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(str(chart_path))


@app.get("/api/export/{analysis_id}/pdf")
async def export_pdf_report(analysis_id: str):
    """Generate and download PDF report for analysis"""
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses[analysis_id]
    if analysis.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    # Get dataset info
    dataset_id = analysis.get("dataset_id")
    dataset_info = datasets.get(dataset_id, {"filename": "Unknown"})
    
    # Generate PDF
    try:
        pdf_bytes = pdf_generator.generate_to_bytes(analysis, dataset_info)
        
        # Return as downloadable file
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=analysis_report_{analysis_id}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


# ============ HEALTH & STATIC ============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Serve frontend static files
frontend_path = BASE_DIR / "platform" / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
