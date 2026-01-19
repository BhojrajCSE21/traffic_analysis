/**
 * Traffic Analytics Platform - Main Application
 * Handles file upload, API communication, and dashboard rendering
 */

// ============================================
// Configuration
// ============================================

// Use Railway backend for production, local for development
const API_BASE = window.location.hostname === 'localhost' 
    ? '/api' 
    : 'https://flowiqintelligenttrafficanalytics-production.up.railway.app/api';

// State
const state = {
    currentDataset: null,
    currentAnalysis: null,
    datasets: [],
    analyses: []
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initUploadZone();
    initNavigation();
    initButtons();
    loadHistory();
    updateStats();
});

// ============================================
// Upload Zone
// ============================================

function initUploadZone() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    
    // Click to browse
    uploadZone.addEventListener('click', () => fileInput.click());
    
    // File input change - handle multiple files
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleMultipleFiles(Array.from(e.target.files));
        }
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length > 0) {
            handleMultipleFiles(Array.from(e.dataTransfer.files));
        }
    });
}

// Handle multiple file uploads
async function handleMultipleFiles(files) {
    if (files.length === 1) {
        // Single file - use existing flow
        handleFileUpload(files[0]);
        return;
    }
    
    // Multiple files - show batch upload UI
    showToast(`Uploading ${files.length} files...`, 'info');
    
    const validFiles = files.filter(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        return ['.csv', '.xlsx', '.xls'].includes(ext);
    });
    
    if (validFiles.length !== files.length) {
        showToast(`Skipped ${files.length - validFiles.length} invalid files`, 'warning');
    }
    
    const uploadedDatasets = [];
    
    for (let i = 0; i < validFiles.length; i++) {
        const file = validFiles[i];
        showUploadProgress(file.name);
        updateUploadProgress((i / validFiles.length) * 100, `Uploading ${i + 1}/${validFiles.length}: ${file.name}`);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const dataset = await response.json();
                uploadedDatasets.push(dataset);
            }
        } catch (error) {
            console.error(`Failed to upload ${file.name}:`, error);
        }
    }
    
    hideUploadProgress();
    
    if (uploadedDatasets.length > 0) {
        showToast(`Successfully uploaded ${uploadedDatasets.length} files!`, 'success');
        state.uploadedBatch = uploadedDatasets;
        
        // If multiple files, show batch selection
        if (uploadedDatasets.length > 1) {
            showBatchSelection(uploadedDatasets);
        } else {
            state.currentDataset = uploadedDatasets[0];
            showSchemaConfirmation(uploadedDatasets[0]);
        }
    } else {
        showToast('No files were uploaded successfully', 'error');
    }
}

// Show batch selection UI for multiple uploads
function showBatchSelection(datasets) {
    document.getElementById('upload').hidden = true;
    document.getElementById('schema-confirm').hidden = true;
    
    // Create batch selection modal
    const container = document.querySelector('.container');
    let batchSection = document.getElementById('batch-selection');
    
    if (!batchSection) {
        batchSection = document.createElement('section');
        batchSection.id = 'batch-selection';
        batchSection.className = 'section';
        container.insertBefore(batchSection, document.getElementById('schema-confirm'));
    }
    
    batchSection.hidden = false;
    batchSection.innerHTML = `
        <div class="section-header">
            <h2><i class="fas fa-layer-group"></i> Select Dataset to Analyze</h2>
            <p>You uploaded ${datasets.length} files. Select one to analyze:</p>
        </div>
        <div class="batch-grid">
            ${datasets.map((d, i) => `
                <div class="batch-item" data-index="${i}">
                    <div class="batch-icon">${d.explanation?.icon || '📊'}</div>
                    <div class="batch-info">
                        <h4>${escapeHtml(d.filename)}</h4>
                        <p>${d.rows?.toLocaleString() || 0} rows • ${d.columns?.length || 0} columns</p>
                        <span class="batch-type">${d.explanation?.detected_as || 'Generic'}</span>
                    </div>
                    <button class="btn btn-primary btn-sm" onclick="selectBatchItem(${i})">
                        <i class="fas fa-arrow-right"></i> Analyze
                    </button>
                </div>
            `).join('')}
        </div>
        <div class="confirm-actions">
            <button class="btn btn-secondary" onclick="cancelBatchSelection()">
                <i class="fas fa-times"></i> Cancel
            </button>
        </div>
    `;
}

// Select a dataset from batch
window.selectBatchItem = function(index) {
    const dataset = state.uploadedBatch[index];
    state.currentDataset = dataset;
    document.getElementById('batch-selection').hidden = true;
    showSchemaConfirmation(dataset);
};

// Cancel batch selection
window.cancelBatchSelection = function() {
    document.getElementById('batch-selection').hidden = true;
    document.getElementById('upload').hidden = false;
    document.getElementById('file-input').value = '';
};


async function handleFileUpload(file) {
    // Validate file type
    const validTypes = ['.csv', '.xlsx', '.xls'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(fileExt)) {
        showToast('Invalid file type. Please upload CSV or Excel files.', 'error');
        return;
    }
    
    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
        showToast('File too large. Maximum size is 50MB.', 'error');
        return;
    }
    
    // Show progress
    showUploadProgress(file.name);
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Upload file
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const dataset = await response.json();
        state.currentDataset = dataset;
        
        // Update progress
        updateUploadProgress(100, 'Upload complete!');
        showToast('File uploaded successfully!', 'success');
        
        // Show schema confirmation step (NEW: instead of going directly to analysis)
        setTimeout(() => {
            hideUploadProgress();
            showSchemaConfirmation(dataset);
        }, 1000);
        
    } catch (error) {
        updateUploadProgress(0, 'Upload failed');
        showToast(error.message, 'error');
        console.error('Upload error:', error);
    }
}

// ============================================
// Schema Confirmation (NEW)
// ============================================

function showSchemaConfirmation(dataset) {
    // Hide other sections, show confirmation
    document.getElementById('upload').hidden = true;
    document.getElementById('schema-confirm').hidden = false;
    document.getElementById('analysis').hidden = true;
    document.getElementById('dashboard').hidden = true;
    
    // Get explanation data
    const explanation = dataset.explanation || {};
    const templateNames = {
        'traffic': 'Traffic & Accidents',
        'healthcare': 'Healthcare',
        'aadhaar': 'Aadhaar/ID Data',
        'generic': 'Generic Dataset'
    };
    const templateIcons = {
        'traffic': '🚗',
        'healthcare': '🏥',
        'aadhaar': '🪪',
        'generic': '📊'
    };
    
    // Update detection header
    document.getElementById('detection-icon').textContent = explanation.icon || templateIcons[dataset.detected_template] || '📊';
    document.getElementById('detected-type-name').textContent = explanation.detected_as || templateNames[dataset.detected_template] || 'Generic Dataset';
    document.getElementById('detected-type-desc').textContent = explanation.summary || `${dataset.rows} rows, ${dataset.columns?.length || 0} columns`;
    document.getElementById('confidence-value').textContent = `${Math.round((dataset.template_confidence || 0) * 100)}%`;
    
    // Show warnings if any
    const warningsCard = document.getElementById('warnings-card');
    const warningsList = document.getElementById('warnings-list');
    if (dataset.warnings && dataset.warnings.length > 0) {
        warningsCard.hidden = false;
        warningsList.innerHTML = dataset.warnings.map(w => `
            <li>
                <span class="warning-message">${escapeHtml(w.message)}</span>
                ${w.suggestion ? `<span class="warning-suggestion">💡 ${escapeHtml(w.suggestion)}</span>` : ''}
            </li>
        `).join('');
    } else {
        warningsCard.hidden = true;
    }
    
    // What we found
    const foundList = document.getElementById('found-list');
    const foundItems = explanation.what_we_found || [];
    foundList.innerHTML = foundItems.length > 0 
        ? foundItems.map(item => `<li>${escapeHtml(item)}</li>`).join('')
        : '<li>No specific column patterns detected</li>';
    
    // What we'll analyze
    const analyzeList = document.getElementById('analyze-list');
    const analyzeItems = explanation.what_we_will_do || [];
    analyzeList.innerHTML = analyzeItems.length > 0
        ? analyzeItems.map(item => `<li>${escapeHtml(item)}</li>`).join('')
        : '<li>Standard statistical analysis</li>';
    
    // Column mappings
    const mappingGrid = document.getElementById('mapping-grid');
    const mappings = dataset.column_mapping || [];
    if (mappings.length > 0) {
        mappingGrid.innerHTML = mappings.map(m => `
            <div class="mapping-item">
                <span class="mapping-icon">${m.icon || '📋'}</span>
                <div class="mapping-info">
                    <span class="mapping-role">${escapeHtml(m.role)}</span>
                    <span class="mapping-column">${escapeHtml(m.column)}</span>
                </div>
            </div>
        `).join('');
    } else {
        mappingGrid.innerHTML = '<p>No specific column mappings detected</p>';
    }
    
    // Data preview
    const previewHead = document.getElementById('preview-head');
    const previewBody = document.getElementById('preview-body');
    const preview = dataset.preview || [];
    
    if (preview.length > 0 && dataset.columns) {
        previewHead.innerHTML = `<tr>${dataset.columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr>`;
        previewBody.innerHTML = preview.map(row => 
            `<tr>${dataset.columns.map(c => `<td>${escapeHtml(String(row[c] ?? ''))}</td>`).join('')}</tr>`
        ).join('');
    }
    
    // Setup button handlers
    document.getElementById('btn-cancel-confirm').onclick = () => {
        document.getElementById('schema-confirm').hidden = true;
        document.getElementById('upload').hidden = false;
        document.getElementById('file-input').value = '';
    };
    
    document.getElementById('btn-proceed-analysis').onclick = async () => {
        // Confirm schema and start analysis
        try {
            await fetch(`${API_BASE}/datasets/${dataset.id}/confirm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            
            document.getElementById('schema-confirm').hidden = true;
            startAnalysis(dataset.id);
        } catch (error) {
            showToast('Failed to confirm schema', 'error');
        }
    };
}


function showUploadProgress(filename) {
    const progress = document.getElementById('upload-progress');
    const filenameEl = document.getElementById('progress-filename');
    const percentEl = document.getElementById('progress-percent');
    const fillEl = document.getElementById('progress-fill');
    const statusEl = document.getElementById('progress-status');
    
    filenameEl.textContent = filename;
    percentEl.textContent = '0%';
    fillEl.style.width = '0%';
    statusEl.textContent = 'Uploading...';
    progress.hidden = false;
    
    // Animate progress (simulated)
    let percent = 0;
    const interval = setInterval(() => {
        percent += Math.random() * 15;
        if (percent > 90) {
            clearInterval(interval);
            percent = 90;
        }
        updateUploadProgress(percent, 'Uploading...');
    }, 200);
    
    // Store interval for cleanup
    progress.dataset.interval = interval;
}

function updateUploadProgress(percent, status) {
    const percentEl = document.getElementById('progress-percent');
    const fillEl = document.getElementById('progress-fill');
    const statusEl = document.getElementById('progress-status');
    
    percentEl.textContent = `${Math.round(percent)}%`;
    fillEl.style.width = `${percent}%`;
    statusEl.textContent = status;
}

function hideUploadProgress() {
    const progress = document.getElementById('upload-progress');
    const interval = progress.dataset.interval;
    if (interval) clearInterval(parseInt(interval));
    progress.hidden = true;
}

// ============================================
// Analysis
// ============================================

async function startAnalysis(datasetId) {
    // Show analysis section
    document.getElementById('upload').hidden = true;
    document.getElementById('analysis').hidden = false;
    document.getElementById('dashboard').hidden = true;
    
    // Reset steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'completed');
    });
    
    // Animate steps
    const steps = ['validate', 'patterns', 'forecast', 'anomaly', 'charts'];
    
    try {
        // Start analysis
        const response = await fetch(`${API_BASE}/analyze/${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        
        if (!response.ok) {
            throw new Error('Failed to start analysis');
        }
        
        const result = await response.json();
        state.currentAnalysis = { id: result.analysis_id, status: 'processing' };
        
        // Animate through steps while polling
        let stepIndex = 0;
        const stepInterval = setInterval(() => {
            if (stepIndex > 0) {
                document.querySelector(`[data-step="${steps[stepIndex - 1]}"]`).classList.remove('active');
                document.querySelector(`[data-step="${steps[stepIndex - 1]}"]`).classList.add('completed');
            }
            if (stepIndex < steps.length) {
                document.querySelector(`[data-step="${steps[stepIndex]}"]`).classList.add('active');
                stepIndex++;
            }
        }, 1500);
        
        // Poll for results
        const analysisResult = await pollAnalysisResults(result.analysis_id);
        clearInterval(stepInterval);
        
        // Mark all steps completed
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active');
            step.classList.add('completed');
        });
        
        // Show dashboard
        setTimeout(() => {
            showDashboard(analysisResult);
        }, 500);
        
    } catch (error) {
        showToast(error.message, 'error');
        document.getElementById('analysis-status').textContent = 'Analysis failed: ' + error.message;
    }
}

async function pollAnalysisResults(analysisId, maxAttempts = 30) {
    for (let i = 0; i < maxAttempts; i++) {
        try {
            const response = await fetch(`${API_BASE}/results/${analysisId}`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                return data;
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Analysis failed');
            }
            
            // Wait before next poll
            await new Promise(resolve => setTimeout(resolve, 2000));
            
        } catch (error) {
            console.error('Poll error:', error);
            throw error;
        }
    }
    
    throw new Error('Analysis timed out');
}

// ============================================
// Dashboard
// ============================================

function showDashboard(analysisResult) {
    document.getElementById('upload').hidden = true;
    document.getElementById('analysis').hidden = true;
    document.getElementById('dashboard').hidden = false;
    
    // Render summary cards
    renderSummaryCards(analysisResult.results);
    
    // Render findings
    renderFindings(analysisResult.results);
    
    // Render recommendations
    renderRecommendations(analysisResult.results);
    
    // Render charts
    renderCharts(analysisResult.charts, analysisResult.id);
    
    // Update stats
    updateStats();
    
    showToast('Analysis complete!', 'success');
}

function renderSummaryCards(results) {
    const container = document.getElementById('summary-cards');
    const dataset = results.dataset_info || {};
    const summary = results.summary || {};
    const analyses = results.analyses || {};
    
    const cards = [
        {
            icon: 'fa-database',
            iconClass: 'primary',
            value: dataset.rows?.toLocaleString() || '0',
            label: 'Data Rows'
        },
        {
            icon: 'fa-chart-bar',
            iconClass: 'success',
            value: summary.completed || 0,
            label: 'Analyses Completed'
        },
        {
            icon: 'fa-exclamation-triangle',
            iconClass: 'warning',
            value: analyses.anomaly?.anomalies?.length || 0,
            label: 'Anomalies Found'
        },
        {
            icon: 'fa-arrow-trend-up',
            iconClass: 'danger',
            value: analyses.forecasting?.trend?.toUpperCase() || 'N/A',
            label: 'Trend Direction'
        }
    ];
    
    container.innerHTML = cards.map(card => `
        <div class="summary-card">
            <div class="summary-icon ${card.iconClass}">
                <i class="fas ${card.icon}"></i>
            </div>
            <div class="summary-content">
                <h4>${card.value}</h4>
                <p>${card.label}</p>
            </div>
        </div>
    `).join('');
}

function renderFindings(results) {
    const container = document.getElementById('findings-list');
    const findings = results.summary?.key_findings || [];
    
    if (findings.length === 0) {
        container.innerHTML = '<li>No significant findings detected.</li>';
        return;
    }
    
    container.innerHTML = findings.map(finding => `
        <li>${escapeHtml(finding)}</li>
    `).join('');
}

function renderRecommendations(results) {
    const container = document.getElementById('recommendations-list');
    const recommendations = results.recommendations || [];
    
    if (recommendations.length === 0) {
        container.innerHTML = '<li>No specific recommendations at this time.</li>';
        return;
    }
    
    container.innerHTML = recommendations.map(rec => `
        <li>${escapeHtml(rec)}</li>
    `).join('');
}

function renderCharts(charts, analysisId) {
    const container = document.getElementById('charts-container');
    
    if (!charts || charts.length === 0) {
        container.innerHTML = '<p>No charts generated.</p>';
        return;
    }
    
    container.innerHTML = charts.map(chart => `
        <div class="chart-card">
            <div class="chart-header">${escapeHtml(chart.title)}</div>
            <div class="chart-body">
                <iframe src="${API_BASE}/charts/${analysisId}/${chart.name}" loading="lazy"></iframe>
            </div>
        </div>
    `).join('');
}

// ============================================
// Navigation
// ============================================

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Show corresponding section
            const targetId = link.getAttribute('href').substring(1);
            showSection(targetId);
        });
    });
}

function showSection(sectionId) {
    if (sectionId === 'upload') {
        document.getElementById('upload').hidden = false;
        document.getElementById('analysis').hidden = true;
        document.getElementById('dashboard').hidden = true;
    } else if (sectionId === 'dashboard') {
        if (state.currentAnalysis) {
            document.getElementById('upload').hidden = true;
            document.getElementById('analysis').hidden = true;
            document.getElementById('dashboard').hidden = false;
        } else {
            showToast('No analysis results. Upload a dataset first.', 'warning');
        }
    }
}

// ============================================
// Buttons
// ============================================

function initButtons() {
    document.getElementById('btn-new-analysis')?.addEventListener('click', () => {
        state.currentDataset = null;
        state.currentAnalysis = null;
        document.getElementById('upload').hidden = false;
        document.getElementById('analysis').hidden = true;
        document.getElementById('dashboard').hidden = true;
        document.getElementById('file-input').value = '';
    });
    
    document.getElementById('btn-export')?.addEventListener('click', () => {
        exportResults();
    });
}

function exportResults() {
    if (!state.currentAnalysis) {
        showToast('No results to export.', 'warning');
        return;
    }
    
    // Export as PDF
    const analysisId = state.currentAnalysis.id;
    showToast('Generating PDF report...', 'info');
    
    // Open PDF download in new tab / trigger download
    const pdfUrl = `${API_BASE}/export/${analysisId}/pdf`;
    
    // Use fetch to handle the download
    fetch(pdfUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to generate PDF');
            }
            return response.blob();
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `analysis_report_${analysisId}.pdf`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showToast('PDF report downloaded!', 'success');
        })
        .catch(error => {
            console.error('PDF export error:', error);
            showToast('Failed to generate PDF report', 'error');
        });
}

// ============================================
// History
// ============================================

async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/datasets`);
        if (response.ok) {
            const datasets = await response.json();
            state.datasets = datasets;
            renderHistory(datasets);
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

function renderHistory(datasets) {
    const container = document.getElementById('history-list');
    
    if (!datasets || datasets.length === 0) {
        container.innerHTML = `
            <div class="history-empty">
                <i class="fas fa-folder-open"></i>
                <p>No analyses yet. Upload a dataset to get started!</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = datasets.map(dataset => `
        <div class="history-item" data-id="${dataset.id}">
            <div class="history-info">
                <h4>${escapeHtml(dataset.filename)}</h4>
                <p>${formatDate(dataset.upload_time)} • ${dataset.rows?.toLocaleString() || 'N/A'} rows</p>
            </div>
            <div class="history-status ${dataset.status}">
                <i class="fas fa-${dataset.status === 'completed' ? 'check-circle' : 'clock'}"></i>
                ${dataset.status}
            </div>
        </div>
    `).join('');
}

// ============================================
// Stats
// ============================================

function updateStats() {
    document.getElementById('stat-datasets').textContent = state.datasets.length;
    document.getElementById('stat-analyses').textContent = state.analyses.length || '0';
}

// ============================================
// Utilities
// ============================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas ${icons[type]} toast-icon"></i>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return unsafe;
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatDate(isoString) {
    try {
        const date = new Date(isoString);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return isoString;
    }
}
