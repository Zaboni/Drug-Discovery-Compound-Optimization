// Drug Discovery Platform - Frontend JavaScript

// Global variables
const API_BASE_URL = '';
let optimizationTargets = [];

// Utility functions
function showLoading() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

function hideLoading() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) modal.hide();
}

function showAlert(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector('main').insertBefore(alertContainer, document.querySelector('main').firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertContainer.parentNode) {
            alertContainer.remove();
        }
    }, 5000);
}

function formatProperty(propertyName) {
    const propertyMap = {
        'molecular_weight': 'Molecular Weight (Da)',
        'logp': 'LogP',
        'tpsa': 'TPSA (Ų)',
        'num_hbd': 'H-bond Donors',
        'num_hba': 'H-bond Acceptors',
        'num_rotatable_bonds': 'Rotatable Bonds',
        'num_aromatic_rings': 'Aromatic Rings'
    };
    return propertyMap[propertyName] || propertyName;
}

function formatValue(value, propertyName) {
    if (typeof value !== 'number') return value;
    
    // Round to appropriate decimal places based on property
    if (['molecular_weight', 'tpsa'].includes(propertyName)) {
        return value.toFixed(2);
    } else if (['logp'].includes(propertyName)) {
        return value.toFixed(3);
    } else if (Number.isInteger(value)) {
        return value.toString();
    } else {
        return value.toFixed(2);
    }
}

// API functions
async function makeAPICall(endpoint, method = 'GET', data = null) {
    try {
        const config = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data && method !== 'GET') {
            config.body = JSON.stringify(data);
        }

        const response = await fetch(`${API_BASE_URL}${endpoint}`, config);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Property Prediction Functions
function addExample(smiles) {
    document.getElementById('smilesInput').value = smiles;
    
    // Check common properties for example
    const commonProps = ['molecular_weight', 'logp', 'tpsa'];
    commonProps.forEach(prop => {
        const checkbox = document.getElementById(`prop_${prop.replace('molecular_weight', 'mw')}`);
        if (checkbox) checkbox.checked = true;
    });
}

function getSelectedProperties() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

async function predictProperties() {
    const smiles = document.getElementById('smilesInput').value.trim();
    const properties = getSelectedProperties();

    if (!smiles) {
        showAlert('Please enter a SMILES string', 'warning');
        return;
    }

    if (properties.length === 0) {
        showAlert('Please select at least one property to predict', 'warning');
        return;
    }

    showLoading();
    try {
        const response = await makeAPICall('/predict_properties', 'POST', {
            smiles: [smiles],
            properties: properties
        });

        displayPropertyResults(response);
    } catch (error) {
        showAlert(`Prediction failed: ${error.message}`, 'danger');
    } finally {
        hideLoading();
    }
}

function displayPropertyResults(response) {
    const resultsContainer = document.getElementById('predictionResults');
    
    if (!response.results || response.results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                No results returned
            </div>
        `;
        return;
    }

    const result = response.results[0];
    
    if (!result.valid) {
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-times me-2"></i>
                Invalid SMILES: ${result.errors ? result.errors.join(', ') : 'Unknown error'}
            </div>
        `;
        return;
    }

    let html = `
        <div class="fade-in">
            <div class="mb-3">
                <strong>SMILES:</strong> <span class="smiles-text">${result.smiles}</span>
            </div>
    `;

    if (result.canonical_smiles && result.canonical_smiles !== result.smiles) {
        html += `
            <div class="mb-3">
                <strong>Canonical:</strong> <span class="smiles-text">${result.canonical_smiles}</span>
            </div>
        `;
    }

    html += '<div class="results-container">';
    
    for (const [property, value] of Object.entries(result.properties)) {
        html += `
            <div class="property-result">
                <div class="property-name">${formatProperty(property)}</div>
                <div class="property-value">${formatValue(value, property)}</div>
            </div>
        `;
    }
    
    html += '</div></div>';
    resultsContainer.innerHTML = html;
}

// Similarity Search Functions
function addSimilarityExample() {
    document.getElementById('querySmiles').value = 'c1ccccc1';
    document.getElementById('targetSmiles').value = `c1ccc2ccccc2c1
c1ccc(cc1)C
c1ccncc1
CCc1ccccc1
c1ccc(cc1)O`;
}

async function calculateSimilarity() {
    const querySmiles = document.getElementById('querySmiles').value.trim();
    const targetSmilesText = document.getElementById('targetSmiles').value.trim();
    const metric = document.getElementById('similarityMetric').value;
    const threshold = parseFloat(document.getElementById('similarityThreshold').value);

    if (!querySmiles) {
        showAlert('Please enter a query SMILES string', 'warning');
        return;
    }

    if (!targetSmilesText) {
        showAlert('Please enter target SMILES strings', 'warning');
        return;
    }

    const targetSmiles = targetSmilesText.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);

    if (targetSmiles.length === 0) {
        showAlert('Please enter at least one target SMILES', 'warning');
        return;
    }

    showLoading();
    try {
        const response = await makeAPICall('/calculate_similarity', 'POST', {
            query_smiles: querySmiles,
            target_smiles: targetSmiles,
            similarity_metric: metric,
            threshold: threshold
        });

        displaySimilarityResults(response);
    } catch (error) {
        showAlert(`Similarity calculation failed: ${error.message}`, 'danger');
    } finally {
        hideLoading();
    }
}

function displaySimilarityResults(response) {
    const resultsContainer = document.getElementById('similarityResults');
    
    if (!response.results || response.results.length === 0) {
        resultsContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                No molecules above the similarity threshold
            </div>
        `;
        return;
    }

    let html = `
        <div class="fade-in">
            <div class="mb-3">
                <strong>Query:</strong> <span class="smiles-text">${response.query_smiles}</span>
            </div>
            <div class="mb-3">
                <small class="text-muted">
                    Found ${response.above_threshold} of ${response.total_comparisons} molecules above threshold
                </small>
            </div>
            <div class="results-container">
    `;

    response.results.forEach((result, index) => {
        const scoreColor = result.similarity_score > 0.7 ? 'text-success' : 
                          result.similarity_score > 0.5 ? 'text-warning' : 'text-danger';
        
        html += `
            <div class="similarity-result">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div><span class="smiles-text">${result.target_smiles}</span></div>
                        <small class="text-muted">${response.metric} similarity</small>
                    </div>
                    <div class="similarity-score ${scoreColor}">
                        ${(result.similarity_score * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div></div>';
    resultsContainer.innerHTML = html;
}

// Optimization Functions
function addOptimizationTarget() {
    const property = document.getElementById('targetProperty').value;
    const value = parseFloat(document.getElementById('targetValue').value);

    if (!value || isNaN(value)) {
        showAlert('Please enter a valid target value', 'warning');
        return;
    }

    const target = { property, value };
    optimizationTargets.push(target);
    
    updateOptimizationTargetsDisplay();
    
    // Clear inputs
    document.getElementById('targetValue').value = '';
}

function removeOptimizationTarget(index) {
    optimizationTargets.splice(index, 1);
    updateOptimizationTargetsDisplay();
}

function updateOptimizationTargetsDisplay() {
    const container = document.getElementById('optimizationTargets');
    
    if (optimizationTargets.length === 0) {
        container.innerHTML = '<p class="text-muted">No targets added yet</p>';
        return;
    }

    let html = '';
    optimizationTargets.forEach((target, index) => {
        html += `
            <div class="target-item">
                <div>
                    <span class="target-property">${formatProperty(target.property)}</span>:
                    <span class="target-value">${target.value}</span>
                </div>
                <span class="remove-target" onclick="removeOptimizationTarget(${index})">
                    <i class="fas fa-times"></i>
                </span>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function addOptimizationExample() {
    document.getElementById('startingSmiles').value = 'CCO';
    optimizationTargets = [
        { property: 'logp', value: 2.5 },
        { property: 'molecular_weight', value: 300 }
    ];
    updateOptimizationTargetsDisplay();
}

async function startOptimization() {
    const startingSmiles = document.getElementById('startingSmiles').value.trim();
    const maxIterations = parseInt(document.getElementById('maxIterations').value);

    if (!startingSmiles) {
        showAlert('Please enter a starting SMILES string', 'warning');
        return;
    }

    if (optimizationTargets.length === 0) {
        showAlert('Please add at least one optimization target', 'warning');
        return;
    }

    // Convert targets to API format
    const targets = optimizationTargets.map(target => ({
        property_name: target.property,
        target_value: target.value,
        weight: 1.0,
        tolerance: 0.1,
        direction: "target"
    }));

    showLoading();
    try {
        const response = await makeAPICall('/optimize_compound', 'POST', {
            starting_smiles: startingSmiles,
            targets: targets,
            max_iterations: maxIterations,
            population_size: 50,
            convergence_threshold: 0.01
        });

        displayOptimizationResults(response);
    } catch (error) {
        showAlert(`Optimization failed: ${error.message}`, 'danger');
    } finally {
        hideLoading();
    }
}

function displayOptimizationResults(response) {
    const resultsContainer = document.getElementById('optimizationResults');
    const statusContainer = document.getElementById('optimizationStatus');
    
    resultsContainer.style.display = 'block';
    
    let html = `
        <div class="fade-in">
            <div class="row">
                <div class="col-md-6">
                    <h6>Task ID</h6>
                    <p class="font-monospace">${response.task_id}</p>
                </div>
                <div class="col-md-6">
                    <h6>Status</h6>
                    <span class="status-badge status-${response.status}">
                        ${response.status.toUpperCase()}
                    </span>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <h6>Starting Molecule</h6>
                    <span class="smiles-text">${response.starting_smiles || 'N/A'}</span>
                </div>
            </div>
    `;

    if (response.estimated_completion_time) {
        const minutes = Math.ceil(response.estimated_completion_time / 60);
        html += `
            <div class="row mt-3">
                <div class="col-12">
                    <h6>Estimated Completion Time</h6>
                    <p>${minutes} minutes</p>
                </div>
            </div>
        `;
    }

    html += `
            <div class="mt-3">
                <small class="text-muted">
                    <i class="fas fa-info-circle me-1"></i>
                    Optimization is running in the background. 
                    Check status at: <span class="font-monospace">/optimization/status/${response.task_id}</span>
                </small>
            </div>
        </div>
    `;

    statusContainer.innerHTML = html;
}

// File Upload Functions
async function uploadFile() {
    const fileInput = document.getElementById('fileUpload');
    const smilesColumn = document.getElementById('smilesColumn').value;
    const targetColumn = document.getElementById('targetColumn').value;

    if (!fileInput.files.length) {
        showAlert('Please select a file to upload', 'warning');
        return;
    }

    const file = fileInput.files[0];
    const maxSize = 100 * 1024 * 1024; // 100MB

    if (file.size > maxSize) {
        showAlert('File size exceeds 100MB limit', 'danger');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Add parameters as query string since we're using FormData
    let url = `/upload/molecules?smiles_column=${encodeURIComponent(smilesColumn)}`;
    if (targetColumn) {
        url += `&target_column=${encodeURIComponent(targetColumn)}`;
    }

    showLoading();
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayUploadResults(result);
    } catch (error) {
        showAlert(`Upload failed: ${error.message}`, 'danger');
    } finally {
        hideLoading();
    }
}

function displayUploadResults(response) {
    const resultsContainer = document.getElementById('uploadResults');
    const statusContainer = document.getElementById('uploadStatus');
    
    resultsContainer.style.display = 'block';
    
    let html = `
        <div class="fade-in">
            <div class="row">
                <div class="col-md-6">
                    <h6>Upload ID</h6>
                    <p class="font-monospace">${response.upload_id}</p>
                </div>
                <div class="col-md-6">
                    <h6>Status</h6>
                    <span class="status-badge status-${response.status}">
                        ${response.status.toUpperCase()}
                    </span>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-md-6">
                    <h6>Filename</h6>
                    <p>${response.filename}</p>
                </div>
                <div class="col-md-6">
                    <h6>File Size</h6>
                    <p>${(response.file_size / 1024).toFixed(1)} KB</p>
                </div>
            </div>
            <div class="mt-3">
                <small class="text-muted">
                    <i class="fas fa-info-circle me-1"></i>
                    File is being processed. Check status at: 
                    <span class="font-monospace">${response.polling_url}</span>
                </small>
            </div>
        </div>
    `;

    statusContainer.innerHTML = html;
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Property Prediction Form
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            predictProperties();
        });
    }

    // Similarity Search Form
    const similarityForm = document.getElementById('similarityForm');
    if (similarityForm) {
        similarityForm.addEventListener('submit', function(e) {
            e.preventDefault();
            calculateSimilarity();
        });
    }

    // Optimization Form
    const optimizationForm = document.getElementById('optimizationForm');
    if (optimizationForm) {
        optimizationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            startOptimization();
        });
    }

    // File Upload Form
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            uploadFile();
        });
    }

    // Example inputs
    const examples = [
        { smiles: 'CCO', name: 'Ethanol' },
        { smiles: 'CC(=O)O', name: 'Acetic Acid' },
        { smiles: 'c1ccccc1', name: 'Benzene' },
        { smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', name: 'Ibuprofen' },
        { smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', name: 'Caffeine' }
    ];

    // Add example dropdown (you could implement this)
    
    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Initialize optimization targets display
    updateOptimizationTargetsDisplay();

    // Auto-expand textareas
    document.querySelectorAll('textarea').forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    });
});

// Export functions for global access
window.addExample = addExample;
window.addSimilarityExample = addSimilarityExample;
window.addOptimizationExample = addOptimizationExample;
window.addOptimizationTarget = addOptimizationTarget;
window.removeOptimizationTarget = removeOptimizationTarget;
window.predictProperties = predictProperties;
window.calculateSimilarity = calculateSimilarity;
window.startOptimization = startOptimization;
window.uploadFile = uploadFile;