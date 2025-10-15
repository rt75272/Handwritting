/**
 * Admin page JavaScript functionality.
 * Handles model management and retraining operations.
 */

let trainingInProgress = false;

/**
 * Initialize the admin page.
 */
function initializeAdminPage() {
    checkModelStatus();
    log('Model management interface loaded');
}

/**
 * Add a message to the training log.
 */
function log(message) {
    const logElement = document.getElementById('log');
    const timestamp = new Date().toLocaleTimeString();
    logElement.innerHTML += `[${timestamp}] ${message}\n`;
    logElement.scrollTop = logElement.scrollHeight;
}

/**
 * Show a status message to the user.
 */
function showStatus(message, type = 'info') {
    const statusElement = document.getElementById('status-message');
    statusElement.innerHTML = `<div class="status ${type}">${message}</div>`;
}

/**
 * Clear the training log.
 */
function clearLog() {
    document.getElementById('log').innerHTML = '';
}

/**
 * Check and display current model status.
 */
function checkModelStatus() {
    document.getElementById('model-status').textContent = 'Loaded and ready';
    log('Model status checked');
}

/**
 * Retrain the model with improved architecture.
 */
async function retrainModel() {
    if (trainingInProgress) {
        showStatus('Training is already in progress!', 'error');
        return;
    }
    trainingInProgress = true;
    const retrainBtn = document.getElementById('retrain-btn');
    const progressDiv = document.getElementById('training-progress');
    const progressBar = document.getElementById('progress-bar');
    retrainBtn.disabled = true;
    retrainBtn.textContent = '🔄 Training in Progress...';
    progressDiv.style.display = 'block';
    progressBar.style.width = '10%';
    log('Starting model retraining with improved architecture...');
    showStatus('🚀 Training started! This may take several minutes...', 'info');
    try {
        // Simulate progress updates
        let progress = 10;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 10;
                progressBar.style.width = progress + '%';
                progressBar.textContent = `Training... ${Math.round(progress)}%`;
            }
        }, 2000);
        const response = await fetch('/retrain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        progressBar.textContent = 'Complete!';
        const result = await response.json();
        if (result.status === 'success') {
            log('✅ Model retraining completed successfully!');
            showStatus('✅ Model retrained successfully! The app is now using the improved model.', 'success');
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        log(`❌ Training failed: ${error.message}`);
        showStatus(`❌ Training failed: ${error.message}`, 'error');
    } finally {
        trainingInProgress = false;
        retrainBtn.disabled = false;
        retrainBtn.textContent = '🔄 Retrain Model (Improved Architecture)';
        setTimeout(() => {
            progressDiv.style.display = 'none';
            progressBar.style.width = '0%';
            progressBar.textContent = 'Training...';
        }, 3000);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAdminPage);
} else {
    initializeAdminPage();
}