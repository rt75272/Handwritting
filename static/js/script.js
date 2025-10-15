/**
 * Enhanced Character Recognition Canvas Drawing Interface.
 * 
 * This JavaScript module handles the HTML5 canvas drawing functionality 
 * for the character recognition web application. It provides:
 * - Mouse-based drawing with proper event handling
 * - Canvas clearing and background management  
 * - Image capture and prediction API communication
 * - Mode switching between digit-only and alphanumeric recognition
 */

// Canvas and drawing context references.
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Drawing state management.
let isDrawing = false;
let currentMode = 'digits'; // Default mode

// Drawing configuration constants.
const DRAWING_CONFIG = {
    lineWidth: 20,           // Pen thickness for good visibility.
    lineCap: "round",        // Smooth line endings.
    strokeStyle: "black",    // Black ink on white background.
    backgroundColor: "white" // White canvas background.
};

/**
 * Initialize the canvas with a white background.
 * This is critical for proper image processing since MNIST expects
 * black digits on white background, but we'll invert colors later.
 */
function initializeCanvas() {
    ctx.fillStyle = DRAWING_CONFIG.backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    console.log("Canvas initialized with white background");
}

/**
 * Start drawing when mouse is pressed down.
 * 
 * @param {MouseEvent} event - The mouse down event.
 */
function startDrawing(event) {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(event.offsetX, event.offsetY);
    console.log(`Started drawing at (${event.offsetX}, ${event.offsetY})`);
}

/**
 * Stop drawing when mouse is released or leaves canvas.
 */
function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        console.log("Stopped drawing");
    }
}

/**
 * Handle drawing while mouse is moving.
 * 
 * @param {MouseEvent} event - The mouse move event.
 */
function draw(event) {
    if (!isDrawing) return;
    // Configure drawing style.
    ctx.lineWidth = DRAWING_CONFIG.lineWidth;
    ctx.lineCap = DRAWING_CONFIG.lineCap;
    ctx.strokeStyle = DRAWING_CONFIG.strokeStyle;
    // Draw line to current mouse position.
    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.stroke();
}

/**
 * Clear the canvas and reset to white background.
 * Also clears any previous prediction results.
 */
function clearCanvas() {
    // Clear the entire canvas.
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Restore white background.
    ctx.fillStyle = DRAWING_CONFIG.backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Reset drawing path.
    ctx.beginPath();
    // Clear previous prediction results.
    clearResults();
    console.log("Canvas cleared and reset");
}

/**
 * Switch between recognition modes (digits vs alphanumeric).
 * 
 * @param {string} mode - The mode to switch to ('digits' or 'alphanumeric')
 */
async function switchMode(mode) {
    try {
        console.log(`Switching to ${mode} mode...`);
        
        // Update UI immediately
        updateModeUI(mode);
        
        // Send mode change request to server
        const response = await fetch('/api/mode', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: mode })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentMode = mode;
            console.log(`Successfully switched to ${mode} mode`);
            
            // Update subtitle text
            const subtitle = document.getElementById('subtitle');
            if (subtitle) {
                if (mode === 'alphanumeric') {
                    subtitle.textContent = 'Draw any digit (0-9) or letter (A-Z) and let AI predict what you wrote!';
                } else {
                    subtitle.textContent = 'Draw any digit (0-9) and let AI predict what you wrote!';
                }
            }
            
            // Clear any previous results
            clearResults();
        } else {
            console.error('Failed to switch mode:', result.message);
            // Revert UI if mode switch failed
            updateModeUI(currentMode);
        }
        
    } catch (error) {
        console.error('Error switching mode:', error);
        // Revert UI if error occurred
        updateModeUI(currentMode);
    }
}

/**
 * Update the mode selector UI.
 * 
 * @param {string} activeMode - The currently active mode
 */
function updateModeUI(activeMode) {
    const digitsBtn = document.getElementById('digits-mode');
    const alphanumericBtn = document.getElementById('alphanumeric-mode');
    
    // Remove active class from both buttons
    digitsBtn.classList.remove('active');
    alphanumericBtn.classList.remove('active');
    
    // Add active class to the selected mode
    if (activeMode === 'digits') {
        digitsBtn.classList.add('active');
    } else {
        alphanumericBtn.classList.add('active');
    }
}

/**
 * Clear prediction results and top predictions display.
 */
function clearResults() {
    const resultElement = document.getElementById('result');
    const topPredictions = document.getElementById('top-predictions');
    
    if (resultElement) {
        resultElement.innerHTML = "";
        resultElement.className = "";
    }
    
    if (topPredictions) {
        topPredictions.style.display = 'none';
    }
}

/**
 * Display top predictions in a formatted list.
 * 
 * @param {Array} predictions - Array of prediction objects
 */
function displayTopPredictions(predictions) {
    const topPredictions = document.getElementById('top-predictions');
    const predictionsList = document.getElementById('predictions-list');
    
    if (!topPredictions || !predictionsList || predictions.length < 2) {
        return;
    }
    
    // Clear previous predictions
    predictionsList.innerHTML = '';
    
    // Add each prediction
    predictions.forEach((pred, index) => {
        const predItem = document.createElement('div');
        predItem.className = 'prediction-item';
        
        predItem.innerHTML = `
            <span class="prediction-character">${pred.character}</span>
            <span class="prediction-confidence">${pred.confidence_percent}</span>
        `;
        
        predictionsList.appendChild(predItem);
    });
    
    // Show the top predictions section
    topPredictions.style.display = 'block';
}

/**
 * Capture canvas image and send to server for character prediction.
 * 
 * This function:
 * 1. Converts canvas to base64 data URL
 * 2. Sends image data to the /predict endpoint
 * 3. Displays the prediction result and confidence with modern styling
 * 4. Shows top 3 predictions
 * 5. Handles errors gracefully with visual feedback
 */
async function predictCharacter() {
    try {
        // Show loading state with modern styling.
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.className = 'loading';
            resultElement.innerHTML = '<span class="loading-spinner"></span>Analyzing your drawing...';
        }
        
        
        // Convert canvas to base64 data URL.
        const dataURL = canvas.toDataURL('image/png');
        console.log(`Captured canvas image, data URL length: ${dataURL.length}`);
        
        // Send prediction request to server.
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ image: dataURL })
        });
        
        // Check if request was successful.
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Parse JSON response.
        const result = await response.json();
        console.log("Prediction result:", result);
        
        // Check for server-side errors.
        if (result.error || !result.success) {
            throw new Error(result.error || 'Prediction failed');
        }
        
        // Display prediction result with success styling.
        if (resultElement) {
            resultElement.className = 'success';
            const modeText = result.mode === 'alphanumeric' ? '🔤' : '🔢';
            resultElement.innerHTML = `
                ${modeText} <strong>Predicted: ${result.prediction}</strong><br>
                📊 Confidence: ${result.confidence}
            `;
        }
        
        // Display top 3 predictions if available
        if (result.top_3 && result.top_3.length > 1) {
            displayTopPredictions(result.top_3);
            console.log("Top 3 predictions:", result.top_3);
        }
        
    } catch (error) {
        console.error("Error during prediction:", error);
        
        // Display error message to user with error styling.
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.className = 'error';
            resultElement.innerHTML = `
                ❌ <strong>Prediction Failed</strong><br>
                ${error.message}
            `;
        }
    }
}

/**
 * Load available modes from server and initialize UI.
 */
async function loadAvailableModes() {
    try {
        const response = await fetch('/api/mode');
        const data = await response.json();
        
        currentMode = data.current_mode;
        
        // Update UI based on available modes
        const alphanumericBtn = document.getElementById('alphanumeric-mode');
        if (!data.available_modes.alphanumeric) {
            alphanumericBtn.classList.add('disabled');
            alphanumericBtn.onclick = () => {
                alert('Alphanumeric model not available. Please train the model first.');
            };
        }
        
        // Set initial mode UI
        updateModeUI(currentMode);
        
        // Update subtitle based on current mode
        const subtitle = document.getElementById('subtitle');
        if (subtitle) {
            if (currentMode === 'alphanumeric') {
                subtitle.textContent = 'Draw any digit (0-9) or letter (A-Z) and let AI predict what you wrote!';
            } else {
                subtitle.textContent = 'Draw any digit (0-9) and let AI predict what you wrote!';
            }
        }
        
        console.log(`Loaded modes. Current: ${currentMode}, Available:`, data.available_modes);
        
    } catch (error) {
        console.error('Error loading available modes:', error);
    }
}

/**
 * Set up event listeners for canvas drawing interactions.
 * Supports both mouse and touch events for mobile compatibility.
 */
function setupEventListeners() {
    // Mouse events for desktop.
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);  // Stop drawing if mouse leaves canvas.
    canvas.addEventListener('mousemove', draw);
    // Touch events for mobile devices.
    canvas.addEventListener('touchstart', (event) => {
        event.preventDefault();  // Prevent scrolling.
        const touch = event.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });
    canvas.addEventListener('touchend', (event) => {
        event.preventDefault();
        stopDrawing();
    });
    canvas.addEventListener('touchmove', (event) => {
        event.preventDefault();
        const touch = event.touches[0];
        const rect = canvas.getBoundingClientRect();
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        // Calculate offset coordinates.
        mouseEvent.offsetX = touch.clientX - rect.left;
        mouseEvent.offsetY = touch.clientY - rect.top;
        canvas.dispatchEvent(mouseEvent);
    });
    console.log("Event listeners configured for canvas");
}

/**
 * Initialize the drawing interface when the page loads.
 */
async function initialize() {
    if (!canvas || !ctx) {
        console.error("Canvas or context not found!");
        return;
    }
    
    initializeCanvas();
    setupEventListeners();
    
    // Load available modes from server
    await loadAvailableModes();
    
    console.log("Enhanced character recognition interface initialized successfully");
}

// Initialize when DOM is ready.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}