/**
 * MNIST Digit Recognition Canvas Drawing Interface.
 * 
 * This JavaScript module handles the HTML5 canvas drawing functionality 
 * for the digit recognition web application. It provides:
 * - Mouse-based drawing with proper event handling
 * - Canvas clearing and background management  
 * - Image capture and prediction API communication
 */

// Canvas and drawing context references.
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Drawing state management.
let isDrawing = false;

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
    const resultElement = document.getElementById('result');
    if (resultElement) {
        resultElement.innerText = "";
    }
    console.log("Canvas cleared and reset");
}

/**
 * Capture canvas image and send to server for digit prediction.
 * 
 * This function:
 * 1. Converts canvas to base64 data URL
 * 2. Sends image data to the /predict endpoint
 * 3. Displays the prediction result and confidence
 * 4. Handles errors gracefully
 */
async function predictDigit() {
    try {
        // Show loading state.
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.innerText = "Predicting...";
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
        if (result.error) {
            throw new Error(result.error);
        }
        // Display prediction result.
        if (resultElement) {
            resultElement.innerText = 
                `Predicted: ${result.prediction} (Confidence: ${result.confidence})`;
        }
        // Log additional prediction details if available.
        if (result.top_3) {
            console.log("Top 3 predictions:", result.top_3);
        }
    } catch (error) {
        console.error("Error during prediction:", error);
        // Display error message to user.
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.innerText = `Error: ${error.message}`;
        }
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
function initialize() {
    if (!canvas || !ctx) {
        console.error("Canvas or context not found!");
        return;
    }
    initializeCanvas();
    setupEventListeners();
    console.log("Drawing interface initialized successfully");
}

// Initialize when DOM is ready.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}