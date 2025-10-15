/**
 * Debug page JavaScript functionality.
 * Provides canvas drawing test functionality and debug information.
 */

let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let drawing = false;

/**
 * Initialize the debug page.
 */
function initializeDebugPage() {
    if (!canvas || !ctx) {
        console.error("Canvas or context not found!");
        return;
    }
    // Set canvas background to white
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setupDebugEventListeners();
    updateDebug('Debug page loaded');
}

/**
 * Set up event listeners for debug canvas.
 */
function setupDebugEventListeners() {
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', () => drawing = false);
    canvas.addEventListener('mouseout', () => drawing = false);
    canvas.addEventListener('mousemove', draw);
}

/**
 * Start drawing on the debug canvas.
 */
function startDrawing(e) {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    updateDebug('Started drawing at: ' + e.offsetX + ', ' + e.offsetY);
}

/**
 * Draw on the canvas.
 */
function draw(e) {
    if (!drawing) return;
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
}

/**
 * Draw a test pattern on the canvas.
 */
function testDraw() {
    // Draw a test pattern
    ctx.beginPath();
    ctx.lineWidth = 20;
    ctx.strokeStyle = "black";
    ctx.moveTo(140, 70);
    ctx.lineTo(140, 210);
    ctx.stroke();
    updateDebug('Drew test line');
}

/**
 * Clear the debug canvas.
 */
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updateDebug('Canvas cleared');
    document.getElementById('result').innerText = "";
}

/**
 * Check canvas data and display debug information.
 */
function checkCanvas() {
    let dataURL = canvas.toDataURL();
    updateDebug('Canvas data URL length: ' + dataURL.length + '\nFirst 100 chars: ' + dataURL.substring(0, 100));
}

/**
 * Send prediction request to server.
 */
async function predictDigit() {
    let dataURL = canvas.toDataURL();
    updateDebug('Sending prediction request...\nData URL length: ' + dataURL.length);
    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({image: dataURL})
        });
        const result = await response.json();
        document.getElementById('result').innerText = `Predicted: ${result.prediction} (Confidence: ${result.confidence})`;
        updateDebug('Prediction successful: ' + result.prediction + ' (' + result.confidence + ')');
    } catch (error) {
        updateDebug('Error: ' + error.message);
    }
}

/**
 * Update the debug information display.
 */
function updateDebug(message) {
    const debugElement = document.getElementById('debug-info');
    const timestamp = new Date().toLocaleTimeString();
    debugElement.textContent = timestamp + ': ' + message + '\n' + debugElement.textContent;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDebugPage);
} else {
    initializeDebugPage();
}