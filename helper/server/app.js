const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Internal canvas size
const pixelWidth = 28;
const pixelHeight = 28;

// Variables for drawing
let isDrawing = false;
let lineWidth = 1;
let drawColor = "White";
let pixelSize = 20; // Size of each "pixel" on screen

// Resize canvas to make each "pixel" visually large
function resizeCanvas() {

    const aspectRatio = window.innerWidth/window.innerHeight;
    let direction = "row";
    if(aspectRatio < 1){
        direction = "column"
    }

    document.documentElement.style.setProperty('--flex_direction',direction);
}

// Update the stroke width dynamically when the slider is used
document.getElementById("stroke-width").addEventListener("input", (e) => {
    lineWidth = e.target.value;
    document.getElementById("stroke-value").textContent = lineWidth;
});

// Helper function to get mouse coordinates on the canvas
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Get mouse position relative to the canvas
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    // Snap the mouse position to the nearest "pixel" on the 28x28 grid
    return {
        x: Math.floor(mouseX),
        y: Math.floor(mouseY),
    };
}

// Start drawing when mouse is pressed
canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    const { x, y } = getMousePos(e);
    drawPixel(x, y);
});

// Stop drawing when mouse is released
canvas.addEventListener("mouseup", () => {
    isDrawing = false;
});

// Draw on the canvas when mouse moves
canvas.addEventListener("mousemove", (e) => {
    if (!isDrawing) return;
    const { x, y } = getMousePos(e);
    drawPixel(x, y);
});

// Function to draw a pixel
function drawPixel(x, y) {
    ctx.fillStyle = drawColor;

    // Scale the "line width" based on visual size (this is optional but gives better control)
    ctx.lineWidth = lineWidth;

    // Draw a filled square representing the "pixel"
    ctx.fillRect(x, y, 1, 1);
}

// Initial resize for the canvas
resizeCanvas();

// Resize canvas when the window size changes
window.addEventListener("resize", resizeCanvas);
