const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d',{ willReadFrequently: true });
ctx.set


// Internal canvas size
const pixelWidth = 28;
const pixelHeight = 28;

// Variables for drawing
let isDrawing = false;
let lineWidth = 1;
let sigma = 1;
let drawColor = "White"

let gaussian = false;
function Kernel(){
    let kernel = [];
    let mean = (lineWidth-1)/2;
    let sum = 0.0;
    for(let x=0;x< lineWidth;x++){
            kernel[x] = Math.exp(-0.5* Math.pow((x-mean)/sigma,2));
            sum += kernel[x];
    }
    for(let x=0; x<lineWidth; x++){
            kernel[x] /= sum;
        }
    return kernel;
}
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
    lineWidth = Number(e.target.value);
    document.getElementById("stroke-value").textContent = lineWidth;
});
document.getElementById("sigma").addEventListener("input", (e) => {
    sigma = Number(e.target.value);
    document.getElementById("sigmaValue").textContent = sigma;
    if(sigma === 0){
        document.getElementById("gaussian").style.backgroundColor = "#000000"
    }else{
        document.getElementById("gaussian").style.backgroundColor = "#ffffff"
    }
});

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

canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    const { x, y } = getMousePos(e);
    drawPixel(x, y);
});

canvas.addEventListener("mouseup", () => {
    isDrawing = false;
});

canvas.addEventListener("mousemove", (e) => {
    if (!isDrawing) return;
    const { x, y } = getMousePos(e);
    drawPixel(x, y);
});

function drawPixel(x, y) {
    ctx.fillStyle = drawColor;

    // Scale the "line width" based on visual size (this is optional but gives better control)
    ctx.lineWidth = lineWidth;

    // Draw a filled square representing the "pixel"
    if(!gaussian){
        ctx.fillRect(x-Math.floor(lineWidth/2), y-Math.floor(lineWidth/2), lineWidth, lineWidth);
    }else{
        let K = Kernel();
        for(let i=x-Math.floor(lineWidth/2); i<x+lineWidth-Math.floor(lineWidth/2) ; i++){
            for(let j=y-Math.floor(lineWidth/2); j<y+lineWidth-Math.floor(lineWidth/2) ; j++){
                var color = ctx.getImageData(i,j,1,1).data[0]; //it'll get red value, and it's enough since the image is black-white gradiant
                var colorValue = Math.min(255,Math.floor(color + 255*K[i-x+Math.floor(lineWidth/2)]*K[j-y+Math.floor(lineWidth/2)]));
                ctx.fillStyle = `rgb(${colorValue},${colorValue},${colorValue})`;
                ctx.fillRect(i,j,1,1);
            }
        }
    }
}
document.getElementById("clear").addEventListener("click",()=>{
    ctx.fillStyle = "Black"
    ctx.fillRect(0,0,28,28);

});

document.getElementById("gaussian").addEventListener("click",()=>{
    gaussian = !gaussian;
    if(gaussian){
        document.getElementById("sigma").disabled = false;
        if(sigma !== 0){
            document.getElementById("gaussian").style.backgroundColor = "#ffffff"
        }
    }else{
        document.getElementById("gaussian").style.backgroundColor = "#000000"
        document.getElementById("sigma").disabled = true;
    }
});
// Initial resize for the canvas
resizeCanvas();

// Resize canvas when the window size changes
window.addEventListener("resize", resizeCanvas);
