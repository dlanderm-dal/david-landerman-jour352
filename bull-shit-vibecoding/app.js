// ============================================
// Color Conversion Functions
// ============================================

function RGB2LAB(Q) {
    var var_R = Q[0] / 255;
    var var_G = Q[1] / 255;
    var var_B = Q[2] / 255;
    
    if (var_R > 0.04045) var_R = Math.pow((var_R + 0.055) / 1.055, 2.4);
    else var_R = var_R / 12.92;
    if (var_G > 0.04045) var_G = Math.pow((var_G + 0.055) / 1.055, 2.4);
    else var_G = var_G / 12.92;
    if (var_B > 0.04045) var_B = Math.pow((var_B + 0.055) / 1.055, 2.4);
    else var_B = var_B / 12.92;
    
    var_R *= 100;
    var_G *= 100;
    var_B *= 100;
    
    var X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
    var Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
    var Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;
    
    var var_X = X / 95.047;
    var var_Y = Y / 100;
    var var_Z = Z / 108.883;
    
    if (var_X > 0.008856) var_X = Math.pow(var_X, 1/3);
    else var_X = (7.787 * var_X) + (16 / 116);
    if (var_Y > 0.008856) var_Y = Math.pow(var_Y, 1/3);
    else var_Y = (7.787 * var_Y) + (16 / 116);
    if (var_Z > 0.008856) var_Z = Math.pow(var_Z, 1/3);
    else var_Z = (7.787 * var_Z) + (16 / 116);
    
    var L = (116 * var_Y) - 16;
    var A = 500 * (var_X - var_Y);
    var B = 200 * (var_Y - var_Z);
    
    return [L, A, B];
}

function LAB2RGB(Q) {
    var var_Y = (Q[0] + 16) / 116;
    var var_X = Q[1] / 500 + var_Y;
    var var_Z = var_Y - Q[2] / 200;
    
    if (var_Y > 0.206893034422) var_Y = Math.pow(var_Y, 3);
    else var_Y = (var_Y - 16 / 116) / 7.787;
    if (var_X > 0.206893034422) var_X = Math.pow(var_X, 3);
    else var_X = (var_X - 16 / 116) / 7.787;
    if (var_Z > 0.206893034422) var_Z = Math.pow(var_Z, 3);
    else var_Z = (var_Z - 16 / 116) / 7.787;
    
    var X = 95.047 * var_X;
    var Y = 100 * var_Y;
    var Z = 108.883 * var_Z;
    
    var_X = X / 100;
    var_Y = Y / 100;
    var_Z = Z / 100;
    
    var var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
    var var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415;
    var var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570;
    
    if (var_R > 0.0031308) var_R = 1.055 * Math.pow(var_R, 1/2.4) - 0.055;
    else var_R = 12.92 * var_R;
    if (var_G > 0.0031308) var_G = 1.055 * Math.pow(var_G, 1/2.4) - 0.055;
    else var_G = 12.92 * var_G;
    if (var_B > 0.0031308) var_B = 1.055 * Math.pow(var_B, 1/2.4) - 0.055;
    else var_B = 12.92 * var_B;
    
    return [var_R * 255, var_G * 255, var_B * 255];
}

function hexToRgb(hex) {
    hex = hex.replace('#', '');
    if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    return [
        parseInt(hex.substring(0, 2), 16),
        parseInt(hex.substring(2, 4), 16),
        parseInt(hex.substring(4, 6), 16)
    ];
}

function rgbToHex(r, g, b) {
    r = Math.max(0, Math.min(255, Math.round(r)));
    g = Math.max(0, Math.min(255, Math.round(g)));
    b = Math.max(0, Math.min(255, Math.round(b)));
    return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('').toUpperCase();
}

// ============================================
// Matrix Operations
// ============================================

function pinv(A) {
    var z = numeric.svd(A), foo = z.S[0];
    var U = z.U, S = z.S, V = z.V;
    var m = A.length, n = A[0].length, tol = Math.max(m, n) * numeric.epsilon * foo, M = S.length;
    var Sinv = new Array(M);
    for (var i = M - 1; i >= 0; i--) {
        Sinv[i] = S[i] > tol ? 1 / S[i] : 0;
    }
    return numeric.dot(numeric.dot(V, numeric.diag(Sinv)), numeric.transpose(U));
}

// ============================================
// WebGL GPU Acceleration
// ============================================

let gl = null;
let webglCanvas = null;
let simpleRecolorProgram = null;
let rbfRecolorProgram = null;
let webglInitialized = false;

// Vertex shader (shared by both programs) - simple fullscreen quad
const vertexShaderSource = `
    attribute vec2 a_position;
    attribute vec2 a_texCoord;
    varying vec2 v_texCoord;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
    }
`;

// Fragment shader for simple weighted nearest-neighbor recoloring
const simpleRecolorFragmentSource = `
    precision highp float;

    varying vec2 v_texCoord;
    uniform sampler2D u_image;
    uniform vec3 u_oldLab[20];
    uniform vec3 u_diffLab[20];
    uniform int u_paletteSize;
    uniform float u_blendSharpness;
    uniform float u_luminosity;

    // RGB to LAB conversion
    vec3 rgb2lab(vec3 rgb) {
        // RGB to XYZ
        vec3 c = rgb;
        vec3 tmp;
        tmp.x = (c.x > 0.04045) ? pow((c.x + 0.055) / 1.055, 2.4) : c.x / 12.92;
        tmp.y = (c.y > 0.04045) ? pow((c.y + 0.055) / 1.055, 2.4) : c.y / 12.92;
        tmp.z = (c.z > 0.04045) ? pow((c.z + 0.055) / 1.055, 2.4) : c.z / 12.92;
        tmp *= 100.0;

        float X = tmp.x * 0.4124 + tmp.y * 0.3576 + tmp.z * 0.1805;
        float Y = tmp.x * 0.2126 + tmp.y * 0.7152 + tmp.z * 0.0722;
        float Z = tmp.x * 0.0193 + tmp.y * 0.1192 + tmp.z * 0.9505;

        // XYZ to LAB
        vec3 xyz = vec3(X / 95.047, Y / 100.0, Z / 108.883);
        xyz.x = (xyz.x > 0.008856) ? pow(xyz.x, 1.0/3.0) : (7.787 * xyz.x) + (16.0 / 116.0);
        xyz.y = (xyz.y > 0.008856) ? pow(xyz.y, 1.0/3.0) : (7.787 * xyz.y) + (16.0 / 116.0);
        xyz.z = (xyz.z > 0.008856) ? pow(xyz.z, 1.0/3.0) : (7.787 * xyz.z) + (16.0 / 116.0);

        return vec3(
            (116.0 * xyz.y) - 16.0,
            500.0 * (xyz.x - xyz.y),
            200.0 * (xyz.y - xyz.z)
        );
    }

    // LAB to RGB conversion
    vec3 lab2rgb(vec3 lab) {
        float var_Y = (lab.x + 16.0) / 116.0;
        float var_X = lab.y / 500.0 + var_Y;
        float var_Z = var_Y - lab.z / 200.0;

        float threshold = 0.206893034422;
        var_Y = (var_Y > threshold) ? pow(var_Y, 3.0) : (var_Y - 16.0 / 116.0) / 7.787;
        var_X = (var_X > threshold) ? pow(var_X, 3.0) : (var_X - 16.0 / 116.0) / 7.787;
        var_Z = (var_Z > threshold) ? pow(var_Z, 3.0) : (var_Z - 16.0 / 116.0) / 7.787;

        float X = 95.047 * var_X / 100.0;
        float Y = 100.0 * var_Y / 100.0;
        float Z = 108.883 * var_Z / 100.0;

        vec3 rgb;
        rgb.x = X * 3.2406 + Y * -1.5372 + Z * -0.4986;
        rgb.y = X * -0.9689 + Y * 1.8758 + Z * 0.0415;
        rgb.z = X * 0.0557 + Y * -0.2040 + Z * 1.0570;

        rgb.x = (rgb.x > 0.0031308) ? 1.055 * pow(rgb.x, 1.0/2.4) - 0.055 : 12.92 * rgb.x;
        rgb.y = (rgb.y > 0.0031308) ? 1.055 * pow(rgb.y, 1.0/2.4) - 0.055 : 12.92 * rgb.y;
        rgb.z = (rgb.z > 0.0031308) ? 1.055 * pow(rgb.z, 1.0/2.4) - 0.055 : 12.92 * rgb.z;

        return clamp(rgb, 0.0, 1.0);
    }

    void main() {
        vec4 texColor = texture2D(u_image, v_texCoord);
        vec3 pixelLab = rgb2lab(texColor.rgb);

        // Find distances to all palette colors and minimum distance
        float distances[20];
        float minDist = 1000000.0;

        for (int j = 0; j < 20; j++) {
            if (j >= u_paletteSize) break;
            vec3 diff = pixelLab - u_oldLab[j];
            float d = sqrt(dot(diff, diff));
            distances[j] = d;
            minDist = min(minDist, d);
        }

        // Calculate soft weights
        float totalWeight = 0.0;
        float weights[20];

        for (int j = 0; j < 20; j++) {
            if (j >= u_paletteSize) break;
            float relDist = distances[j] / max(minDist, 1.0);
            float w = exp(-u_blendSharpness * (relDist - 1.0));
            weights[j] = w;
            totalWeight += w;
        }

        // Apply weighted color shift
        vec3 dLab = vec3(0.0);
        if (totalWeight > 0.0) {
            for (int j = 0; j < 20; j++) {
                if (j >= u_paletteSize) break;
                float normalizedWeight = weights[j] / totalWeight;
                dLab += normalizedWeight * u_diffLab[j];
            }
        }

        vec3 newLab = pixelLab + dLab;
        vec3 newRgb = lab2rgb(newLab);

        // Apply luminosity
        if (u_luminosity != 0.0) {
            float factor = 1.0 + (u_luminosity / 100.0);
            newRgb = clamp(newRgb * factor, 0.0, 1.0);
        }

        gl_FragColor = vec4(newRgb, 1.0);
    }
`;

// Fragment shader for RBF recoloring with 3D LUT
const rbfRecolorFragmentSource = `
    precision highp float;

    varying vec2 v_texCoord;
    uniform sampler2D u_image;
    uniform sampler2D u_lut;
    uniform float u_lutSize;
    uniform float u_luminosity;

    void main() {
        vec4 texColor = texture2D(u_image, v_texCoord);

        // Trilinear interpolation in 3D LUT
        float ngrid = u_lutSize - 1.0;
        vec3 scaled = texColor.rgb * ngrid;
        vec3 base = floor(scaled);
        vec3 frac = scaled - base;

        // Clamp to valid range
        base = clamp(base, 0.0, ngrid - 1.0);

        // LUT is stored as a 2D texture: each row is a slice of the 3D LUT
        // Layout: lutSize^2 width x lutSize height
        // For position (r,g,b): x = r + g*lutSize, y = b
        float lutWidth = u_lutSize * u_lutSize;

        // Sample 8 corners of the cube
        vec3 c000 = texture2D(u_lut, vec2((base.x + base.y * u_lutSize + 0.5) / lutWidth, (base.z + 0.5) / u_lutSize)).rgb;
        vec3 c001 = texture2D(u_lut, vec2((base.x + base.y * u_lutSize + 0.5) / lutWidth, (base.z + 1.0 + 0.5) / u_lutSize)).rgb;
        vec3 c010 = texture2D(u_lut, vec2((base.x + (base.y + 1.0) * u_lutSize + 0.5) / lutWidth, (base.z + 0.5) / u_lutSize)).rgb;
        vec3 c011 = texture2D(u_lut, vec2((base.x + (base.y + 1.0) * u_lutSize + 0.5) / lutWidth, (base.z + 1.0 + 0.5) / u_lutSize)).rgb;
        vec3 c100 = texture2D(u_lut, vec2((base.x + 1.0 + base.y * u_lutSize + 0.5) / lutWidth, (base.z + 0.5) / u_lutSize)).rgb;
        vec3 c101 = texture2D(u_lut, vec2((base.x + 1.0 + base.y * u_lutSize + 0.5) / lutWidth, (base.z + 1.0 + 0.5) / u_lutSize)).rgb;
        vec3 c110 = texture2D(u_lut, vec2((base.x + 1.0 + (base.y + 1.0) * u_lutSize + 0.5) / lutWidth, (base.z + 0.5) / u_lutSize)).rgb;
        vec3 c111 = texture2D(u_lut, vec2((base.x + 1.0 + (base.y + 1.0) * u_lutSize + 0.5) / lutWidth, (base.z + 1.0 + 0.5) / u_lutSize)).rgb;

        // Trilinear interpolation
        vec3 c00 = mix(c000, c001, frac.z);
        vec3 c01 = mix(c010, c011, frac.z);
        vec3 c10 = mix(c100, c101, frac.z);
        vec3 c11 = mix(c110, c111, frac.z);
        vec3 c0 = mix(c00, c01, frac.y);
        vec3 c1 = mix(c10, c11, frac.y);
        vec3 newRgb = mix(c0, c1, frac.x);

        // Apply luminosity
        if (u_luminosity != 0.0) {
            float factor = 1.0 + (u_luminosity / 100.0);
            newRgb = clamp(newRgb * factor, 0.0, 1.0);
        }

        gl_FragColor = vec4(newRgb, 1.0);
    }
`;

function initWebGL() {
    if (webglInitialized) return webglInitialized;

    try {
        webglCanvas = document.createElement('canvas');
        gl = webglCanvas.getContext('webgl', { preserveDrawingBuffer: true }) ||
             webglCanvas.getContext('experimental-webgl', { preserveDrawingBuffer: true });

        if (!gl) {
            console.warn('WebGL not available, falling back to CPU');
            return false;
        }

        // Compile shaders and create programs
        simpleRecolorProgram = createWebGLProgram(gl, vertexShaderSource, simpleRecolorFragmentSource);
        rbfRecolorProgram = createWebGLProgram(gl, vertexShaderSource, rbfRecolorFragmentSource);

        if (!simpleRecolorProgram || !rbfRecolorProgram) {
            console.warn('Failed to compile WebGL shaders, falling back to CPU');
            return false;
        }

        webglInitialized = true;
        console.log('WebGL initialized successfully');
        return true;
    } catch (e) {
        console.warn('WebGL initialization failed:', e);
        return false;
    }
}

function createWebGLProgram(gl, vertexSource, fragmentSource) {
    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

    if (!vertexShader || !fragmentShader) return null;

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(program));
        return null;
    }

    return program;
}

function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

function setupWebGLBuffers(gl, program) {
    // Create fullscreen quad
    const positions = new Float32Array([
        -1, -1,  1, -1,  -1, 1,
        -1,  1,  1, -1,   1, 1
    ]);
    const texCoords = new Float32Array([
        0, 1,  1, 1,  0, 0,
        0, 0,  1, 1,  1, 0
    ]);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

    const texCoordLoc = gl.getAttribLocation(program, 'a_texCoord');
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);
}

function createTexture(gl, imageData) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, imageData.width, imageData.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, imageData.data);
    return texture;
}

// ============================================
// Global State
// ============================================

let imageData = null;
let originalImageData = null;
let originalPalette = [];
let targetPalette = [];
let selectedSlotIndex = 0;
let canvas, ctx;
let displayCanvas, displayCtx; // For high-quality scaled display
let useHighQualityDisplay = true; // Toggle for high-quality rendering
let pickerMode = false;
let pickedColors = [];
let pickedPositions = [];
let pickerTargetCategory = 0; // Which target category (0 = Background, -1 = Locked, 1+ = Accent N)
let pickedCategories = []; // Track which category each picked color belongs to

// Category constants
const CATEGORY_LOCKED = -1;
const CATEGORY_BACKGROUND = 0;

// Helper to get target category label
function getTargetCategoryLabel(index, short = false) {
    if (index === CATEGORY_LOCKED) return short ? 'L' : 'Locked';
    if (index === CATEGORY_BACKGROUND) return short ? 'B' : 'Background';
    return short ? `A${index}` : `Accent ${index}`;
}

// Cycle to next category: B → L → A1 → A2 → ... → B
function getNextCategory(currentCategory) {
    if (currentCategory === CATEGORY_BACKGROUND) return CATEGORY_LOCKED; // B → L
    if (currentCategory === CATEGORY_LOCKED) return 1; // L → A1
    // A1 → A2 → ... → B (wrap around based on targetCount)
    const nextAccent = currentCategory + 1;
    if (nextAccent >= targetCount) return CATEGORY_BACKGROUND; // wrap to B
    return nextAccent;
}
let colorPercentages = [];
let fullColorDistribution = [];

// Column-based mapping: originToColumn[originIndex] = columnIndex (or 'locked' or 'bank')
let originToColumn = [];

// Per-column bypass (lock) state - when true, origins in that column keep their original colors
let columnBypass = [];

// Live preview toggle - when false, autoRecolorImage does nothing
let livePreviewEnabled = false;

// Picked color markers stay visible
let shouldKeepPickedMarkers = false;

// Selected theme tracking
let selectedTheme = '';

// Algorithm selection: 'simple' or 'rbf'
let selectedAlgorithm = 'simple';

// Palette counts
let originCount = 5;
let targetCount = 5;

// Color tolerance for extraction merging (0 = no merging, higher = more merging)
let colorTolerance = 0;
const DEFAULT_COLOR_TOLERANCE = 0;

// Zoom and pan state
let zoomLevel = 1;
let panX = 0, panY = 0;
let isPanning = false;
let panStartX = 0, panStartY = 0;
let draggingMarker = null;
let altHeld = false;
let draggedOriginIndex = null;

// Theme Presets
const THEME_NAMES = {
    'catppuccin-latte': 'Catppuccin Latte',
    'catppuccin-frappe': 'Catppuccin Frappé',
    'catppuccin-macchiato': 'Catppuccin Macchiato',
    'catppuccin-mocha': 'Catppuccin Mocha',
    'nord-frost': 'Nord Frost',
    'nord-aurora': 'Nord Aurora',
    'nord-snow': 'Nord Snow Storm',
    'gruvbox-dark': 'Gruvbox Dark',
    'gruvbox-light': 'Gruvbox Light',
    'dracula': 'Dracula',
    'solarized-dark': 'Solarized Dark',
    'solarized-light': 'Solarized Light',
    'tokyonight-night': 'Tokyo Night',
    'tokyonight-storm': 'Tokyo Night Storm',
    'onedark': 'One Dark',
    'monokai': 'Monokai',
    'synthwave84': 'Synthwave 84',
    'palenight': 'Palenight',
    'ayu-dark': 'Ayu Dark',
    'ayu-light': 'Ayu Light',
    'material-ocean': 'Material Ocean',
    'everforest': 'Everforest'
};

const THEMES = {
    'catppuccin-latte': ['#dc8a78', '#dd7878', '#ea76cb', '#8839ef', '#d20f39'],
    'catppuccin-frappe': ['#f2d5cf', '#eebebe', '#f4b8e4', '#ca9ee6', '#e78284'],
    'catppuccin-macchiato': ['#f4dbd6', '#f0c6c6', '#f5bde6', '#c6a0f6', '#ed8796'],
    'catppuccin-mocha': ['#f5e0dc', '#f2cdcd', '#f5c2e7', '#cba6f7', '#f38ba8'],
    'nord-frost': ['#8fbcbb', '#88c0d0', '#81a1c1', '#5e81ac', '#4c566a'],
    'nord-aurora': ['#bf616a', '#d08770', '#ebcb8b', '#a3be8c', '#b48ead'],
    'nord-snow': ['#eceff4', '#e5e9f0', '#d8dee9', '#4c566a', '#2e3440'],
    'gruvbox-dark': ['#fb4934', '#fabd2f', '#b8bb26', '#83a598', '#d3869b'],
    'gruvbox-light': ['#9d0006', '#b57614', '#79740e', '#427b58', '#8f3f71'],
    'dracula': ['#ff79c6', '#bd93f9', '#8be9fd', '#50fa7b', '#f1fa8c'],
    'solarized-dark': ['#b58900', '#cb4b16', '#dc322f', '#d33682', '#6c71c4'],
    'solarized-light': ['#002b36', '#073642', '#586e75', '#657b83', '#839496'],
    'tokyonight-night': ['#f7768e', '#ff9e64', '#e0af68', '#9ece6a', '#7aa2f7'],
    'tokyonight-storm': ['#f7768e', '#ff9e64', '#e0af68', '#9ece6a', '#7aa2f7'],
    'onedark': ['#e06c75', '#d19a66', '#e5c07b', '#98c379', '#61afef'],
    'monokai': ['#f92672', '#fd971f', '#e6db74', '#a6e22e', '#66d9ef'],
    'synthwave84': ['#f97e72', '#ff7edb', '#36f9f6', '#72f1b8', '#fede5d'],
    'palenight': ['#f07178', '#f78c6c', '#ffcb6b', '#c3e88d', '#82aaff'],
    'ayu-dark': ['#f07178', '#ffb454', '#ffd580', '#c2d94c', '#59c2ff'],
    'ayu-light': ['#f07178', '#fa8d3e', '#f2ae49', '#86b300', '#399ee6'],
    'material-ocean': ['#f07178', '#f78c6c', '#ffcb6b', '#c3e88d', '#82aaff'],
    'everforest': ['#e67e80', '#e69875', '#dbbc7f', '#a7c080', '#7fbbb3']
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Create display canvas for high-quality scaled rendering
    displayCanvas = document.createElement('canvas');
    displayCanvas.id = 'displayCanvas';
    displayCtx = displayCanvas.getContext('2d', { alpha: false });
    displayCtx.imageSmoothingEnabled = true;
    displayCtx.imageSmoothingQuality = 'high';

    // Initialize color gradient picker
    initColorGradientPicker();

    // Initialize resize handles
    initResizeHandles();

    // Initialize theme browser
    const themeContainer = document.getElementById('themeScrollContainer');
    Object.keys(THEMES).forEach(themeKey => {
        const item = document.createElement('div');
        item.className = 'theme-item';
        item.dataset.theme = themeKey;
        item.onclick = () => selectTheme(themeKey);
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'theme-item-name';
        nameSpan.textContent = THEME_NAMES[themeKey] || themeKey;
        item.appendChild(nameSpan);
        
        const swatchesDiv = document.createElement('div');
        swatchesDiv.className = 'theme-item-swatches';
        THEMES[themeKey].forEach(color => {
            const swatch = document.createElement('div');
            swatch.className = 'theme-item-swatch';
            swatch.style.background = color;
            swatchesDiv.appendChild(swatch);
        });
        item.appendChild(swatchesDiv);
        
        themeContainer.appendChild(item);
    });
    
    // Hex input preview
    document.getElementById('hexInput').addEventListener('input', function() {
        let hex = this.value.trim();
        if (!hex.startsWith('#')) hex = '#' + hex;
        if (/^#[0-9A-Fa-f]{6}$/.test(hex)) {
            document.getElementById('hexPreview').style.background = hex;
        }
    });

    // Tolerance slider
    document.getElementById('toleranceSlider').addEventListener('input', function() {
        colorTolerance = parseInt(this.value);
        document.getElementById('toleranceValue').textContent = colorTolerance;
    });

    // Alt key tracking
    document.addEventListener('keydown', function(e) {
        if (e.altKey) {
            altHeld = true;
            document.getElementById('canvasWrapper').classList.add('alt-held');
        }
    });
    
    document.addEventListener('keyup', function(e) {
        if (!e.altKey) {
            altHeld = false;
            document.getElementById('canvasWrapper').classList.remove('alt-held');
        }
    });
    
    const canvasInner = document.getElementById('canvasInner');
    const canvasWrapper = document.getElementById('canvasWrapper');
    
    // Canvas click for picker mode
    canvasInner.addEventListener('click', function(e) {
        if (!pickerMode || !originalImageData || draggingMarker !== null) return;
        if (altHeld) return;
        if (e.target.closest('.picker-marker')) return;
        
        const coords = getCanvasCoords(e);
        const x = coords.x;
        const y = coords.y;
        
        if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) return;
        
        const idx = (y * canvas.width + x) * 4;
        const r = originalImageData.data[idx];
        const g = originalImageData.data[idx + 1];
        const b = originalImageData.data[idx + 2];
        
        if (!shouldKeepPickedMarkers) {
            addPickedColor([r, g, b], x, y);
        }
    });
    
    // Pan functionality
    canvasWrapper.addEventListener('mousedown', function(e) {
        if (e.target.closest('.picker-marker') || e.target.closest('.zoom-slider-container') || 
            e.target.closest('.zoom-controls') || e.target.closest('.picker-overlay')) return;
        if (zoomLevel <= 1) return;
        if (pickerMode && !altHeld) return;
        
        isPanning = true;
        panStartX = e.clientX - panX;
        panStartY = e.clientY - panY;
        canvasWrapper.classList.add('panning');
    });
    
    document.addEventListener('mousemove', function(e) {
        if (isPanning) {
            panX = e.clientX - panStartX;
            panY = e.clientY - panStartY;
            constrainPan();
            updateCanvasTransform();
        }
        
        if (draggingMarker !== null) {
            const coords = getCanvasCoords(e);
            let x = coords.x;
            let y = coords.y;
            
            x = Math.max(0, Math.min(canvas.width - 1, x));
            y = Math.max(0, Math.min(canvas.height - 1, y));
            
            pickedPositions[draggingMarker].x = x;
            pickedPositions[draggingMarker].y = y;
            
            const idx = (y * canvas.width + x) * 4;
            const r = originalImageData.data[idx];
            const g = originalImageData.data[idx + 1];
            const b = originalImageData.data[idx + 2];
            pickedPositions[draggingMarker].color = [r, g, b];
            pickedColors[draggingMarker] = [r, g, b];
            
            updateMarkers();
            updatePickerOverlay();
        }
    });
    
    document.addEventListener('mouseup', function() {
        if (isPanning) {
            isPanning = false;
            canvasWrapper.classList.remove('panning');
        }
        if (draggingMarker !== null) {
            const marker = document.querySelector(`.picker-marker[data-index="${draggingMarker}"]`);
            if (marker) marker.classList.remove('dragging');
            draggingMarker = null;
        }
    });
    
    // Alt+scroll to zoom at cursor position
    canvasWrapper.addEventListener('wheel', function(e) {
        if (e.altKey) {
            e.preventDefault();

            // Apply zoom
            const delta = e.deltaY > 0 ? -0.15 : 0.15;
            const oldZoom = zoomLevel;
            zoomLevel = Math.max(1, Math.min(4, zoomLevel + delta));

            // Reset pan when returning to zoom 1
            if (zoomLevel <= 1) {
                panX = 0;
                panY = 0;
            } else {
                // Scale pan to maintain relative position
                const zoomRatio = zoomLevel / oldZoom;
                panX *= zoomRatio;
                panY *= zoomRatio;
            }

            constrainPan();
            updateCanvasTransform(true);
            updateZoomDisplay();
        }
    }, { passive: false });
});

// ============================================
// File Handling
// ============================================

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            loadImage(img);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function loadImage(img) {
    document.getElementById('uploadZone').classList.add('hidden');
    canvas.style.display = 'block';
    document.getElementById('zoomControls').classList.remove('hidden');
    document.getElementById('zoomSliderContainer').classList.remove('hidden');
    document.getElementById('pickerOverlay').classList.remove('hidden');
    document.getElementById('colorStripContainer').classList.remove('hidden');
    document.getElementById('toleranceContainer').classList.remove('hidden');
    
    zoomLevel = 1;
    panX = 0;
    panY = 0;
    updateCanvasTransform(true);
    updateZoomDisplay();

    if (!shouldKeepPickedMarkers) {
        clearMarkers();
    }
    if (pickerMode || shouldKeepPickedMarkers) {
        pickedPositions.forEach((_, i) => createMarker(i));
    }
    updatePickerOverlay();
    
    let width = img.width;
    let height = img.height;

    canvas.width = width;
    canvas.height = height;

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0, width, height);

    imageData = ctx.getImageData(0, 0, width, height);
    originalImageData = ctx.getImageData(0, 0, width, height);

    // Auto-size the canvas area to fit the image
    autoSizeCanvasArea();

    // Update high-quality display
    updateDisplayCanvas();

    extractPalette();
    setStatus('Image loaded. Palette extracted.');
}

function extractPalette() {
    showLoading();
    
    setTimeout(() => {
        const colors = extractColorsKMeans(originalImageData, 20);
        fullColorDistribution = colors;
        
        originalPalette = colors.slice(0, originCount).map(c => [...c.color]);
        colorPercentages = colors.slice(0, originCount).map(c => c.pct);
        targetPalette = originalPalette.map(c => [...c]);
        
        // Initialize column mapping: each origin maps to its corresponding target column
        originToColumn = [];
        columnBypass = []; // Reset bypass states
        for (let i = 0; i < originCount; i++) {
            if (i < targetCount) {
                originToColumn[i] = i;
            } else {
                originToColumn[i] = 'bank'; // Extra origins go to bank
            }
        }
        shouldKeepPickedMarkers = false;
        
        renderColorStrip();
        renderColumnMapping();
        updateHarmonyWheel();
        updateGradientFromSelectedColor();
        renderThemesSortedByMatch(); // Sort themes by match to extracted palette
        hideLoading();
    }, 50);
}

function extractColorsKMeans(imgData, numColors) {
    const pixels = [];
    const step = Math.max(1, Math.floor(imgData.data.length / 4 / 50000));
    
    for (let i = 0; i < imgData.data.length; i += 4 * step) {
        pixels.push([imgData.data[i], imgData.data[i+1], imgData.data[i+2]]);
    }
    
    // K-means++ initialization
    const centroids = [];
    const labPixels = pixels.map(p => RGB2LAB(p));
    
    centroids.push(labPixels[Math.floor(Math.random() * labPixels.length)]);
    
    while (centroids.length < numColors) {
        const distances = labPixels.map(p => {
            const minDist = Math.min(...centroids.map(c => 
                Math.pow(p[0]-c[0], 2) + Math.pow(p[1]-c[1], 2) + Math.pow(p[2]-c[2], 2)
            ));
            return minDist;
        });
        
        const totalDist = distances.reduce((a, b) => a + b, 0);
        let random = Math.random() * totalDist;
        let idx = 0;
        while (random > 0 && idx < distances.length) {
            random -= distances[idx];
            idx++;
        }
        centroids.push(labPixels[Math.max(0, idx - 1)]);
    }
    
    // K-means iterations
    for (let iter = 0; iter < 10; iter++) {
        const clusters = Array(numColors).fill(null).map(() => []);
        
        labPixels.forEach((p, i) => {
            let minDist = Infinity;
            let minIdx = 0;
            centroids.forEach((c, j) => {
                const dist = Math.pow(p[0]-c[0], 2) + Math.pow(p[1]-c[1], 2) + Math.pow(p[2]-c[2], 2);
                if (dist < minDist) {
                    minDist = dist;
                    minIdx = j;
                }
            });
            clusters[minIdx].push(i);
        });
        
        clusters.forEach((cluster, i) => {
            if (cluster.length > 0) {
                const avg = [0, 0, 0];
                cluster.forEach(idx => {
                    avg[0] += labPixels[idx][0];
                    avg[1] += labPixels[idx][1];
                    avg[2] += labPixels[idx][2];
                });
                centroids[i] = [avg[0]/cluster.length, avg[1]/cluster.length, avg[2]/cluster.length];
            }
        });
    }
    
    // Calculate percentages using closest-match assignment (ensures sum = 100%)
    const counts = new Array(numColors).fill(0);
    labPixels.forEach(p => {
        let minDist = Infinity;
        let minIdx = 0;
        centroids.forEach((c, j) => {
            const dist = Math.pow(p[0]-c[0], 2) + Math.pow(p[1]-c[1], 2) + Math.pow(p[2]-c[2], 2);
            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
            }
        });
        counts[minIdx]++;
    });
    
    const results = centroids.map((lab, i) => {
        const rgb = LAB2RGB(lab);
        return {
            color: [Math.round(rgb[0]), Math.round(rgb[1]), Math.round(rgb[2])],
            pct: (counts[i] / labPixels.length) * 100
        };
    });
    
    // Sort by percentage (highest first)
    results.sort((a, b) => b.pct - a.pct);

    return results;
}

// Merge similar colors based on tolerance
function mergeColorsWithTolerance(colors, tolerance) {
    if (tolerance === 0 || colors.length === 0) return colors;

    // Convert tolerance to LAB distance threshold (tolerance 50 = ~50 LAB units)
    const labThreshold = tolerance;

    const merged = [];
    const used = new Set();

    for (let i = 0; i < colors.length; i++) {
        if (used.has(i)) continue;

        const baseColor = colors[i];
        const baseLab = RGB2LAB(baseColor.color);
        let totalPct = baseColor.pct;
        let colorSum = [...baseColor.color];
        let count = 1;

        // Find all similar colors
        for (let j = i + 1; j < colors.length; j++) {
            if (used.has(j)) continue;

            const testLab = RGB2LAB(colors[j].color);
            const dist = Math.sqrt(
                Math.pow(baseLab[0] - testLab[0], 2) +
                Math.pow(baseLab[1] - testLab[1], 2) +
                Math.pow(baseLab[2] - testLab[2], 2)
            );

            if (dist <= labThreshold) {
                used.add(j);
                totalPct += colors[j].pct;
                // Weighted average of colors by percentage
                colorSum[0] += colors[j].color[0] * colors[j].pct;
                colorSum[1] += colors[j].color[1] * colors[j].pct;
                colorSum[2] += colors[j].color[2] * colors[j].pct;
                count++;
            }
        }

        // Calculate weighted average color
        if (count > 1) {
            const avgColor = [
                Math.round((baseColor.color[0] * baseColor.pct + colorSum[0] - baseColor.color[0]) / totalPct),
                Math.round((baseColor.color[1] * baseColor.pct + colorSum[1] - baseColor.color[1]) / totalPct),
                Math.round((baseColor.color[2] * baseColor.pct + colorSum[2] - baseColor.color[2]) / totalPct)
            ];
            merged.push({ color: avgColor, pct: totalPct });
        } else {
            merged.push({ color: [...baseColor.color], pct: totalPct });
        }

        used.add(i);
    }

    // Re-sort by percentage
    merged.sort((a, b) => b.pct - a.pct);

    return merged;
}

// Reset tolerance to default
function resetTolerance() {
    colorTolerance = DEFAULT_COLOR_TOLERANCE;
    document.getElementById('toleranceSlider').value = colorTolerance;
    document.getElementById('toleranceValue').textContent = colorTolerance;
    setStatus('Tolerance reset to default');
}

// Re-extract palette with current tolerance setting
function reExtractWithTolerance() {
    if (!originalImageData) {
        setStatus('Load an image first');
        return;
    }

    showLoading();
    setStatus('Recalculating percentages...');

    setTimeout(() => {
        // DON'T re-run k-means - just recalculate percentages for existing origin colors
        // This preserves the user's picked colors

        if (originalPalette.length === 0) {
            hideLoading();
            setStatus('No colors to recalculate. Pick or extract colors first.');
            return;
        }

        // Recalculate percentages based on current origin palette
        const labOrigins = originalPalette.map(c => RGB2LAB(c));
        const counts = new Array(originalPalette.length).fill(0);
        let totalSamples = 0;

        // Sample every 16th pixel for speed (i += 64 means every 16th pixel since each pixel is 4 bytes)
        for (let i = 0; i < originalImageData.data.length; i += 64) {
            const lab = RGB2LAB([
                originalImageData.data[i],
                originalImageData.data[i + 1],
                originalImageData.data[i + 2]
            ]);

            // Find closest origin color
            let minDist = Infinity;
            let minIdx = 0;
            for (let j = 0; j < labOrigins.length; j++) {
                const dist = Math.pow(lab[0] - labOrigins[j][0], 2) +
                    Math.pow(lab[1] - labOrigins[j][1], 2) +
                    Math.pow(lab[2] - labOrigins[j][2], 2);
                if (dist < minDist) {
                    minDist = dist;
                    minIdx = j;
                }
            }
            counts[minIdx]++;
            totalSamples++;
        }

        // Update percentages
        colorPercentages = counts.map(c => (c / totalSamples) * 100);

        // Update fullColorDistribution to match current palette
        fullColorDistribution = originalPalette.map((color, i) => ({
            color: [...color],
            pct: colorPercentages[i]
        }));

        renderColorStrip();
        renderColumnMapping();
        hideLoading();

        setStatus(`Recalculated percentages. Origin colors preserved.`);
    }, 50);
}

// Render color strip with optional descale
function renderColorStrip() {
    const strip = document.getElementById('colorStrip');
    strip.innerHTML = '';
    
    const descaleEnabled = document.getElementById('descaleBgCheckbox').checked;
    let displayData = [...fullColorDistribution];
    
    if (descaleEnabled && displayData.length >= 2) {
        const secondLargest = displayData[1].pct;
        displayData = displayData.map((item, i) => ({
            ...item,
            displayPct: i === 0 ? secondLargest : item.pct
        }));
        const total = displayData.reduce((sum, item) => sum + item.displayPct, 0);
        displayData = displayData.map(item => ({
            ...item,
            displayPct: (item.displayPct / total) * 100
        }));
    } else {
        displayData = displayData.map(item => ({ ...item, displayPct: item.pct }));
    }
    
    for (const item of displayData) {
        if (item.pct < 0.5) continue;
        const seg = document.createElement('div');
        seg.className = 'color-strip-segment';
        seg.style.background = rgbToHex(...item.color);
        seg.style.width = item.displayPct + '%';
        seg.dataset.pct = item.pct.toFixed(1) + '%';
        seg.title = rgbToHex(...item.color) + ' - ' + item.pct.toFixed(1) + '%';
        strip.appendChild(seg);
    }
}

// Render the recolored image's color distribution strip
function renderRecoloredStrip() {
    const strip = document.getElementById('recoloredStrip');
    if (!strip) return;
    strip.innerHTML = '';

    if (!imageData || targetPalette.length === 0) {
        return;
    }

    // Calculate color distribution of the recolored image based on target palette
    const labTargets = targetPalette.map(c => RGB2LAB(c));
    const counts = new Array(targetPalette.length).fill(0);
    let totalSamples = 0;

    // Sample every 16th pixel for speed
    for (let i = 0; i < imageData.data.length; i += 64) {
        const lab = RGB2LAB([
            imageData.data[i],
            imageData.data[i + 1],
            imageData.data[i + 2]
        ]);

        // Find closest target color
        let minDist = Infinity;
        let minIdx = 0;
        for (let j = 0; j < labTargets.length; j++) {
            const dist = Math.pow(lab[0] - labTargets[j][0], 2) +
                Math.pow(lab[1] - labTargets[j][1], 2) +
                Math.pow(lab[2] - labTargets[j][2], 2);
            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
            }
        }
        counts[minIdx]++;
        totalSamples++;
    }

    // Build distribution data
    const recoloredDistribution = targetPalette.map((color, i) => ({
        color: [...color],
        pct: (counts[i] / totalSamples) * 100
    }));

    // Sort by percentage descending for display
    recoloredDistribution.sort((a, b) => b.pct - a.pct);

    for (const item of recoloredDistribution) {
        if (item.pct < 0.5) continue;
        const seg = document.createElement('div');
        seg.className = 'color-strip-segment';
        seg.style.background = rgbToHex(...item.color);
        seg.style.width = item.pct + '%';
        seg.dataset.pct = item.pct.toFixed(1) + '%';
        seg.title = rgbToHex(...item.color) + ' - ' + item.pct.toFixed(1) + '%';
        strip.appendChild(seg);
    }
}

function changeOriginCount(delta) {
    const newCount = Math.max(2, Math.min(20, originCount + delta));
    if (newCount !== originCount) {
        originCount = newCount;
        document.getElementById('originCountDisplay').value = originCount;
        
        originalPalette = fullColorDistribution.slice(0, originCount).map(d => [...d.color]);
        colorPercentages = fullColorDistribution.slice(0, originCount).map(d => d.pct);
        
        // Don't auto-adjust target count
        targetPalette = targetPalette.slice(0, targetCount);
        while (targetPalette.length < targetCount) {
            targetPalette.push([...originalPalette[targetPalette.length % originalPalette.length]]);
        }
        
        // Re-initialize column mapping
        originToColumn = [];
        columnBypass = []; // Reset bypass states
        for (let i = 0; i < originCount; i++) {
            if (i < targetCount) {
                originToColumn[i] = i;
            } else {
                originToColumn[i] = 'bank'; // Extra origins go to bank
            }
        }

        renderColumnMapping();
    }
}

function changeTargetCount(delta) {
    const newCount = Math.max(1, Math.min(20, targetCount + delta));
    if (newCount !== targetCount) {
        targetCount = newCount;
        document.getElementById('targetCountDisplay').value = targetCount;
        
        // Update data attribute for responsive styling
        document.getElementById('columnMappingContainer').setAttribute('data-target-count', targetCount);
        
        if (targetCount > targetPalette.length) {
            while (targetPalette.length < targetCount) {
                const idx = targetPalette.length;
                if (idx < originalPalette.length) {
                    targetPalette.push([...originalPalette[idx]]);
                } else {
                    targetPalette.push([...originalPalette[idx % originalPalette.length]]);
                }
            }
        } else {
            targetPalette = targetPalette.slice(0, targetCount);
        }
        
        // Re-assign origins that were in removed columns
        for (let i = 0; i < originCount; i++) {
            if (typeof originToColumn[i] === 'number' && originToColumn[i] >= targetCount) {
                originToColumn[i] = 'bank';
            }
        }
        
        renderColumnMapping();
    }
}

// ============================================
// Column-based Mapping UI
// ============================================

function renderColumnMapping() {
    const container = document.getElementById('mappingColumns');
    const targetsRow = document.getElementById('mappingTargetsRow');
    const mappingContainer = document.getElementById('columnMappingContainer');
    const overflowBank = document.getElementById('originOverflowBank');
    
    // Update data attribute for responsive styling
    mappingContainer.setAttribute('data-target-count', targetCount);
    const overflowGrid = document.getElementById('overflowOriginsGrid');
    
    container.innerHTML = '';
    targetsRow.innerHTML = '';
    overflowGrid.innerHTML = '';
    
    // Get origins by type (bank origins are now treated as locked)
    const bankOrigins = [];
    const columnOrigins = {};
    
    for (let i = 0; i < originCount; i++) {
        const col = originToColumn[i];
        if (col === 'locked' || col === 'bank') {
            bankOrigins.push(i);
        } else {
            if (!columnOrigins[col]) columnOrigins[col] = [];
            columnOrigins[col].push(i);
        }
    }
    
    // Check if there are unused colors from extraction that can be added
    const unusedColors = getUnusedExtractedColors();
    const showAddButton = unusedColors.length > 0;

    // Bank is ALWAYS visible once an image is loaded - it's where you drag colors to lock them
    if (originalImageData) {
        overflowBank.classList.add('visible');

        // Drag events for overflow bank
        overflowGrid.addEventListener('dragover', (e) => {
            e.preventDefault();
            overflowGrid.classList.add('drag-over');
        });
        overflowGrid.addEventListener('dragleave', () => {
            overflowGrid.classList.remove('drag-over');
        });
        overflowGrid.addEventListener('drop', (e) => {
            e.preventDefault();
            overflowGrid.classList.remove('drag-over');
            if (draggedOriginIndex !== null) {
                originToColumn[draggedOriginIndex] = 'bank';
                renderColumnMapping();
                autoRecolorImage();
            }
        });

        // Add bank origins to overflow
        bankOrigins.forEach(i => {
            const swatch = createOriginSwatch(i);
            overflowGrid.appendChild(swatch);
        });

        // Add the "add swatch" button to the bank if origins >= 5
        if (showAddButton && originCount >= 5) {
            const addBtn = createAddSwatchButton(unusedColors, 'bank');
            overflowGrid.appendChild(addBtn);
        }
    } else {
        overflowBank.classList.remove('visible');
    }

    // Find the first empty column (no origins assigned to it)
    let firstEmptyColumn = -1;
    for (let colIdx = 0; colIdx < targetCount; colIdx++) {
        if (!columnOrigins[colIdx] || columnOrigins[colIdx].length === 0) {
            firstEmptyColumn = colIdx;
            break;
        }
    }
    
    // Create target columns with origins
    for (let colIdx = 0; colIdx < targetCount; colIdx++) {
        const column = document.createElement('div');
        column.className = 'mapping-column';
        column.dataset.columnIndex = colIdx;
        
        // Origins drop zone
        const originsZone = document.createElement('div');
        originsZone.className = 'column-origins';
        originsZone.dataset.columnIndex = colIdx;
        
        // Drag events for drop zone
        originsZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            originsZone.classList.add('drag-over');
        });
        originsZone.addEventListener('dragleave', () => {
            originsZone.classList.remove('drag-over');
        });
        originsZone.addEventListener('drop', (e) => {
            e.preventDefault();
            originsZone.classList.remove('drag-over');
            if (draggedOriginIndex !== null) {
                originToColumn[draggedOriginIndex] = colIdx;
                renderColumnMapping();
                autoRecolorImage();
            }
        });
        
        // Add origins that belong to this column
        if (columnOrigins[colIdx]) {
            columnOrigins[colIdx].forEach(i => {
                const originSwatch = createOriginSwatch(i);
                originsZone.appendChild(originSwatch);
            });
        }

        // Add the "add swatch" button to the first EMPTY column if origins < 5
        if (colIdx === firstEmptyColumn && showAddButton && originCount < 5) {
            const addBtn = createAddSwatchButton(unusedColors, colIdx);
            originsZone.appendChild(addBtn);
        }

        column.appendChild(originsZone);
        container.appendChild(column);
    }
    
    // Render target swatches in a separate row
    for (let colIdx = 0; colIdx < targetCount; colIdx++) {
        const targetColumn = document.createElement('div');
        targetColumn.className = 'mapping-targets-column';
        targetColumn.dataset.columnIndex = colIdx;
        
        const targetDiv = document.createElement('div');
        targetDiv.className = 'column-target';
        
        const slot = document.createElement('div');
        slot.className = 'color-slot' + (colIdx === selectedSlotIndex ? ' selected' : '');
        slot.style.background = rgbToHex(...targetPalette[colIdx]);
        slot.title = rgbToHex(...targetPalette[colIdx]) + ' (click to select, use picker button below)';
        slot.onclick = () => selectSlot(colIdx);

        // Delete button on the slot (styled like origin X button)
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'target-delete-btn';
        deleteBtn.innerHTML = '×';
        deleteBtn.title = 'Remove this target';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deleteTarget(colIdx);
        };
        slot.appendChild(deleteBtn);

        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.className = 'hidden-color-input';
        colorInput.id = 'colorPicker_' + colIdx;
        
        const hexLabel = document.createElement('div');
        hexLabel.className = 'swatch-hex';
        hexLabel.textContent = rgbToHex(...targetPalette[colIdx]);
        
        // Calculate percentage sum for this column
        let pctSum = 0;
        for (let i = 0; i < originCount; i++) {
            if (originToColumn[i] === colIdx) {
                pctSum += colorPercentages[i] || 0;
            }
        }
        
        const pctLabel = document.createElement('div');
        pctLabel.className = 'column-percentage';
        pctLabel.textContent = pctSum.toFixed(1) + '%';
        
        // Buttons
        const btnRow = document.createElement('div');
        btnRow.className = 'swatch-buttons';
        
        const revertBtn = document.createElement('button');
        const origIdx = colIdx % originalPalette.length;
        if (colorsMatch(targetPalette[colIdx], originalPalette[origIdx])) {
            revertBtn.classList.add('is-original');
        }
        revertBtn.innerHTML = '↩';
        revertBtn.title = 'Revert to original';
        revertBtn.onclick = (e) => {
            e.stopPropagation();
            revertSlot(colIdx);
        };
        
        const pickerBtn = document.createElement('button');
        pickerBtn.innerHTML = '🎨';
        pickerBtn.title = 'Open color picker';
        pickerBtn.onclick = (e) => {
            e.stopPropagation();
            toggleSwatchPicker(colIdx, targetDiv);
        };

        // Bypass/Lock button - when active, origins keep their original colors
        const bypassBtn = document.createElement('button');
        bypassBtn.className = 'bypass-btn' + (columnBypass[colIdx] ? ' active' : '');
        bypassBtn.innerHTML = '🔒';
        bypassBtn.title = columnBypass[colIdx] ? 'Bypass ON: Origins keep original colors' : 'Bypass OFF: Origins consolidated to target';
        bypassBtn.onclick = (e) => {
            e.stopPropagation();
            toggleColumnBypass(colIdx);
        };

        btnRow.appendChild(revertBtn);
        btnRow.appendChild(pickerBtn);
        btnRow.appendChild(bypassBtn);

        targetDiv.appendChild(slot);
        targetDiv.appendChild(colorInput);
        targetDiv.appendChild(hexLabel);
        targetDiv.appendChild(pctLabel);
        targetDiv.appendChild(btnRow);

        // Create per-swatch color picker (hidden by default)
        const swatchPicker = createSwatchColorPicker(colIdx);
        targetDiv.appendChild(swatchPicker);

        // Add category label below everything
        const categoryLabel = document.createElement('div');
        categoryLabel.className = 'target-category-label';
        categoryLabel.textContent = getTargetCategoryLabel(colIdx);
        targetDiv.appendChild(categoryLabel);

        targetColumn.appendChild(targetDiv);
        targetsRow.appendChild(targetColumn);
    }
    
    // Update hex input
    if (targetPalette.length > 0) {
        const hex = rgbToHex(...targetPalette[selectedSlotIndex]);
        document.getElementById('hexInput').value = hex;
        document.getElementById('hexPreview').style.background = hex;
    }
    
    // Update debug display
    const debugEl = document.getElementById('mappingDebug');
    if (debugEl) {
        debugEl.textContent = 'Mapping: ' + JSON.stringify(originToColumn);
    }
}

function toggleColumnBypass(colIdx) {
    columnBypass[colIdx] = !columnBypass[colIdx];
    renderColumnMapping();
    if (livePreviewEnabled) {
        autoRecolorImage();
    }
}

function deleteTarget(colIdx) {
    // Move all origins from this target back to bank
    for (let i = 0; i < originCount; i++) {
        if (originToColumn[i] === colIdx) {
            originToColumn[i] = 'bank';
        }
    }
    
    // Remove the target
    targetPalette.splice(colIdx, 1);
    targetCount--;
    document.getElementById('targetCountDisplay').value = targetCount;
    
    // Shift indices for targets after the deleted one
    for (let i = 0; i < originCount; i++) {
        if (typeof originToColumn[i] === 'number' && originToColumn[i] > colIdx) {
            originToColumn[i]--;
        }
    }
    
    // Adjust selected slot
    if (selectedSlotIndex >= targetCount) {
        selectedSlotIndex = Math.max(0, targetCount - 1);
    }
    
    renderColumnMapping();
    setStatus(`Target ${colIdx + 1} removed. Origins moved to bank.`);
}

function createOriginSwatch(originIndex) {
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.flexDirection = 'column';
    wrapper.style.alignItems = 'center';
    wrapper.style.marginBottom = '8px';

    const swatch = document.createElement('div');
    swatch.className = 'origin-swatch';
    swatch.style.background = rgbToHex(...originalPalette[originIndex]);
    swatch.draggable = true;
    swatch.dataset.originIndex = originIndex;
    swatch.title = `${rgbToHex(...originalPalette[originIndex])} - ${colorPercentages[originIndex]?.toFixed(1)}% (drag to reassign)`;

    // Add number label for picked colors
    if (shouldKeepPickedMarkers && pickedColors.length > 0) {
        const numberLabel = document.createElement('div');
        numberLabel.className = 'origin-swatch-number';
        numberLabel.textContent = (originIndex + 1).toString();
        swatch.appendChild(numberLabel);
    }

    // Add delete button (X)
    const deleteBtn = document.createElement('div');
    deleteBtn.className = 'origin-swatch-delete';
    deleteBtn.innerHTML = '×';
    deleteBtn.title = 'Remove from mapping';
    deleteBtn.onclick = (e) => {
        e.stopPropagation();
        e.preventDefault();
        removeOriginFromMapping(originIndex);
    };
    swatch.appendChild(deleteBtn);

    swatch.addEventListener('dragstart', (e) => {
        draggedOriginIndex = originIndex;
        swatch.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
    });

    swatch.addEventListener('dragend', () => {
        swatch.classList.remove('dragging');
        draggedOriginIndex = null;
        document.querySelectorAll('.column-origins.drag-over, .overflow-origins-grid.drag-over').forEach(el => {
            el.classList.remove('drag-over');
        });
    });

    wrapper.appendChild(swatch);

    const info = document.createElement('div');
    info.className = 'origin-swatch-info';
    const hex = rgbToHex(...originalPalette[originIndex]);
    info.innerHTML = `${hex}<br>${colorPercentages[originIndex]?.toFixed(1)}%`;
    wrapper.appendChild(info);

    return wrapper;
}

function removeOriginFromMapping(originIndex) {
    // Remove this origin from the palette entirely
    originalPalette.splice(originIndex, 1);
    colorPercentages.splice(originIndex, 1);
    originCount--;
    document.getElementById('originCountDisplay').value = originCount;

    // Rebuild originToColumn array - shift indices down
    const newOriginToColumn = [];
    for (let i = 0; i < originToColumn.length; i++) {
        if (i < originIndex) {
            newOriginToColumn.push(originToColumn[i]);
        } else if (i > originIndex) {
            newOriginToColumn.push(originToColumn[i]);
        }
        // Skip the removed index
    }
    originToColumn = newOriginToColumn;

    renderColumnMapping();
    autoRecolorImage();
    setStatus(`Removed origin color ${originIndex + 1} from mapping`);
}

function colorsMatch(c1, c2) {
    return c1[0] === c2[0] && c1[1] === c2[1] && c1[2] === c2[2];
}

// Get extracted colors not currently in the origin palette
function getUnusedExtractedColors() {
    if (!fullColorDistribution || fullColorDistribution.length === 0) return [];

    const unused = [];
    for (const item of fullColorDistribution) {
        const isUsed = originalPalette.some(op =>
            Math.abs(op[0] - item.color[0]) < 3 &&
            Math.abs(op[1] - item.color[1]) < 3 &&
            Math.abs(op[2] - item.color[2]) < 3
        );
        if (!isUsed) {
            unused.push(item);
        }
    }
    return unused;
}

// Create the "add swatch" button with dropdown
function createAddSwatchButton(unusedColors, targetColumn) {
    const wrapper = document.createElement('div');
    wrapper.className = 'add-swatch-wrapper';

    const btn = document.createElement('button');
    btn.className = 'add-swatch-btn';
    btn.innerHTML = '+';
    btn.title = 'Add a color from extracted palette';

    const dropdown = document.createElement('div');
    dropdown.className = 'add-swatch-dropdown';

    unusedColors.forEach(item => {
        const option = document.createElement('div');
        option.className = 'add-swatch-option';

        const colorSwatch = document.createElement('div');
        colorSwatch.className = 'add-swatch-color';
        colorSwatch.style.background = rgbToHex(...item.color);

        const info = document.createElement('div');
        info.className = 'add-swatch-info';
        info.innerHTML = `${rgbToHex(...item.color)}<br>${item.pct.toFixed(1)}%`;

        option.appendChild(colorSwatch);
        option.appendChild(info);

        option.onclick = (e) => {
            e.stopPropagation();
            addOriginColor(item.color, item.pct, targetColumn);
            dropdown.classList.remove('visible');
        };

        dropdown.appendChild(option);
    });

    btn.onclick = (e) => {
        e.stopPropagation();
        // Close any other open dropdowns
        document.querySelectorAll('.add-swatch-dropdown.visible').forEach(d => d.classList.remove('visible'));
        dropdown.classList.toggle('visible');
    };

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!wrapper.contains(e.target)) {
            dropdown.classList.remove('visible');
        }
    });

    const label = document.createElement('div');
    label.className = 'add-swatch-label';
    label.textContent = 'Add Color';

    wrapper.appendChild(btn);
    wrapper.appendChild(dropdown);
    wrapper.appendChild(label);

    return wrapper;
}

// Add a new origin color to the palette
function addOriginColor(color, pct, targetColumn) {
    originalPalette.push([...color]);
    colorPercentages.push(pct);
    originCount++;
    document.getElementById('originCountDisplay').value = originCount;

    // Assign to the specified column or bank
    if (targetColumn === 'bank' || originCount > targetCount) {
        originToColumn.push('bank');
    } else {
        originToColumn.push(targetColumn);
    }

    renderColumnMapping();
    autoRecolorImage();
    setStatus(`Added ${rgbToHex(...color)} to origin palette`);
}

function selectSlot(index) {
    selectedSlotIndex = index;
    renderColumnMapping();
    updateHarmonyWheel();
    updateGradientFromSelectedColor();
}

function openColorPicker(index) {
    const picker = document.getElementById('colorPicker_' + index);
    if (picker) picker.click();
}

// Per-swatch color picker state
let activeSwatchPicker = null;
let swatchPickerState = {};

function createSwatchColorPicker(colIdx) {
    const picker = document.createElement('div');
    picker.className = 'swatch-color-picker';
    picker.id = `swatchPicker_${colIdx}`;

    // Get current color
    const currentColor = targetPalette[colIdx];
    const [h, s, v] = rgbToHsv(...currentColor);

    // Initialize state for this picker
    swatchPickerState[colIdx] = { h, s, v };

    // Gradient area
    const gradient = document.createElement('div');
    gradient.className = 'swatch-picker-gradient';
    gradient.id = `swatchGradient_${colIdx}`;

    const gradientWhite = document.createElement('div');
    gradientWhite.className = 'swatch-picker-gradient-white';

    const gradientBlack = document.createElement('div');
    gradientBlack.className = 'swatch-picker-gradient-black';

    const cursor = document.createElement('div');
    cursor.className = 'swatch-picker-cursor';
    cursor.id = `swatchCursor_${colIdx}`;
    cursor.style.left = s + '%';
    cursor.style.top = (100 - v) + '%';

    gradient.appendChild(gradientWhite);
    gradient.appendChild(gradientBlack);
    gradient.appendChild(cursor);

    // Set initial gradient color
    const [gr, gg, gb] = hslToRgb(h, 100, 50);
    gradient.style.background = `rgb(${gr}, ${gg}, ${gb})`;

    // Gradient mouse events
    let isDraggingSwatchGradient = false;

    gradient.addEventListener('mousedown', (e) => {
        isDraggingSwatchGradient = true;
        updateSwatchGradientFromMouse(e, colIdx);
    });

    document.addEventListener('mousemove', (e) => {
        if (isDraggingSwatchGradient && activeSwatchPicker === colIdx) {
            updateSwatchGradientFromMouse(e, colIdx);
        }
    });

    document.addEventListener('mouseup', () => {
        isDraggingSwatchGradient = false;
    });

    // Hue slider
    const hueSlider = document.createElement('input');
    hueSlider.type = 'range';
    hueSlider.className = 'swatch-picker-hue';
    hueSlider.id = `swatchHue_${colIdx}`;
    hueSlider.min = 0;
    hueSlider.max = 360;
    hueSlider.value = h;
    hueSlider.oninput = () => {
        swatchPickerState[colIdx].h = parseInt(hueSlider.value);
        updateSwatchGradientBackground(colIdx);
        updateSwatchPickerPreview(colIdx);
    };

    // Preview row
    const previewRow = document.createElement('div');
    previewRow.className = 'swatch-picker-preview-row';

    const preview = document.createElement('div');
    preview.className = 'swatch-picker-preview';
    preview.id = `swatchPreview_${colIdx}`;
    preview.style.background = rgbToHex(...currentColor);

    const hexInput = document.createElement('input');
    hexInput.type = 'text';
    hexInput.className = 'swatch-picker-hex';
    hexInput.id = `swatchHex_${colIdx}`;
    hexInput.value = rgbToHex(...currentColor);
    hexInput.maxLength = 7;
    hexInput.onchange = () => {
        let hex = hexInput.value.trim();
        if (!hex.startsWith('#')) hex = '#' + hex;
        if (/^#[0-9A-Fa-f]{6}$/.test(hex)) {
            const rgb = hexToRgb(hex);
            const [h, s, v] = rgbToHsv(...rgb);
            swatchPickerState[colIdx] = { h, s, v };
            updateSwatchGradientBackground(colIdx);
            updateSwatchPickerCursor(colIdx);
            document.getElementById(`swatchHue_${colIdx}`).value = h;
            updateSwatchPickerPreview(colIdx);
        }
    };

    previewRow.appendChild(preview);
    previewRow.appendChild(hexInput);

    // Apply button
    const applyBtn = document.createElement('button');
    applyBtn.className = 'swatch-picker-apply';
    applyBtn.textContent = 'Apply';
    applyBtn.onclick = () => {
        const state = swatchPickerState[colIdx];
        const rgb = hsvToRgb(state.h, state.s, state.v);
        targetPalette[colIdx] = rgb;
        closeAllSwatchPickers();
        renderColumnMapping();
        autoRecolorImage();
        setStatus(`Color ${colIdx + 1} set to ${rgbToHex(...rgb)}`);
    };

    picker.appendChild(gradient);
    picker.appendChild(hueSlider);
    picker.appendChild(previewRow);
    picker.appendChild(applyBtn);

    // Prevent clicks inside picker from closing it
    picker.onclick = (e) => e.stopPropagation();

    return picker;
}

function updateSwatchGradientFromMouse(e, colIdx) {
    const gradient = document.getElementById(`swatchGradient_${colIdx}`);
    const cursor = document.getElementById(`swatchCursor_${colIdx}`);
    if (!gradient || !cursor) return;

    const rect = gradient.getBoundingClientRect();
    let x = (e.clientX - rect.left) / rect.width;
    let y = (e.clientY - rect.top) / rect.height;

    x = Math.max(0, Math.min(1, x));
    y = Math.max(0, Math.min(1, y));

    swatchPickerState[colIdx].s = x * 100;
    swatchPickerState[colIdx].v = (1 - y) * 100;

    cursor.style.left = (x * 100) + '%';
    cursor.style.top = (y * 100) + '%';

    updateSwatchPickerPreview(colIdx);
}

function updateSwatchGradientBackground(colIdx) {
    const gradient = document.getElementById(`swatchGradient_${colIdx}`);
    if (!gradient) return;

    const h = swatchPickerState[colIdx].h;
    const [r, g, b] = hslToRgb(h, 100, 50);
    gradient.style.background = `rgb(${r}, ${g}, ${b})`;
}

function updateSwatchPickerCursor(colIdx) {
    const cursor = document.getElementById(`swatchCursor_${colIdx}`);
    if (!cursor) return;

    const state = swatchPickerState[colIdx];
    cursor.style.left = state.s + '%';
    cursor.style.top = (100 - state.v) + '%';
}

function updateSwatchPickerPreview(colIdx) {
    const preview = document.getElementById(`swatchPreview_${colIdx}`);
    const hexInput = document.getElementById(`swatchHex_${colIdx}`);
    if (!preview || !hexInput) return;

    const state = swatchPickerState[colIdx];
    const rgb = hsvToRgb(state.h, state.s, state.v);
    const hex = rgbToHex(...rgb);

    preview.style.background = hex;
    hexInput.value = hex;
}

function toggleSwatchPicker(colIdx, targetDiv) {
    const picker = document.getElementById(`swatchPicker_${colIdx}`);
    if (!picker) return;

    // Close other pickers first
    document.querySelectorAll('.swatch-color-picker.visible').forEach(p => {
        if (p.id !== `swatchPicker_${colIdx}`) {
            p.classList.remove('visible');
        }
    });

    const isVisible = picker.classList.contains('visible');

    if (isVisible) {
        picker.classList.remove('visible');
        activeSwatchPicker = null;
    } else {
        // Update picker state from current target color
        const currentColor = targetPalette[colIdx];
        const [h, s, v] = rgbToHsv(...currentColor);
        swatchPickerState[colIdx] = { h, s, v };

        // Update all picker UI elements
        updateSwatchGradientBackground(colIdx);
        updateSwatchPickerCursor(colIdx);
        document.getElementById(`swatchHue_${colIdx}`).value = h;
        updateSwatchPickerPreview(colIdx);

        picker.classList.add('visible');
        activeSwatchPicker = colIdx;
    }
}

function closeAllSwatchPickers() {
    document.querySelectorAll('.swatch-color-picker.visible').forEach(p => {
        p.classList.remove('visible');
    });
    activeSwatchPicker = null;
}

// Close pickers when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.swatch-color-picker') && !e.target.closest('.swatch-buttons')) {
        closeAllSwatchPickers();
    }
});

function revertSlot(index) {
    const origIdx = index % originalPalette.length;
    targetPalette[index] = [...originalPalette[origIdx]];
    renderColumnMapping();
    autoRecolorImage();
    setStatus(`Slot ${index + 1} reverted to original`);
}

function shuffleTargetPalette() {
    for (let i = targetPalette.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [targetPalette[i], targetPalette[j]] = [targetPalette[j], targetPalette[i]];
    }
    renderColumnMapping();
    setStatus('Target colors shuffled');
}

function applyHexToSelected() {
    let hex = document.getElementById('hexInput').value.trim();
    if (!hex.startsWith('#')) hex = '#' + hex;
    
    if (!/^#[0-9A-Fa-f]{6}$/.test(hex)) {
        setStatus('Invalid hex code. Use format: #FF5733');
        return;
    }
    
    const rgb = hexToRgb(hex);
    targetPalette[selectedSlotIndex] = rgb;
    renderColumnMapping();
    setStatus(`Color ${selectedSlotIndex + 1} set to ${hex}`);
    autoRecolorImage();
}

// ============================================
// Color Gradient Picker
// ============================================

let gradientHue = 0;
let gradientSaturation = 100;
let gradientValue = 100;
let isDraggingGradient = false;

function initColorGradientPicker() {
    const container = document.getElementById('colorGradientContainer');
    const cursor = document.getElementById('colorGradientCursor');
    
    if (!container || !cursor) return;
    
    // Update gradient background on hue change
    updateGradientBackground();
    
    // Mouse events for gradient area
    container.addEventListener('mousedown', (e) => {
        isDraggingGradient = true;
        updateGradientFromMouse(e);
    });
    
    document.addEventListener('mousemove', (e) => {
        if (isDraggingGradient) {
            updateGradientFromMouse(e);
        }
    });
    
    document.addEventListener('mouseup', () => {
        isDraggingGradient = false;
    });
}

function updateGradientFromMouse(e) {
    const container = document.getElementById('colorGradientContainer');
    const cursor = document.getElementById('colorGradientCursor');
    
    if (!container || !cursor) return;
    
    const rect = container.getBoundingClientRect();
    let x = (e.clientX - rect.left) / rect.width;
    let y = (e.clientY - rect.top) / rect.height;
    
    // Clamp values
    x = Math.max(0, Math.min(1, x));
    y = Math.max(0, Math.min(1, y));
    
    gradientSaturation = x * 100;
    gradientValue = (1 - y) * 100;
    
    // Update cursor position
    cursor.style.left = (x * 100) + '%';
    cursor.style.top = (y * 100) + '%';
    
    // Update selected color
    applyGradientColor();
}

function updateGradientHue(hue) {
    gradientHue = parseInt(hue);
    updateGradientBackground();
    applyGradientColor();
}

function updateGradientBackground() {
    const sv = document.getElementById('colorGradientSV');
    if (!sv) return;
    
    const [r, g, b] = hslToRgb(gradientHue, 100, 50);
    sv.style.background = `rgb(${r}, ${g}, ${b})`;
}

function applyGradientColor() {
    // Convert HSV to RGB
    const rgb = hsvToRgb(gradientHue, gradientSaturation, gradientValue);
    const hex = rgbToHex(...rgb);

    // Update hex input and preview ONLY - don't apply to slot
    document.getElementById('hexInput').value = hex;
    document.getElementById('hexPreview').style.background = hex;

    // Update cursor color
    const cursor = document.getElementById('colorGradientCursor');
    if (cursor) {
        cursor.style.background = hex;
    }

    // DO NOT auto-apply - wait for the "Set" button click
}

function hsvToRgb(h, s, v) {
    h = h / 360;
    s = s / 100;
    v = v / 100;
    
    let r, g, b;
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, v = max;
    
    const d = max - min;
    s = max === 0 ? 0 : d / max;
    
    if (max === min) {
        h = 0;
    } else {
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    
    return [h * 360, s * 100, v * 100];
}

function updateGradientFromSelectedColor() {
    if (targetPalette.length === 0) return;
    
    const rgb = targetPalette[selectedSlotIndex];
    const [h, s, v] = rgbToHsv(...rgb);
    
    gradientHue = h;
    gradientSaturation = s;
    gradientValue = v;
    
    // Update UI
    const hueSlider = document.getElementById('colorHueSlider');
    if (hueSlider) hueSlider.value = h;
    
    updateGradientBackground();
    
    // Update cursor position
    const cursor = document.getElementById('colorGradientCursor');
    if (cursor) {
        cursor.style.left = (s) + '%';
        cursor.style.top = (100 - v) + '%';
        cursor.style.background = rgbToHex(...rgb);
    }
}

// ============================================
// Recoloring
// ============================================

function applyRecolor() {
    if (!originalImageData) {
        setStatus('Load an image first');
        return;
    }
    
    showLoading();
    setStatus('Applying recolor...');
    
    setTimeout(() => {
        recolorImage();
        hideLoading();
        setStatus('Recolor applied!');
    }, 50);
}

function autoRecolorImage() {
    // Only recolor if an image is loaded and has been previously recolored
    if (!originalImageData) {
        return;
    }
    // Only auto-recolor if live preview is enabled
    if (!livePreviewEnabled) {
        return;
    }
    // Silently recolor without loading UI (but still async for UI responsiveness)
    setTimeout(() => recolorImage(), 10);
}

function toggleLivePreview(enabled) {
    livePreviewEnabled = enabled;
    if (enabled && originalImageData) {
        // Immediately trigger a recolor when turning on
        recolorImage();
    }
}

// Manual recolor trigger (Apply Recolor button)
function manualRecolorImage() {
    if (!originalImageData) {
        setStatus('Load an image first');
        return;
    }
    recolorImage();
}

function recolorImage() {
    const width = canvas.width;
    const height = canvas.height;
    const k = originalPalette.length;
    
    if (k === 0) return;
    
    showLoading();
    
    // Use setTimeout to let the loading indicator render
    setTimeout(() => {
        doRecolorImage();
        hideLoading();
    }, 50);
}

function doRecolorImage() {
    // Dispatch to selected algorithm
    if (selectedAlgorithm === 'rbf') {
        doRecolorRBF();
    } else {
        doRecolorSimple();
    }
}

// Simple weighted nearest-neighbor algorithm (v14 default)
function doRecolorSimple() {
    const width = canvas.width;
    const height = canvas.height;
    const k = originalPalette.length;

    if (k === 0) return;

    const oldLab = originalPalette.map(c => RGB2LAB(c));
    const newLab = [];

    // Build origin-to-target mapping based on columns
    for (let i = 0; i < k; i++) {
        const col = originToColumn[i];
        if (col === 'locked' || col === 'bank' || typeof col !== 'number' || col >= targetPalette.length) {
            newLab.push(oldLab[i]);
        } else if (columnBypass[col]) {
            newLab.push(oldLab[i]);
        } else {
            newLab.push(RGB2LAB(targetPalette[col]));
        }
    }

    const diffLab = oldLab.map((old, i) => [
        newLab[i][0] - old[0],
        newLab[i][1] - old[1],
        newLab[i][2] - old[2]
    ]);

    const luminosity = parseInt(document.getElementById('luminositySlider').value);

    // Try WebGL first
    if (initWebGL() && doRecolorSimpleWebGL(width, height, k, oldLab, diffLab, luminosity)) {
        return;
    }

    // CPU fallback
    console.log('Using CPU fallback for simple recolor');
    doRecolorSimpleCPU(width, height, k, oldLab, diffLab, luminosity);
}

// WebGL implementation of simple recolor
function doRecolorSimpleWebGL(width, height, k, oldLab, diffLab, luminosity) {
    try {
        const startTime = performance.now();

        webglCanvas.width = width;
        webglCanvas.height = height;
        gl.viewport(0, 0, width, height);

        gl.useProgram(simpleRecolorProgram);
        setupWebGLBuffers(gl, simpleRecolorProgram);

        // Upload source image as texture
        const texture = createTexture(gl, originalImageData);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(gl.getUniformLocation(simpleRecolorProgram, 'u_image'), 0);

        // Upload palette data as uniform arrays
        const oldLabFlat = [];
        const diffLabFlat = [];
        for (let i = 0; i < 20; i++) {
            if (i < k) {
                oldLabFlat.push(oldLab[i][0], oldLab[i][1], oldLab[i][2]);
                diffLabFlat.push(diffLab[i][0], diffLab[i][1], diffLab[i][2]);
            } else {
                oldLabFlat.push(0, 0, 0);
                diffLabFlat.push(0, 0, 0);
            }
        }

        gl.uniform3fv(gl.getUniformLocation(simpleRecolorProgram, 'u_oldLab'), new Float32Array(oldLabFlat));
        gl.uniform3fv(gl.getUniformLocation(simpleRecolorProgram, 'u_diffLab'), new Float32Array(diffLabFlat));
        gl.uniform1i(gl.getUniformLocation(simpleRecolorProgram, 'u_paletteSize'), k);
        gl.uniform1f(gl.getUniformLocation(simpleRecolorProgram, 'u_blendSharpness'), 2.0);
        gl.uniform1f(gl.getUniformLocation(simpleRecolorProgram, 'u_luminosity'), luminosity);

        // Render
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Read back result
        const pixels = new Uint8Array(width * height * 4);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

        // WebGL has Y flipped, need to flip it back
        const newData = ctx.createImageData(width, height);
        for (let y = 0; y < height; y++) {
            const srcRow = (height - 1 - y) * width * 4;
            const dstRow = y * width * 4;
            for (let x = 0; x < width * 4; x++) {
                newData.data[dstRow + x] = pixels[srcRow + x];
            }
        }

        ctx.putImageData(newData, 0, 0);
        imageData = newData;

        // Cleanup
        gl.deleteTexture(texture);

        const elapsed = performance.now() - startTime;
        console.log(`WebGL simple recolor: ${elapsed.toFixed(1)}ms`);

        updateDisplayCanvas();
        renderRecoloredStrip();
        return true;
    } catch (e) {
        console.warn('WebGL simple recolor failed:', e);
        return false;
    }
}

// CPU fallback for simple recolor
function doRecolorSimpleCPU(width, height, k, oldLab, diffLab, luminosity) {
    const startTime = performance.now();
    const newData = ctx.createImageData(width, height);
    const blendSharpness = 2.0;

    for (let i = 0; i < originalImageData.data.length; i += 4) {
        const r = originalImageData.data[i];
        const g = originalImageData.data[i + 1];
        const b = originalImageData.data[i + 2];

        const pixelLab = RGB2LAB([r, g, b]);

        const distances = [];
        let minDist = Infinity;
        for (let j = 0; j < k; j++) {
            const d2 = Math.pow(pixelLab[0] - oldLab[j][0], 2) +
                      Math.pow(pixelLab[1] - oldLab[j][1], 2) +
                      Math.pow(pixelLab[2] - oldLab[j][2], 2);
            const d = Math.sqrt(d2);
            distances.push(d);
            if (d < minDist) minDist = d;
        }

        let totalWeight = 0;
        const weights = [];
        for (let j = 0; j < k; j++) {
            const relDist = distances[j] / Math.max(minDist, 1);
            const w = Math.exp(-blendSharpness * (relDist - 1));
            weights.push(w);
            totalWeight += w;
        }

        let dL = 0, dA = 0, dB = 0;
        if (totalWeight > 0) {
            for (let j = 0; j < k; j++) {
                const normalizedWeight = weights[j] / totalWeight;
                dL += normalizedWeight * diffLab[j][0];
                dA += normalizedWeight * diffLab[j][1];
                dB += normalizedWeight * diffLab[j][2];
            }
        }

        const newPixelLab = [
            pixelLab[0] + dL,
            pixelLab[1] + dA,
            pixelLab[2] + dB
        ];

        let [newR, newG, newB] = LAB2RGB(newPixelLab);

        if (luminosity !== 0) {
            const factor = 1 + (luminosity / 100);
            newR = Math.max(0, Math.min(255, newR * factor));
            newG = Math.max(0, Math.min(255, newG * factor));
            newB = Math.max(0, Math.min(255, newB * factor));
        }

        newData.data[i] = Math.round(Math.max(0, Math.min(255, newR)));
        newData.data[i + 1] = Math.round(Math.max(0, Math.min(255, newG)));
        newData.data[i + 2] = Math.round(Math.max(0, Math.min(255, newB)));
        newData.data[i + 3] = 255;
    }

    const elapsed = performance.now() - startTime;
    console.log(`CPU simple recolor: ${elapsed.toFixed(1)}ms`);

    ctx.putImageData(newData, 0, 0);
    imageData = newData;
    updateDisplayCanvas();
    renderRecoloredStrip();
}

// RBF (Radial Basis Function) algorithm with grid precomputation (from v2)
// Better for smooth gradients in photos
function doRecolorRBF() {
    const RBF_param_coff = 5;
    const ngrid = 10;
    const width = canvas.width;
    const height = canvas.height;
    const k = originalPalette.length;

    if (k === 0) return;

    // Build LAB grid (this is fast - only 1331 entries)
    const gridSize = (ngrid + 1) ** 3;
    const gridLab = [];
    const step = 255.0 / ngrid;

    for (let b = 0; b <= ngrid; b++) {
        for (let g = 0; g <= ngrid; g++) {
            for (let r = 0; r <= ngrid; r++) {
                gridLab.push(RGB2LAB([r * step, g * step, b * step]));
            }
        }
    }

    // Build origin-to-target mapping based on columns
    const oldLab = originalPalette.map(c => RGB2LAB(c));
    const newLab = [];

    for (let i = 0; i < k; i++) {
        const col = originToColumn[i];
        if (col === 'locked' || col === 'bank' || typeof col !== 'number' || col >= targetPalette.length) {
            newLab.push(oldLab[i]);
        } else if (columnBypass[col]) {
            newLab.push(oldLab[i]);
        } else {
            newLab.push(RGB2LAB(targetPalette[col]));
        }
    }

    const diffLab = oldLab.map((o, i) => [
        newLab[i][0] - o[0],
        newLab[i][1] - o[1],
        newLab[i][2] - o[2]
    ]);

    // Calculate average distance for param
    let totDist = 0, cnt = 0;
    for (let i = 0; i < k; i++) {
        for (let j = i + 1; j < k; j++) {
            const d = Math.sqrt(
                Math.pow(oldLab[i][0] - oldLab[j][0], 2) +
                Math.pow(oldLab[i][1] - oldLab[j][1], 2) +
                Math.pow(oldLab[i][2] - oldLab[j][2], 2)
            );
            totDist += d;
            cnt++;
        }
    }
    const avgDist = cnt > 0 ? totDist / cnt : 1;
    const param = RBF_param_coff / (avgDist * avgDist);

    // Build RBF matrix
    const rbfMatrix = [];
    for (let i = 0; i < k; i++) {
        rbfMatrix[i] = [];
        for (let j = 0; j < k; j++) {
            const d2 = Math.pow(oldLab[i][0] - oldLab[j][0], 2) +
                      Math.pow(oldLab[i][1] - oldLab[j][1], 2) +
                      Math.pow(oldLab[i][2] - oldLab[j][2], 2);
            rbfMatrix[i][j] = Math.exp(-d2 * param);
        }
    }

    const rbfInv = pinv(rbfMatrix);

    // Calculate grid transformation (CPU - fast for 1331 entries)
    const gridRGB = [];
    for (let i = 0; i < gridSize; i++) {
        const vsrc = gridLab[i];

        const tmpD = [];
        for (let j = 0; j < k; j++) {
            const d2 = Math.pow(oldLab[j][0] - vsrc[0], 2) +
                      Math.pow(oldLab[j][1] - vsrc[1], 2) +
                      Math.pow(oldLab[j][2] - vsrc[2], 2);
            tmpD.push(Math.exp(-d2 * param));
        }

        const weights = numeric.dotMV(rbfInv, tmpD);
        let scale = 0;
        for (let j = 0; j < k; j++) {
            scale += Math.max(weights[j], 0);
        }
        if (scale < 1) scale = 1;

        let dL = 0, dA = 0, dB = 0;
        for (let j = 0; j < k; j++) {
            if (weights[j] > 0) {
                dL += weights[j] / scale * diffLab[j][0];
                dA += weights[j] / scale * diffLab[j][1];
                dB += weights[j] / scale * diffLab[j][2];
            }
        }

        const newLabColor = [vsrc[0] + dL, vsrc[1] + dA, vsrc[2] + dB];
        const newRGBColor = LAB2RGB(newLabColor);
        gridRGB.push([
            Math.max(0, Math.min(255, newRGBColor[0])),
            Math.max(0, Math.min(255, newRGBColor[1])),
            Math.max(0, Math.min(255, newRGBColor[2]))
        ]);
    }

    const luminosity = parseInt(document.getElementById('luminositySlider').value);

    // Try WebGL for the per-pixel trilinear interpolation (the slow part)
    if (initWebGL() && doRecolorRBFWebGL(width, height, ngrid, gridRGB, luminosity)) {
        return;
    }

    // CPU fallback
    console.log('Using CPU fallback for RBF recolor');
    doRecolorRBFCPU(width, height, ngrid, gridRGB, luminosity);
}

// WebGL implementation of RBF recolor (trilinear interpolation on GPU)
function doRecolorRBFWebGL(width, height, ngrid, gridRGB, luminosity) {
    try {
        const startTime = performance.now();
        const lutSize = ngrid + 1;

        webglCanvas.width = width;
        webglCanvas.height = height;
        gl.viewport(0, 0, width, height);

        gl.useProgram(rbfRecolorProgram);
        setupWebGLBuffers(gl, rbfRecolorProgram);

        // Upload source image as texture
        const imageTexture = createTexture(gl, originalImageData);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, imageTexture);
        gl.uniform1i(gl.getUniformLocation(rbfRecolorProgram, 'u_image'), 0);

        // Create 3D LUT as 2D texture
        // Layout: lutSize^2 width x lutSize height
        // Position (r,g,b) -> pixel at (r + g*lutSize, b)
        const lutWidth = lutSize * lutSize;
        const lutHeight = lutSize;
        const lutData = new Uint8Array(lutWidth * lutHeight * 4);

        for (let b = 0; b < lutSize; b++) {
            for (let g = 0; g < lutSize; g++) {
                for (let r = 0; r < lutSize; r++) {
                    const srcIdx = b * lutSize * lutSize + g * lutSize + r;
                    const dstX = r + g * lutSize;
                    const dstY = b;
                    const dstIdx = (dstY * lutWidth + dstX) * 4;

                    lutData[dstIdx] = Math.round(gridRGB[srcIdx][0]);
                    lutData[dstIdx + 1] = Math.round(gridRGB[srcIdx][1]);
                    lutData[dstIdx + 2] = Math.round(gridRGB[srcIdx][2]);
                    lutData[dstIdx + 3] = 255;
                }
            }
        }

        const lutTexture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, lutTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, lutWidth, lutHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, lutData);

        gl.uniform1i(gl.getUniformLocation(rbfRecolorProgram, 'u_lut'), 1);
        gl.uniform1f(gl.getUniformLocation(rbfRecolorProgram, 'u_lutSize'), lutSize);
        gl.uniform1f(gl.getUniformLocation(rbfRecolorProgram, 'u_luminosity'), luminosity);

        // Render
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Read back result
        const pixels = new Uint8Array(width * height * 4);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

        // WebGL has Y flipped, need to flip it back
        const newData = ctx.createImageData(width, height);
        for (let y = 0; y < height; y++) {
            const srcRow = (height - 1 - y) * width * 4;
            const dstRow = y * width * 4;
            for (let x = 0; x < width * 4; x++) {
                newData.data[dstRow + x] = pixels[srcRow + x];
            }
        }

        ctx.putImageData(newData, 0, 0);
        imageData = newData;

        // Cleanup
        gl.deleteTexture(imageTexture);
        gl.deleteTexture(lutTexture);

        const elapsed = performance.now() - startTime;
        console.log(`WebGL RBF recolor: ${elapsed.toFixed(1)}ms`);

        updateDisplayCanvas();
        renderRecoloredStrip();
        return true;
    } catch (e) {
        console.warn('WebGL RBF recolor failed:', e);
        return false;
    }
}

// CPU fallback for RBF recolor
function doRecolorRBFCPU(width, height, ngrid, gridRGB, luminosity) {
    const startTime = performance.now();
    const newData = ctx.createImageData(width, height);
    const ntmp = ngrid + 1;
    const ntmpsqr = ntmp * ntmp;

    for (let i = 0; i < originalImageData.data.length; i += 4) {
        const r = originalImageData.data[i];
        const g = originalImageData.data[i + 1];
        const b = originalImageData.data[i + 2];

        let tmpx = r / 255 * ngrid;
        let diff_x = tmpx - Math.floor(tmpx);
        tmpx = Math.floor(tmpx);
        if (tmpx >= ngrid) { tmpx = ngrid - 1; diff_x = 1; }

        let tmpy = g / 255 * ngrid;
        let diff_y = tmpy - Math.floor(tmpy);
        tmpy = Math.floor(tmpy);
        if (tmpy >= ngrid) { tmpy = ngrid - 1; diff_y = 1; }

        let tmpz = b / 255 * ngrid;
        let diff_z = tmpz - Math.floor(tmpz);
        tmpz = Math.floor(tmpz);
        if (tmpz >= ngrid) { tmpz = ngrid - 1; diff_z = 1; }

        const corner = tmpz * ntmpsqr + tmpy * ntmp + tmpx;

        const indices = [
            corner, corner + ntmpsqr, corner + ntmp, corner + ntmp + ntmpsqr,
            corner + 1, corner + ntmpsqr + 1, corner + ntmp + 1, corner + ntmp + ntmpsqr + 1
        ];
        const weights = [
            (1-diff_x)*(1-diff_y)*(1-diff_z),
            (1-diff_x)*(1-diff_y)*diff_z,
            (1-diff_x)*diff_y*(1-diff_z),
            (1-diff_x)*diff_y*diff_z,
            diff_x*(1-diff_y)*(1-diff_z),
            diff_x*(1-diff_y)*diff_z,
            diff_x*diff_y*(1-diff_z),
            diff_x*diff_y*diff_z
        ];

        let newR = 0, newG = 0, newB = 0;
        for (let idx = 0; idx < 8; idx++) {
            newR += gridRGB[indices[idx]][0] * weights[idx];
            newG += gridRGB[indices[idx]][1] * weights[idx];
            newB += gridRGB[indices[idx]][2] * weights[idx];
        }

        if (luminosity !== 0) {
            const factor = 1 + (luminosity / 100);
            newR = Math.max(0, Math.min(255, newR * factor));
            newG = Math.max(0, Math.min(255, newG * factor));
            newB = Math.max(0, Math.min(255, newB * factor));
        }

        newData.data[i] = Math.round(newR);
        newData.data[i + 1] = Math.round(newG);
        newData.data[i + 2] = Math.round(newB);
        newData.data[i + 3] = 255;
    }

    const elapsed = performance.now() - startTime;
    console.log(`CPU RBF recolor: ${elapsed.toFixed(1)}ms`);

    ctx.putImageData(newData, 0, 0);
    imageData = newData;
    updateDisplayCanvas();
    renderRecoloredStrip();
}

// ============================================
// Color Picker Mode
// ============================================

function togglePickerMode() {
    pickerMode = !pickerMode;
    const btn = document.getElementById('pickerToggleBtn');
    const canvasInner = document.getElementById('canvasInner');
    const canvasWrapper = document.getElementById('canvasWrapper');
    const categorySelector = document.getElementById('pickerCategorySelector');

    if (pickerMode) {
        // Preserve picked colors - don't clear them when re-entering picker mode
        // User can use "Clear Selections" button if they want to start over
        shouldKeepPickedMarkers = false; // Allow markers to be managed normally

        btn.classList.add('active');
        canvasInner.classList.add('picking-mode');
        canvasWrapper.classList.add('picking-mode');

        // Rebuild markers for any existing picked colors
        document.querySelectorAll('.picker-marker').forEach(m => m.remove());
        pickedPositions.forEach((_, i) => createMarker(i));

        // Show and populate category selector
        categorySelector.classList.remove('hidden');
        updatePickerCategoryOptions();

        if (pickedColors.length > 0) {
            setStatus('Picker mode: ' + pickedColors.length + ' colors already selected. Click to add more.');
        } else {
            setStatus('Picker mode: click to pick colors');
        }
    } else {
        btn.classList.remove('active');
        canvasInner.classList.remove('picking-mode');
        canvasWrapper.classList.remove('picking-mode');
        categorySelector.classList.add('hidden');
        // Don't clear markers - keep them visible
    }

    updatePickerOverlay();
}

function clearPickerSelections() {
    pickedColors = [];
    pickedPositions = [];
    pickedCategories = [];
    document.querySelectorAll('.picker-marker').forEach(m => m.remove());
    shouldKeepPickedMarkers = false;
    updatePickerOverlay();
    setStatus('Cleared all picked colors');
}

function updatePickerCategoryOptions() {
    const select = document.getElementById('pickerCategorySelect');
    select.innerHTML = '';

    // Add Background first
    const bgOption = document.createElement('option');
    bgOption.value = CATEGORY_BACKGROUND;
    bgOption.textContent = getTargetCategoryLabel(CATEGORY_BACKGROUND);
    if (pickerTargetCategory === CATEGORY_BACKGROUND) bgOption.selected = true;
    select.appendChild(bgOption);

    // Add Locked option
    const lockedOption = document.createElement('option');
    lockedOption.value = CATEGORY_LOCKED;
    lockedOption.textContent = getTargetCategoryLabel(CATEGORY_LOCKED);
    if (pickerTargetCategory === CATEGORY_LOCKED) lockedOption.selected = true;
    select.appendChild(lockedOption);

    // Add Accent 1, 2, 3, etc.
    for (let i = 1; i < targetCount; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = getTargetCategoryLabel(i);
        if (i === pickerTargetCategory) option.selected = true;
        select.appendChild(option);
    }
}

function updatePickerCategory(value) {
    pickerTargetCategory = parseInt(value);
}

function addPickedColor(color, x, y) {
    pickedColors.push(color);
    pickedPositions.push({ x, y, color: [...color] });
    pickedCategories.push(pickerTargetCategory);
    createMarker(pickedColors.length - 1);
    updatePickerOverlay();
    setStatus('Picked ' + getTargetCategoryLabel(pickerTargetCategory) + ': ' + rgbToHex(...color));
}

function updatePickerOverlay() {
    const list = document.getElementById('pickerSwatchesList');
    const applyBtn = document.getElementById('pickerApplyBtn');
    const clearBtn = document.getElementById('pickerClearBtn');

    list.innerHTML = '';

    if (pickedColors.length === 0) {
        list.classList.add('hidden');
        applyBtn.classList.add('hidden');
        clearBtn.classList.add('hidden');
        return;
    }

    list.classList.remove('hidden');
    applyBtn.classList.remove('hidden');
    clearBtn.classList.remove('hidden');
    applyBtn.disabled = pickedColors.length < 1;

    pickedColors.forEach((color, i) => {
        const item = document.createElement('div');
        item.className = 'picker-swatch-item';

        // Category label on the left (B, L, A1, A2, etc.) - CLICKABLE to cycle
        const categoryLabel = document.createElement('div');
        categoryLabel.className = 'picker-swatch-category clickable';
        const cat = pickedCategories[i] !== undefined ? pickedCategories[i] : CATEGORY_BACKGROUND;
        categoryLabel.textContent = getTargetCategoryLabel(cat, true);
        categoryLabel.title = 'Click to change category';
        categoryLabel.onclick = (e) => {
            e.stopPropagation();
            cyclePickedColorCategory(i);
        };

        const colorDiv = document.createElement('div');
        colorDiv.className = 'picker-swatch-color';
        colorDiv.style.background = rgbToHex(...color);

        const infoDiv = document.createElement('div');
        infoDiv.className = 'picker-swatch-info';
        infoDiv.innerHTML = rgbToHex(...color);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'picker-swatch-delete';
        deleteBtn.innerHTML = '×';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            removePickedColor(i);
        };

        item.appendChild(categoryLabel);
        item.appendChild(colorDiv);
        item.appendChild(infoDiv);
        item.appendChild(deleteBtn);
        list.appendChild(item);
    });
}

function cyclePickedColorCategory(index) {
    const currentCat = pickedCategories[index] !== undefined ? pickedCategories[index] : CATEGORY_BACKGROUND;
    const nextCat = getNextCategory(currentCat);
    pickedCategories[index] = nextCat;

    // Update the marker label on the image
    const markers = document.querySelectorAll('.picker-marker');
    const marker = markers[index];
    if (marker) {
        const label = marker.querySelector('.picker-marker-label');
        if (label) {
            label.textContent = getTargetCategoryLabel(nextCat, true);
        }
    }

    updatePickerOverlay();
    setStatus('Changed color ' + (index + 1) + ' to ' + getTargetCategoryLabel(nextCat));
}

function getCanvasCoords(e) {
    // Use whichever canvas is currently visible for positioning
    const visibleCanvas = (displayCanvas && displayCanvas.style.display !== 'none') ? displayCanvas : canvas;
    const rect = visibleCanvas.getBoundingClientRect();
    // Always map to original canvas coordinates
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: Math.floor((e.clientX - rect.left) * scaleX),
        y: Math.floor((e.clientY - rect.top) * scaleY)
    };
}

function createMarker(index) {
    const canvasInner = document.getElementById('canvasInner');
    const pos = pickedPositions[index];

    // Use whichever canvas is currently visible
    const visibleCanvas = (displayCanvas && displayCanvas.style.display !== 'none') ? displayCanvas : canvas;
    const rect = visibleCanvas.getBoundingClientRect();
    // Map original canvas coordinates to display coordinates
    const displayX = (pos.x / canvas.width) * rect.width;
    const displayY = (pos.y / canvas.height) * rect.height;
    
    const marker = document.createElement('div');
    marker.className = 'picker-marker';
    marker.dataset.index = index;
    marker.style.left = displayX + 'px';
    marker.style.top = displayY + 'px';
    marker.style.background = rgbToHex(...pos.color);
    
    const label = document.createElement('div');
    label.className = 'picker-marker-label';
    // Show category label instead of index
    const cat = pickedCategories[index] !== undefined ? pickedCategories[index] : CATEGORY_BACKGROUND;
    label.textContent = getTargetCategoryLabel(cat, true);
    marker.appendChild(label);
    
    const deleteBtn = document.createElement('div');
    deleteBtn.className = 'picker-marker-delete';
    deleteBtn.innerHTML = '×';
    deleteBtn.onclick = (e) => {
        e.stopPropagation();
        removePickedColor(index);
    };
    marker.appendChild(deleteBtn);
    
    marker.addEventListener('mousedown', (e) => {
        if (e.target.closest('.picker-marker-delete')) return;
        e.stopPropagation();
        draggingMarker = index;
        marker.classList.add('dragging');
    });
    
    canvasInner.appendChild(marker);
}

function updateMarkers() {
    // Use whichever canvas is currently visible
    const visibleCanvas = (displayCanvas && displayCanvas.style.display !== 'none') ? displayCanvas : canvas;
    const rect = visibleCanvas.getBoundingClientRect();
    pickedPositions.forEach((pos, i) => {
        const marker = document.querySelector(`.picker-marker[data-index="${i}"]`);
        if (marker) {
            // Map original canvas coordinates to display coordinates
            const displayX = (pos.x / canvas.width) * rect.width;
            const displayY = (pos.y / canvas.height) * rect.height;
            marker.style.left = displayX + 'px';
            marker.style.top = displayY + 'px';
            marker.style.background = rgbToHex(...pos.color);
        }
    });
}

function clearMarkers() {
    if (!shouldKeepPickedMarkers) {
        document.querySelectorAll('.picker-marker').forEach(m => m.remove());
    }
}

function removePickedColor(index) {
    pickedColors.splice(index, 1);
    pickedPositions.splice(index, 1);
    pickedCategories.splice(index, 1);

    // Always remove and rebuild markers to ensure correct indices
    document.querySelectorAll('.picker-marker').forEach(m => m.remove());
    pickedPositions.forEach((_, i) => createMarker(i));

    updatePickerOverlay();

    setStatus('Removed color. ' + pickedColors.length + ' colors selected.');
}

function applyPickedAsOriginal() {
    if (pickedColors.length < 1) {
        setStatus('Pick at least 1 color first');
        return;
    }

    // Group picked colors by category
    // categoryGroups[categoryIndex] = [{color, pickedIndex, position}, ...]
    const categoryGroups = {};
    const lockedColors = [];

    for (let i = 0; i < pickedColors.length; i++) {
        const cat = pickedCategories[i] !== undefined ? pickedCategories[i] : 0;
        if (cat === CATEGORY_LOCKED) {
            lockedColors.push({ color: pickedColors[i], pickedIndex: i, position: pickedPositions[i] });
        } else {
            if (!categoryGroups[cat]) categoryGroups[cat] = [];
            categoryGroups[cat].push({ color: pickedColors[i], pickedIndex: i, position: pickedPositions[i] });
        }
    }

    // Find the highest category index used (determines how many target columns we need)
    const usedCategories = Object.keys(categoryGroups).map(k => parseInt(k)).sort((a, b) => a - b);
    const maxCategory = usedCategories.length > 0 ? Math.max(...usedCategories) : 0;

    // Target count = highest category + 1 (since categories are 0-indexed)
    // But cap at reasonable max
    targetCount = Math.min(Math.max(maxCategory + 1, 1), 10);

    // Build the new origin palette and mapping
    // Origins are ordered by category: all B colors, then all A1 colors, etc.
    const newOriginalPalette = [];
    const newOriginToColumn = [];
    const newPickedIndicesOrder = []; // Track original picked indices for marker rebuilding

    // Add colors for each category (0 = Background, 1 = Accent 1, etc.)
    for (let catIdx = 0; catIdx < targetCount; catIdx++) {
        const colorsInCat = categoryGroups[catIdx] || [];
        for (const item of colorsInCat) {
            newOriginalPalette.push([...item.color]);
            newOriginToColumn.push(catIdx);
            newPickedIndicesOrder.push(item.pickedIndex);
        }
    }

    // Add locked colors to bank
    for (const item of lockedColors) {
        newOriginalPalette.push([...item.color]);
        newOriginToColumn.push('bank');
        newPickedIndicesOrder.push(item.pickedIndex);
    }

    originCount = newOriginalPalette.length;

    // Calculate percentages using closest-match assignment
    const labPicked = newOriginalPalette.map(c => RGB2LAB(c));
    const counts = new Array(originCount).fill(0);
    let totalSamples = 0;

    for (let i = 0; i < originalImageData.data.length; i += 16) {
        const lab = RGB2LAB([
            originalImageData.data[i],
            originalImageData.data[i + 1],
            originalImageData.data[i + 2]
        ]);

        let minDist = Infinity;
        let minIdx = 0;
        for (let j = 0; j < labPicked.length; j++) {
            const dist = Math.pow(lab[0] - labPicked[j][0], 2) +
                         Math.pow(lab[1] - labPicked[j][1], 2) +
                         Math.pow(lab[2] - labPicked[j][2], 2);
            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
            }
        }
        counts[minIdx]++;
        totalSamples++;
    }

    const pcts = counts.map(c => (c / totalSamples) * 100);

    document.getElementById('originCountDisplay').value = originCount;
    document.getElementById('targetCountDisplay').value = targetCount;

    originalPalette = newOriginalPalette;
    colorPercentages = pcts;
    originToColumn = newOriginToColumn;
    columnBypass = []; // Reset bypass states

    // Build target palette - use the first color from each category, or a default
    targetPalette = [];
    for (let catIdx = 0; catIdx < targetCount; catIdx++) {
        const colorsInCat = categoryGroups[catIdx] || [];
        if (colorsInCat.length > 0) {
            // Use first color in that category as the target
            targetPalette.push([...colorsInCat[0].color]);
        } else {
            // No colors picked for this category - use a placeholder gray
            targetPalette.push([128, 128, 128]);
        }
    }

    // Default selectedSlotIndex to first slot
    selectedSlotIndex = 0;

    // Update data attribute for responsive styling
    document.getElementById('columnMappingContainer').setAttribute('data-target-count', targetCount);

    // Keep markers on image - keep picked positions and keep marker mode active
    // Force-remove all existing markers first (regardless of flag), then recreate
    document.querySelectorAll('.picker-marker').forEach(m => m.remove());
    shouldKeepPickedMarkers = true;
    pickedPositions.forEach((_, i) => createMarker(i));

    // Turn off picker mode UI but keep markers visible
    if (pickerMode) {
        pickerMode = false;
        const btn = document.getElementById('pickerToggleBtn');
        btn.classList.remove('active');
        document.getElementById('canvasInner').classList.remove('picking-mode');
        document.getElementById('canvasWrapper').classList.remove('picking-mode');
        document.getElementById('pickerCategorySelector').classList.add('hidden');
    }

    updatePickerOverlay();
    renderColumnMapping();
    updateGradientFromSelectedColor();
    renderThemesSortedByMatch(); // Re-sort themes based on new palette
    autoRecolorImage();

    const lockedCount = lockedColors.length;
    const lockedMsg = lockedCount > 0 ? ` (${lockedCount} locked)` : '';
    setStatus('Applied ' + originCount + ' picked colors grouped by category' + lockedMsg + '.');
}

// ============================================
// Zoom and Pan Functions
// ============================================

function resetZoom() {
    zoomLevel = 1;
    panX = 0;
    panY = 0;
    updateCanvasTransform(true);
    updateZoomDisplay();
}

function setZoomFromSlider(value) {
    zoomLevel = parseInt(value) / 100;
    constrainPan();
    updateCanvasTransform(true);
    updateZoomDisplay();
}

function constrainPan() {
    if (zoomLevel <= 1) {
        panX = 0;
        panY = 0;
        return;
    }

    const wrapper = document.getElementById('canvasWrapper');
    const wrapperRect = wrapper.getBoundingClientRect();

    // Canvas is rendered at zoomed dimensions, so use its actual size
    const visibleCanvas = (displayCanvas && displayCanvas.style.display !== 'none') ? displayCanvas : canvas;
    const canvasWidth = visibleCanvas.offsetWidth || visibleCanvas.width;
    const canvasHeight = visibleCanvas.offsetHeight || visibleCanvas.height;

    // Max pan is half the overflow (since we're centered)
    const maxPanX = Math.max(0, (canvasWidth - wrapperRect.width) / 2);
    const maxPanY = Math.max(0, (canvasHeight - wrapperRect.height) / 2);

    panX = Math.max(-maxPanX, Math.min(maxPanX, panX));
    panY = Math.max(-maxPanY, Math.min(maxPanY, panY));
}

function updateCanvasTransform(zoomChanged = false) {
    const canvasInner = document.getElementById('canvasInner');
    // Canvas-inner is position:absolute at left:50%, top:50%
    // First translate -50%,-50% to center it, then apply pan offset
    canvasInner.style.transform = `translate(calc(-50% + ${panX}px), calc(-50% + ${panY}px))`;

    const wrapper = document.getElementById('canvasWrapper');
    if (zoomLevel > 1) {
        wrapper.classList.add('zoomed', 'can-pan');
    } else {
        wrapper.classList.remove('zoomed', 'can-pan');
    }

    // Only re-render the high-quality display when zoom actually changes, not during panning
    if (zoomChanged) {
        updateDisplayCanvas();
    }
    updateMarkers();
}

// High-quality display canvas rendering
// Renders at base resolution, uses CSS transform for instant zoom, then re-renders at high quality after delay
// Track the locked wrapper height and base dimensions when zoomed
let lockedWrapperHeight = 0;
let baseDisplayWidth = 0;
let baseDisplayHeight = 0;
let lastRenderedZoom = 1;
let zoomRenderTimeout = null;

function updateDisplayCanvas() {
    if (!canvas || !imageData || !useHighQualityDisplay) return;

    const canvasInner = document.getElementById('canvasInner');
    const wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) return;

    const wrapperRect = wrapper.getBoundingClientRect();
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;
    const aspectRatio = imgWidth / imgHeight;

    // Calculate base display size to fit wrapper width (at zoom=1)
    const baseWidth = wrapperRect.width;
    const baseHeight = baseWidth / aspectRatio;

    if (zoomLevel === 1) {
        // At zoom level 1, wrapper height conforms to image aspect ratio
        wrapper.style.height = baseHeight + 'px';
        lockedWrapperHeight = baseHeight;
        baseDisplayWidth = baseWidth;
        baseDisplayHeight = baseHeight;
    } else {
        // When zoomed, keep the wrapper height locked to what it was at zoom=1
        if (lockedWrapperHeight > 0) {
            wrapper.style.height = lockedWrapperHeight + 'px';
        }
    }

    // Use CSS transform for instant zoom feedback
    // Scale relative to what was last rendered
    const cssScale = zoomLevel / lastRenderedZoom;
    displayCanvas.style.transform = `scale(${cssScale})`;
    displayCanvas.style.transformOrigin = 'center center';

    // Debounce the high-quality re-render
    if (zoomRenderTimeout) {
        clearTimeout(zoomRenderTimeout);
    }

    zoomRenderTimeout = setTimeout(() => {
        renderAtCurrentZoom();
    }, 150);

    // Show display canvas, hide original
    if (!displayCanvas.parentNode) {
        canvasInner.insertBefore(displayCanvas, canvas);
    }
    displayCanvas.style.display = 'block';
    canvas.style.display = 'none';
}

// Render the canvas at the current zoom level for high quality
function renderAtCurrentZoom() {
    if (!canvas || !imageData) return;

    const wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) return;

    const wrapperRect = wrapper.getBoundingClientRect();
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;
    const aspectRatio = imgWidth / imgHeight;

    const baseWidth = wrapperRect.width;
    const baseHeight = baseWidth / aspectRatio;

    // Calculate display size (what we want to show on screen)
    const displayWidth = Math.round(baseWidth * zoomLevel);
    const displayHeight = Math.round(baseHeight * zoomLevel);

    // Render at 2x resolution for sharpness (like retina), but cap at original image size
    const scaleFactor = Math.min(2, imgWidth / displayWidth, imgHeight / displayHeight);
    const renderWidth = Math.round(displayWidth * Math.max(1, scaleFactor));
    const renderHeight = Math.round(displayHeight * Math.max(1, scaleFactor));

    // Set canvas pixel buffer to high resolution
    displayCanvas.width = renderWidth;
    displayCanvas.height = renderHeight;

    // Use high-quality smoothing
    displayCtx.imageSmoothingEnabled = true;
    displayCtx.imageSmoothingQuality = 'high';
    displayCtx.drawImage(canvas, 0, 0, renderWidth, renderHeight);

    // Use CSS to set display size (smaller than pixel buffer = sharper)
    displayCanvas.style.width = displayWidth + 'px';
    displayCanvas.style.height = displayHeight + 'px';

    // Reset CSS transform since we've rendered at full resolution
    displayCanvas.style.transform = 'scale(1)';
    lastRenderedZoom = zoomLevel;
}

function updateZoomDisplay() {
    const pct = Math.round(zoomLevel * 100);
    document.getElementById('zoomValueDisplay').textContent = pct + '%';
    document.getElementById('zoomSlider').value = pct;
}

// ============================================
// Resize Handles
// ============================================

let isResizing = false;
let resizeType = null;
let resizeStartX = 0;
let resizeStartY = 0;
let resizeStartWidth = 0;
let resizeStartHeight = 0;

function initResizeHandles() {
    const canvasArea = document.getElementById('canvasArea');
    const resizeRight = document.getElementById('resizeRight');
    const resizeBottom = document.getElementById('resizeBottom');
    const resizeCorner = document.getElementById('resizeCorner');

    if (!resizeRight || !resizeBottom || !resizeCorner) return;

    const startResize = (type) => (e) => {
        isResizing = true;
        resizeType = type;
        resizeStartX = e.clientX;
        resizeStartY = e.clientY;
        resizeStartWidth = canvasArea.offsetWidth;
        resizeStartHeight = canvasArea.offsetHeight;
        document.body.style.cursor = type === 'corner' ? 'nwse-resize' : (type === 'right' ? 'ew-resize' : 'ns-resize');
        e.target.classList.add('active');
    };

    resizeRight.addEventListener('mousedown', startResize('right'));
    resizeBottom.addEventListener('mousedown', startResize('bottom'));
    resizeCorner.addEventListener('mousedown', startResize('corner'));

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const dx = e.clientX - resizeStartX;
        const dy = e.clientY - resizeStartY;

        if (resizeType === 'right' || resizeType === 'corner') {
            const newWidth = Math.max(300, resizeStartWidth + dx);
            canvasArea.style.width = newWidth + 'px';
            canvasArea.style.flex = 'none';
        }

        // Only allow height adjustment when zoomed in
        // At zoom=1, height conforms to image aspect ratio automatically
        if ((resizeType === 'bottom' || resizeType === 'corner') && zoomLevel > 1) {
            const newHeight = Math.max(200, resizeStartHeight + dy);
            canvasArea.style.height = newHeight + 'px';
        }
        // Don't update display during drag - too slow
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizeType = null;
            document.body.style.cursor = '';
            document.querySelectorAll('.resize-handle').forEach(h => h.classList.remove('active'));
            // Re-render after resize is complete
            updateDisplayCanvas();
            updateMarkers();
        }
    });

    // Also update markers and display canvas on window resize
    window.addEventListener('resize', () => {
        updateDisplayCanvas();
        updateMarkers();
    });
}

function autoSizeCanvasArea() {
    // Auto-size the canvas area to fit the image when not zoomed
    if (!canvas || !originalImageData || zoomLevel > 1) return;

    const canvasArea = document.getElementById('canvasArea');
    const wrapper = document.getElementById('canvasWrapper');

    // Get the natural dimensions of the image
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;

    // Calculate aspect ratio
    const aspectRatio = imgWidth / imgHeight;

    // Get available space
    const maxWidth = window.innerWidth - 550; // Leave room for sidebar
    const maxHeight = window.innerHeight - 200; // Leave room for header/controls

    // Calculate best fit dimensions
    let fitWidth = maxWidth;
    let fitHeight = fitWidth / aspectRatio;

    if (fitHeight > maxHeight) {
        fitHeight = maxHeight;
        fitWidth = fitHeight * aspectRatio;
    }

    // Apply to canvas area (minimum constraints apply via CSS)
    canvasArea.style.width = Math.max(300, fitWidth) + 'px';
    canvasArea.style.height = 'auto';
    canvasArea.style.flex = 'none';
}

// ============================================
// Theme and Import Functions
// ============================================

// Calculate how well a theme matches the image's extracted palette
function calculateThemeMatchScore(themeKey) {
    if (originalPalette.length === 0) return 0;
    
    const themeColors = THEMES[themeKey].map(hex => hexToRgb(hex));
    const themeLab = themeColors.map(c => RGB2LAB(c));
    const imageLab = originalPalette.map(c => RGB2LAB(c));
    
    // For each image color, find the closest theme color
    let totalScore = 0;
    for (let i = 0; i < imageLab.length; i++) {
        let minDist = Infinity;
        for (let j = 0; j < themeLab.length; j++) {
            const dist = Math.sqrt(
                Math.pow(imageLab[i][0] - themeLab[j][0], 2) +
                Math.pow(imageLab[i][1] - themeLab[j][1], 2) +
                Math.pow(imageLab[i][2] - themeLab[j][2], 2)
            );
            if (dist < minDist) minDist = dist;
        }
        // Weight by color percentage in image
        const weight = colorPercentages[i] || 1;
        // Convert distance to a 0-100 score (closer = higher)
        // LAB distance of 0 = 100%, distance of 100+ = 0%
        const colorScore = Math.max(0, 100 - minDist);
        totalScore += colorScore * weight;
    }
    
    // Normalize by total percentage
    const totalWeight = colorPercentages.slice(0, imageLab.length).reduce((a, b) => a + b, 0) || 1;
    return totalScore / totalWeight;
}

// Render themes sorted by match score
function renderThemesSortedByMatch() {
    const themeContainer = document.getElementById('themeScrollContainer');
    
    // Calculate scores for all themes
    const themeScores = Object.keys(THEMES).map(themeKey => ({
        key: themeKey,
        score: calculateThemeMatchScore(themeKey)
    }));
    
    // Sort by score descending
    themeScores.sort((a, b) => b.score - a.score);
    
    // Clear and rebuild
    themeContainer.innerHTML = '';
    
    themeScores.forEach(({ key: themeKey, score }) => {
        const item = document.createElement('div');
        item.className = 'theme-item' + (themeKey === selectedTheme ? ' selected' : '');
        item.dataset.theme = themeKey;
        item.onclick = () => selectTheme(themeKey);
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'theme-item-name';
        nameSpan.textContent = THEME_NAMES[themeKey] || themeKey;
        item.appendChild(nameSpan);
        
        // Add match score badge
        if (originalPalette.length > 0) {
            const matchBadge = document.createElement('span');
            matchBadge.className = 'theme-item-match';
            const roundedScore = Math.round(score);
            matchBadge.textContent = roundedScore + '%';
            if (roundedScore >= 70) {
                matchBadge.classList.add('high');
            } else if (roundedScore >= 50) {
                matchBadge.classList.add('medium');
            } else {
                matchBadge.classList.add('low');
            }
            item.appendChild(matchBadge);
        }
        
        const swatchesDiv = document.createElement('div');
        swatchesDiv.className = 'theme-item-swatches';
        THEMES[themeKey].forEach(color => {
            const swatch = document.createElement('div');
            swatch.className = 'theme-item-swatch';
            swatch.style.background = color;
            swatchesDiv.appendChild(swatch);
        });
        item.appendChild(swatchesDiv);
        
        themeContainer.appendChild(item);
    });
}

function selectTheme(themeKey) {
    selectedTheme = themeKey;
    document.querySelectorAll('.theme-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.theme === themeKey);
    });
}

function applyTheme() {
    if (!selectedTheme || !THEMES[selectedTheme]) {
        setStatus('Select a theme first');
        return;
    }

    const colors = THEMES[selectedTheme].map(hex => hexToRgb(hex));
    for (let i = 0; i < Math.min(colors.length, targetPalette.length); i++) {
        targetPalette[i] = colors[i];
    }

    renderColumnMapping();
    autoRecolorImage();
    setStatus('Applied theme: ' + (THEME_NAMES[selectedTheme] || selectedTheme));
}

function extractHexColors(input) {
    const hexRegex = /color:\s*#([0-9A-Fa-f]{6})/g;
    const matches = [...input.matchAll(hexRegex)];
    
    if (matches.length > 0) {
        const seen = new Set();
        const colors = [];
        for (const m of matches) {
            const hex = m[1].toUpperCase();
            if (!seen.has(hex)) {
                seen.add(hex);
                colors.push(hexToRgb(hex));
            }
        }
        return colors;
    }
    
    const fallbackRegex = /#?([0-9A-Fa-f]{6})/g;
    const fallbackMatches = [...input.matchAll(fallbackRegex)];
    const seen = new Set();
    const colors = [];
    for (const m of fallbackMatches) {
        const hex = m[1].toUpperCase();
        if (!seen.has(hex)) {
            seen.add(hex);
            colors.push(hexToRgb(hex));
        }
    }
    return colors;
}

function previewAdobeColors(input) {
    const colors = extractHexColors(input);
    const preview = document.getElementById('importPreview');
    const container = document.getElementById('importColors');
    
    if (colors.length === 0) {
        preview.classList.add('hidden');
        return;
    }
    
    preview.classList.remove('hidden');
    container.innerHTML = '';
    
    colors.forEach(rgb => {
        const chip = document.createElement('div');
        chip.className = 'import-color-chip';
        chip.style.background = rgbToHex(...rgb);
        chip.title = rgbToHex(...rgb);
        container.appendChild(chip);
    });
}

function importAdobePalette() {
    const input = document.getElementById('adobeInput').value.trim();
    const colors = extractHexColors(input);

    if (colors.length === 0) {
        setStatus('No colors found. Paste CSS from Adobe Color.');
        return;
    }

    for (let i = 0; i < Math.min(colors.length, targetPalette.length); i++) {
        targetPalette[i] = colors[i];
    }

    renderColumnMapping();
    autoRecolorImage();
    setStatus(`Imported ${Math.min(colors.length, targetPalette.length)} colors from Adobe Color`);
}

// ============================================
// Harmony Functions
// ============================================

function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;
    
    if (max === min) {
        h = s = 0;
    } else {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
            case g: h = ((b - r) / d + 2) / 6; break;
            case b: h = ((r - g) / d + 4) / 6; break;
        }
    }
    return [h * 360, s * 100, l * 100];
}

function hslToRgb(h, s, l) {
    h /= 360; s /= 100; l /= 100;
    let r, g, b;
    
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function harmonizePalette() {
    if (targetPalette.length === 0) {
        setStatus('Load an image first');
        return;
    }
    
    const baseColor = targetPalette[selectedSlotIndex];
    const [h, s, l] = rgbToHsl(...baseColor);
    const harmonyType = document.getElementById('harmonyType').value;
    
    let newHues = [];
    const n = targetPalette.length;
    
    switch (harmonyType) {
        case 'complementary':
            newHues = Array(n).fill(0).map((_, i) => {
                return i % 2 === 0 ? h : (h + 180) % 360;
            });
            break;
        case 'analogous':
            const analogousSpread = 30;
            newHues = Array(n).fill(0).map((_, i) => {
                const offset = (i - Math.floor(n/2)) * analogousSpread;
                return (h + offset + 360) % 360;
            });
            break;
        case 'triadic':
            newHues = Array(n).fill(0).map((_, i) => {
                return (h + (i % 3) * 120) % 360;
            });
            break;
        case 'split':
            newHues = Array(n).fill(0).map((_, i) => {
                const angles = [0, 150, 210];
                return (h + angles[i % 3]) % 360;
            });
            break;
        case 'tetradic':
            newHues = Array(n).fill(0).map((_, i) => {
                return (h + (i % 4) * 90) % 360;
            });
            break;
    }
    
    // Check for locked origins - don't change targets that only have locked origins
    const lockedTargets = new Set();
    for (let i = 0; i < originCount; i++) {
        if (originToColumn[i] === 'locked') {
            // This origin is locked, doesn't affect any target
        }
    }
    
    targetPalette.forEach((color, i) => {
        const [_, origS, origL] = rgbToHsl(...color);
        const newRgb = hslToRgb(newHues[i], origS, origL);
        targetPalette[i] = newRgb;
    });
    
    renderColumnMapping();
    setStatus('Applied ' + harmonyType + ' harmony');
    updateHarmonyWheel();
}

let harmonyDragging = null;

function updateHarmonyWheel() {
    const canvas = document.getElementById('harmonyWheelCanvas');
    if (!canvas) return;
    
    const ctx2 = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const outerRadius = Math.min(centerX, centerY) - 5;
    const innerRadius = outerRadius * 0.4;
    
    ctx2.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw color wheel
    for (let angle = 0; angle < 360; angle++) {
        const startAngle = (angle - 90) * Math.PI / 180;
        const endAngle = (angle - 89) * Math.PI / 180;
        
        ctx2.beginPath();
        ctx2.moveTo(
            centerX + innerRadius * Math.cos(startAngle),
            centerY + innerRadius * Math.sin(startAngle)
        );
        ctx2.arc(centerX, centerY, outerRadius, startAngle, endAngle);
        ctx2.arc(centerX, centerY, innerRadius, endAngle, startAngle, true);
        ctx2.closePath();
        
        const [r, g, b] = hslToRgb(angle, 80, 50);
        ctx2.fillStyle = `rgb(${r},${g},${b})`;
        ctx2.fill();
    }
    
    // Draw center circle
    ctx2.beginPath();
    ctx2.arc(centerX, centerY, innerRadius - 2, 0, Math.PI * 2);
    ctx2.fillStyle = '#1a1a1a';
    ctx2.fill();
    
    // Draw lines from center to each target color's hue
    if (targetPalette.length > 0) {
        const markerDistance = (outerRadius + innerRadius) / 2;
        ctx2.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx2.lineWidth = 1;
        targetPalette.forEach((color) => {
            const [h] = rgbToHsl(...color);
            const angle = (h - 90) * Math.PI / 180;
            ctx2.beginPath();
            ctx2.moveTo(centerX, centerY);
            ctx2.lineTo(
                centerX + markerDistance * Math.cos(angle),
                centerY + markerDistance * Math.sin(angle)
            );
            ctx2.stroke();
        });
    }
    
    // Draw selected color in center
    if (targetPalette.length > 0) {
        ctx2.beginPath();
        ctx2.arc(centerX, centerY, innerRadius - 8, 0, Math.PI * 2);
        ctx2.fillStyle = rgbToHex(...targetPalette[selectedSlotIndex]);
        ctx2.fill();
        ctx2.strokeStyle = '#fff';
        ctx2.lineWidth = 2;
        ctx2.stroke();
    }
    
    // Create/update draggable dots for each target color
    updateHarmonyDots();
}

function updateHarmonyDots() {
    const container = document.querySelector('.harmony-wheel-container');
    const canvas = document.getElementById('harmonyWheelCanvas');
    if (!container || !canvas) return;
    
    // Remove existing dots
    container.querySelectorAll('.harmony-dot').forEach(d => d.remove());
    
    if (targetPalette.length === 0) return;
    
    const rect = canvas.getBoundingClientRect();
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const outerRadius = Math.min(centerX, centerY) - 5;
    const innerRadius = outerRadius * 0.4;
    const markerDistance = (outerRadius + innerRadius) / 2;
    
    targetPalette.forEach((color, i) => {
        const [h, s, l] = rgbToHsl(...color);
        const angle = (h - 90) * Math.PI / 180;
        const x = centerX + markerDistance * Math.cos(angle);
        const y = centerY + markerDistance * Math.sin(angle);
        
        const dot = document.createElement('div');
        dot.className = 'harmony-dot' + (i === selectedSlotIndex ? ' selected' : '');
        dot.style.left = (canvas.offsetLeft + x) + 'px';
        dot.style.top = (canvas.offsetTop + y) + 'px';
        dot.style.background = rgbToHex(...color);
        dot.dataset.index = i;
        dot.title = `Target ${i + 1} - Drag to change hue`;
        
        dot.addEventListener('mousedown', (e) => {
            e.preventDefault();
            harmonyDragging = i;
            dot.classList.add('dragging');
            selectSlot(i);
        });
        
        container.appendChild(dot);
    });
}

// Global mouse handlers for harmony dot dragging
document.addEventListener('mousemove', (e) => {
    if (harmonyDragging === null) return;
    
    const canvas = document.getElementById('harmonyWheelCanvas');
    const container = document.querySelector('.harmony-wheel-container');
    if (!canvas || !container) return;
    
    const rect = canvas.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Calculate angle from center to mouse
    const dx = e.clientX - centerX;
    const dy = e.clientY - centerY;
    let angle = Math.atan2(dy, dx) * 180 / Math.PI + 90;
    if (angle < 0) angle += 360;
    
    // Update target color with new hue, preserving saturation and lightness
    const color = targetPalette[harmonyDragging];
    const [_, s, l] = rgbToHsl(...color);
    const newRgb = hslToRgb(angle, s, l);
    targetPalette[harmonyDragging] = newRgb;
    
    // Update UI
    updateHarmonyWheel();
    renderColumnMapping();
});

document.addEventListener('mouseup', () => {
    if (harmonyDragging !== null) {
        document.querySelectorAll('.harmony-dot').forEach(d => d.classList.remove('dragging'));
        harmonyDragging = null;
        autoRecolorImage();
    }
});

// ============================================
// Utility Functions
// ============================================

function resetImage() {
    if (!originalImageData) return;
    ctx.putImageData(originalImageData, 0, 0);
    imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    updateDisplayCanvas();
    setStatus('Image reset to original');
}

function removeImage() {
    // Clear all image data
    imageData = null;
    originalImageData = null;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = 300;
    canvas.height = 200;

    // Reset palettes
    originalPalette = [];
    targetPalette = [];
    colorPercentages = [];
    fullColorDistribution = [];
    originToColumn = [];
    columnBypass = [];

    // Reset picker state
    pickedColors = [];
    pickedPositions = [];
    pickedCategories = [];
    pickerMode = false;
    shouldKeepPickedMarkers = false;
    document.querySelectorAll('.picker-marker').forEach(m => m.remove());

    // Reset counts
    originCount = 5;
    targetCount = 5;
    document.getElementById('originCountDisplay').value = 5;
    document.getElementById('targetCountDisplay').value = 5;

    // Reset zoom
    zoomLevel = 1;
    panX = 0;
    panY = 0;
    updateCanvasTransform(true);

    // Hide UI elements
    document.getElementById('pickerOverlay').classList.add('hidden');
    document.getElementById('zoomControls').classList.add('hidden');
    document.getElementById('toleranceContainer').classList.add('hidden');
    document.getElementById('originOverflowBank').classList.remove('visible');

    // Clear mapping display
    document.getElementById('mappingColumns').innerHTML = '';
    document.getElementById('mappingTargetsRow').innerHTML = '';
    document.getElementById('colorStrip').innerHTML = '';

    // Reset file input so the same file can be re-selected
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';

    setStatus('Image removed. Upload a new image to start.');
}

function downloadImage() {
    if (!imageData) {
        setStatus('No image to download');
        return;
    }
    
    const link = document.createElement('a');
    link.download = 'recolored-image.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
    setStatus('Image downloaded');
}

function showLoading() {
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

function setStatus(message) {
    document.getElementById('statusPanel').textContent = message;
}

// ============================================
// Algorithm Toggle Functions
// ============================================

function setAlgorithm(algo) {
    selectedAlgorithm = algo;
    updateAlgorithmUI();
    if (originalImageData) {
        autoRecolorImage();
    }
}

function toggleAlgorithm() {
    selectedAlgorithm = selectedAlgorithm === 'simple' ? 'rbf' : 'simple';
    updateAlgorithmUI();
    if (originalImageData) {
        autoRecolorImage();
    }
}

function updateAlgorithmUI() {
    const switchEl = document.getElementById('algoSwitch');
    const labelSimple = document.getElementById('algoLabelSimple');
    const labelRBF = document.getElementById('algoLabelRBF');
    
    if (selectedAlgorithm === 'rbf') {
        switchEl.classList.add('rbf');
        labelRBF.classList.add('active');
        labelSimple.classList.remove('active');
    } else {
        switchEl.classList.remove('rbf');
        labelSimple.classList.add('active');
        labelRBF.classList.remove('active');
    }
    
    setStatus('Algorithm: ' + (selectedAlgorithm === 'rbf' ? 'RBF (smooth gradients)' : 'Simple (faster)'));
}

// Initialize algorithm UI on page load
document.addEventListener('DOMContentLoaded', function() {
    updateAlgorithmUI();
});

function updateLuminosityValue(value) {
    document.getElementById('luminosityValue').textContent = value;
}

// Debug function - call from console: debugMapping()
window.debugMapping = function() {
    console.log('=== Debug Mapping ===');
    console.log('originCount:', originCount);
    console.log('targetCount:', targetCount);
    console.log('originToColumn:', JSON.stringify(originToColumn));
    console.log('originalPalette length:', originalPalette.length);
    console.log('targetPalette length:', targetPalette.length);
    console.log('targetPalette:', targetPalette.map(c => rgbToHex(...c)));

    // Show which origins map to which targets
    for (let i = 0; i < originCount; i++) {
        const col = originToColumn[i];
        const originColor = rgbToHex(...originalPalette[i]);
        if (col === 'bank' || col === 'locked') {
            console.log(`  Origin ${i} (${originColor}) -> ${col} (no change)`);
        } else if (typeof col === 'number' && col < targetPalette.length) {
            console.log(`  Origin ${i} (${originColor}) -> Target ${col} (${rgbToHex(...targetPalette[col])})`);
        } else {
            console.log(`  Origin ${i} (${originColor}) -> INVALID col=${col}`);
        }
    }
    return {originToColumn, originCount, targetCount, originalPalette, targetPalette};
};

// ============================================
// Configuration Save/Load System
// ============================================

let savedConfigs = [];
let configCounter = 0;

function saveCurrentConfig() {
    if (!originalImageData) {
        setStatus('Load an image first before saving config');
        return;
    }

    configCounter++;
    const config = {
        id: Date.now(),
        name: `Config ${configCounter}`,
        timestamp: new Date().toISOString(),
        originCount,
        targetCount,
        originalPalette: originalPalette.map(c => [...c]),
        targetPalette: targetPalette.map(c => [...c]),
        colorPercentages: [...colorPercentages],
        originToColumn: [...originToColumn],
        algorithm: selectedAlgorithm,
        luminosity: parseInt(document.getElementById('luminositySlider').value)
    };

    savedConfigs.push(config);
    renderConfigList();
    setStatus(`Saved configuration: ${config.name}`);
}

function loadConfig(configId) {
    const config = savedConfigs.find(c => c.id === configId);
    if (!config) {
        setStatus('Configuration not found');
        return;
    }

    originCount = config.originCount;
    targetCount = config.targetCount;
    originalPalette = config.originalPalette.map(c => [...c]);
    targetPalette = config.targetPalette.map(c => [...c]);
    colorPercentages = [...config.colorPercentages];
    originToColumn = [...config.originToColumn];
    selectedAlgorithm = config.algorithm || 'simple';
    document.getElementById('luminositySlider').value = config.luminosity || 0;
    document.getElementById('luminosityValue').textContent = config.luminosity || 0;

    document.getElementById('originCountDisplay').value = originCount;
    document.getElementById('targetCountDisplay').value = targetCount;
    document.getElementById('columnMappingContainer').setAttribute('data-target-count', targetCount);

    updateAlgorithmUI();
    renderColumnMapping();
    autoRecolorImage();
    renderConfigList();
    setStatus(`Loaded configuration: ${config.name}`);
}

function deleteConfig(configId) {
    savedConfigs = savedConfigs.filter(c => c.id !== configId);
    renderConfigList();
    setStatus('Configuration deleted');
}

function renderConfigList() {
    const list = document.getElementById('configList');
    if (!list) return;

    list.innerHTML = '';

    savedConfigs.forEach((config, index) => {
        const item = document.createElement('div');
        item.className = 'config-item';
        item.onclick = () => loadConfig(config.id);

        // Color swatches preview
        const swatches = document.createElement('div');
        swatches.className = 'config-item-swatches';
        config.targetPalette.slice(0, 5).forEach(color => {
            const swatch = document.createElement('div');
            swatch.className = 'config-item-swatch';
            swatch.style.background = rgbToHex(...color);
            swatches.appendChild(swatch);
        });

        // Info
        const info = document.createElement('div');
        info.className = 'config-item-info';
        const name = document.createElement('div');
        name.className = 'config-item-name';
        name.textContent = config.name;
        const details = document.createElement('div');
        details.textContent = `${config.originCount} origins → ${config.targetCount} targets`;
        info.appendChild(name);
        info.appendChild(details);

        // Delete button
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'config-item-delete';
        deleteBtn.innerHTML = '×';
        deleteBtn.title = 'Delete this configuration';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deleteConfig(config.id);
        };

        item.appendChild(swatches);
        item.appendChild(info);
        item.appendChild(deleteBtn);
        list.appendChild(item);
    });
}

function exportAllConfigs() {
    if (savedConfigs.length === 0) {
        setStatus('No configurations to export');
        return;
    }

    const exportData = {
        version: '18',
        exportDate: new Date().toISOString(),
        configs: savedConfigs
    };

    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `palette-configs-${new Date().toISOString().split('T')[0]}.json`;
    link.click();

    URL.revokeObjectURL(url);
    setStatus(`Exported ${savedConfigs.length} configurations`);
}

function importConfigFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);

            if (data.configs && Array.isArray(data.configs)) {
                // Merge imported configs with existing ones
                const existingIds = new Set(savedConfigs.map(c => c.id));
                let importedCount = 0;

                data.configs.forEach(config => {
                    // Generate new ID if there's a conflict
                    if (existingIds.has(config.id)) {
                        config.id = Date.now() + Math.random();
                    }
                    savedConfigs.push(config);
                    importedCount++;
                });

                // Update config counter
                configCounter = Math.max(configCounter, savedConfigs.length);

                renderConfigList();
                setStatus(`Imported ${importedCount} configurations`);
            } else {
                setStatus('Invalid config file format');
            }
        } catch (err) {
            setStatus('Error reading config file: ' + err.message);
        }
    };
    reader.readAsText(file);

    // Reset file input
    event.target.value = '';
}
