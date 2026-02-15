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

// Self-test numeric.js library on load
(function() {
    if (typeof numeric === 'undefined') {
        console.error('[numeric self-test] numeric library not loaded at all!');
        return;
    }
    var missing = [];
    ['svd', 'dot', 'transpose', 'diag', 'clone', 'rep', 'epsilon'].forEach(function(fn) {
        if (typeof numeric[fn] === 'undefined') missing.push(fn);
    });
    if (missing.length > 0) {
        console.error('[numeric self-test] MISSING functions: ' + missing.join(', '));
        console.error('[numeric self-test] Total keys: ' + Object.keys(numeric).length);
    } else {
        // Quick SVD smoke test with 2x2 identity
        try {
            var result = numeric.svd([[1, 0], [0, 1]]);
            if (!result || !result.S || result.S.length !== 2) {
                console.error('[numeric self-test] SVD returned unexpected result:', result);
            } else {
                console.log('[numeric self-test] OK — svd, dot, transpose, diag, clone, rep all present. SVD smoke test passed.');
            }
        } catch (e) {
            console.error('[numeric self-test] SVD smoke test FAILED:', e.message || e);
        }
    }
})();

function pinv(A) {
    if (typeof numeric === 'undefined') throw new Error('numeric library not loaded');
    if (typeof numeric.svd !== 'function') {
        var allKeys = Object.keys(numeric);
        debugLog('numeric type: ' + typeof numeric, 'error');
        debugLog('numeric.version: ' + (numeric.version || 'undefined'), 'error');
        debugLog('total keys: ' + allKeys.length, 'error');
        debugLog('all keys: ' + allKeys.join(', '), 'error');
        debugLog('numeric.svd type: ' + typeof numeric.svd, 'error');
        debugLog('has dot: ' + (typeof numeric.dot), 'error');
        debugLog('has transpose: ' + (typeof numeric.transpose), 'error');
        debugLog('has epsilon: ' + (typeof numeric.epsilon), 'error');
        throw new Error('numeric.svd is ' + typeof numeric.svd + '. Library has ' + allKeys.length + ' keys (expected ~120+).');
    }
    // Validate matrix before passing to SVD
    debugLog('pinv: matrix ' + A.length + 'x' + A[0].length + ', numeric.svd type=' + typeof numeric.svd);
    var z;
    try {
        z = numeric.svd(A);
    } catch (svdErr) {
        debugLog('numeric.svd() internal error: ' + (svdErr.message || svdErr), 'error');
        debugLog('numeric.svd type at call time: ' + typeof numeric.svd, 'error');
        debugLog('numeric.clone type: ' + typeof numeric.clone, 'error');
        debugLog('numeric.rep type: ' + typeof numeric.rep, 'error');
        debugLog('numeric.epsilon: ' + numeric.epsilon, 'error');
        debugLog('Matrix sample [0][0..2]: ' + (A[0] ? A[0].slice(0, 3).join(', ') : 'empty'), 'error');
        // Check for NaN/Infinity in matrix
        var hasNaN = false, hasInf = false;
        for (var ri = 0; ri < A.length && !hasNaN && !hasInf; ri++) {
            for (var ci = 0; ci < A[ri].length; ci++) {
                if (isNaN(A[ri][ci])) { hasNaN = true; break; }
                if (!isFinite(A[ri][ci])) { hasInf = true; break; }
            }
        }
        if (hasNaN) debugLog('Matrix contains NaN values!', 'error');
        if (hasInf) debugLog('Matrix contains Infinity values!', 'error');
        throw svdErr;
    }
    var foo = z.S[0];
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
let _webglImageTexture = null; // Persistent source image texture (avoids re-upload per frame)
let _webglDirty = true;        // True when CPU-side imageData is stale and needs sync
let _lastWebGLRenderType = null; // 'simple' or 'rbf' — tracks what's currently on the WebGL canvas
let _lastSimpleUniforms = null;  // Cache last simple recolor uniforms for re-render on zoom
let _lastRBFUniforms = null;     // Cache last RBF recolor uniforms for re-render on zoom
let _simpleUniformLocs = null;   // Cached uniform locations for simple shader
let _rbfUniformLocs = null;      // Cached uniform locations for RBF shader

// Vertex shader (shared by both programs) - simple fullscreen quad
// Tex coords in setupWebGLBuffers already handle Y orientation, so pass through directly.
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

    // Bypass correction: palette colors whose columns are locked/bypassed.
    // After the LUT lookup, pixels close to a bypassed color are blended back
    // toward their original value so the global RBF transform cannot bleed into them.
    uniform vec3 u_bypassedRGB[20];
    uniform int  u_bypassedCount;

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

        // Bypass correction: blend back toward original for pixels near bypassed colors.
        if (u_bypassedCount > 0) {
            float maxKeep = 0.0;
            for (int j = 0; j < 20; j++) {
                if (j >= u_bypassedCount) break;
                vec3 diff = texColor.rgb - u_bypassedRGB[j];
                float d2 = dot(diff, diff);
                // Gaussian falloff: sigma ~0.12 in 0-1 space (~30 in 0-255 space)
                float keep = exp(-d2 / (2.0 * 0.12 * 0.12));
                maxKeep = max(maxKeep, keep);
            }
            newRgb = mix(newRgb, texColor.rgb, maxKeep);
        }

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
        // Create a dedicated WebGL canvas (separate from 2D displayCanvas)
        webglCanvas = document.getElementById('webglDisplayCanvas') || document.createElement('canvas');
        webglCanvas.id = 'webglDisplayCanvas';
        gl = webglCanvas.getContext('webgl', { preserveDrawingBuffer: true, alpha: false }) ||
             webglCanvas.getContext('experimental-webgl', { preserveDrawingBuffer: true, alpha: false });

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

        // Cache uniform locations (avoids repeated lookups on every render)
        _simpleUniformLocs = {
            u_image: gl.getUniformLocation(simpleRecolorProgram, 'u_image'),
            u_oldLab: gl.getUniformLocation(simpleRecolorProgram, 'u_oldLab'),
            u_diffLab: gl.getUniformLocation(simpleRecolorProgram, 'u_diffLab'),
            u_paletteSize: gl.getUniformLocation(simpleRecolorProgram, 'u_paletteSize'),
            u_blendSharpness: gl.getUniformLocation(simpleRecolorProgram, 'u_blendSharpness'),
            u_luminosity: gl.getUniformLocation(simpleRecolorProgram, 'u_luminosity')
        };
        _rbfUniformLocs = {
            u_image: gl.getUniformLocation(rbfRecolorProgram, 'u_image'),
            u_lut: gl.getUniformLocation(rbfRecolorProgram, 'u_lut'),
            u_lutSize: gl.getUniformLocation(rbfRecolorProgram, 'u_lutSize'),
            u_luminosity: gl.getUniformLocation(rbfRecolorProgram, 'u_luminosity'),
            u_bypassedRGB: gl.getUniformLocation(rbfRecolorProgram, 'u_bypassedRGB[0]'),
            u_bypassedCount: gl.getUniformLocation(rbfRecolorProgram, 'u_bypassedCount')
        };

        webglInitialized = true;
        console.log('WebGL initialized successfully (direct display mode)');
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
        const errMsg = gl.getProgramInfoLog(program);
        console.error('Program link error:', errMsg);
        debugLog(`[shader-link-error] ${errMsg}`, 'error');
        return null;
    }

    return program;
}

function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const errMsg = gl.getShaderInfoLog(shader);
        const shaderKind = (type === gl.VERTEX_SHADER) ? 'vertex' : 'fragment';
        console.error('Shader compile error (' + shaderKind + '):', errMsg);
        debugLog(`[shader-compile-error] ${shaderKind}: ${errMsg}`, 'error');
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

let _webglPositionBuffer = null;
let _webglTexCoordBuffer = null;

function setupWebGLBuffers(gl, program) {
    // Create fullscreen quad buffers once, reuse on subsequent calls
    if (!_webglPositionBuffer) {
        const positions = new Float32Array([
            -1, -1,  1, -1,  -1, 1,
            -1,  1,  1, -1,   1, 1
        ]);
        _webglPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, _webglPositionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    }
    if (!_webglTexCoordBuffer) {
        const texCoords = new Float32Array([
            0, 1,  1, 1,  0, 0,
            0, 0,  1, 1,  1, 0
        ]);
        _webglTexCoordBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, _webglTexCoordBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    }

    // Bind position buffer and set attribute pointer
    gl.bindBuffer(gl.ARRAY_BUFFER, _webglPositionBuffer);
    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    // Bind texcoord buffer and set attribute pointer
    gl.bindBuffer(gl.ARRAY_BUFFER, _webglTexCoordBuffer);
    const texCoordLoc = gl.getAttribLocation(program, 'a_texCoord');
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);
}

function createTexture(gl, imageData, useLinear) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // Source image uses LINEAR for smooth display at various resolutions.
    // LUT textures use NEAREST to avoid interpolation artifacts in lookup tables.
    const filter = useLinear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, imageData.width, imageData.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, imageData.data);
    return texture;
}

// Ensure the persistent source image texture is uploaded (reuse across frames)
function ensureImageTexture() {
    if (!_webglImageTexture && originalImageData && gl) {
        _webglImageTexture = createTexture(gl, originalImageData, true);
    }
    return _webglImageTexture;
}

// Invalidate persistent image texture (call when image changes)
function invalidateImageTexture() {
    if (_webglImageTexture && gl) {
        gl.deleteTexture(_webglImageTexture);
    }
    _webglImageTexture = null;
}

// Lazy sync: read WebGL pixels back to CPU-side imageData and 2D canvas.
// Only called when something needs CPU pixel access (export, strip, luminosity).
// Since renderWebGLToDisplay() always renders at full image resolution,
// the framebuffer already contains the full-res result (preserveDrawingBuffer: true).
// We just readPixels directly — no re-render needed.
function syncWebGLToCPU() {
    if (!_webglDirty || !gl || !webglCanvas) return;
    const width = canvas.width;
    const height = canvas.height;
    debugLog(`[sync-webgl-cpu] readback ${width}x${height}, dirty=${_webglDirty}`);

    // Safety check: if webglCanvas dimensions don't match image dimensions
    // (shouldn't happen now that we always render at full res, but just in case),
    // force a re-render at correct resolution.
    if (webglCanvas.width !== width || webglCanvas.height !== height) {
        console.warn('syncWebGLToCPU: canvas dimensions mismatch, re-rendering at full res');
        renderWebGLToDisplay(_lastWebGLRenderType);
    }

    // Read pixels back — readPixels always returns bottom-to-top row order,
    // so we must flip Y rows to get correct top-to-bottom orientation for the 2D canvas.
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Flip rows: swap row i with row (height-1-i)
    const rowSize = width * 4;
    const tempRow = new Uint8Array(rowSize);
    for (let y = 0; y < Math.floor(height / 2); y++) {
        const topOffset = y * rowSize;
        const bottomOffset = (height - 1 - y) * rowSize;
        tempRow.set(pixels.subarray(topOffset, topOffset + rowSize));
        pixels.copyWithin(topOffset, bottomOffset, bottomOffset + rowSize);
        pixels.set(tempRow, bottomOffset);
    }

    const newData = ctx.createImageData(width, height);
    newData.data.set(pixels);
    ctx.putImageData(newData, 0, 0);
    imageData = newData;

    _webglDirty = false;

    // No need to re-render — the framebuffer is preserved (preserveDrawingBuffer: true)
    // and is still at full image resolution. CSS sizing is unchanged.

    console.log('syncWebGLToCPU: pixels synced to CPU');
}

// ============================================
// Global State
// ============================================

let imageData = null;
let originalImageData = null;
let originalPalette = [];
let targetPalette = [];
let selectedSlotIndex = 0;
let harmonyMode = 'harmony'; // 'harmony' or 'color'
let _currentHarmonyType = 'complementary'; // cached harmony type, survives DOM hidden/show resets
let lockColorSelectedIdx = 0; // Which target is selected in Lock a Color mode
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
let rawColorDistribution = []; // Original k-means colors before tolerance merging

// Column-based mapping: originToColumn[originIndex] = columnIndex (or 'locked' or 'bank')
let originToColumn = [];

// Per-column bypass (lock) state - when true, origins in that column keep their original colors
let columnBypass = [];

// Per-target opacity (0-100): controls how much of the target color shows vs original
let targetOpacity = {};

// Per-origin opacity (0-100): controls individual origin recolor strength
let originOpacity = {};

// Opacity recolor cache — avoids full pipeline re-run when only opacity changes
let _opacityCache = null; // { oldLab, fullTargetLab, k, algorithm }
let _opacityRafPending = false;

// Live preview toggle - when false, autoRecolorImage does nothing
let livePreviewEnabled = false;

// Picked color markers stay visible
let shouldKeepPickedMarkers = false;

// Selected theme tracking
let selectedTheme = '';

// Algorithm selection: 'simple' or 'rbf'
let selectedAlgorithm = 'simple';

// Progressive UI stage: 'initial' | 'image-loaded' | 'colors-picked' | 'target-selection' | 'complete'
let uiStage = 'initial';

let instructionsCollapsed = false;

// Beginner/Advanced mode state
let appMode = 'beginner'; // 'beginner' or 'advanced'

function setAppMode(mode) {
    appMode = mode;
    const isAdvanced = mode === 'advanced';

    // Update toggle UI
    document.getElementById('modeSwitch').classList.toggle('advanced', isAdvanced);
    document.getElementById('modeLabelBeginner').classList.toggle('active', !isAdvanced);
    document.getElementById('modeLabelAdvanced').classList.toggle('active', isAdvanced);

    // Set body class for CSS-driven feature hiding
    document.body.classList.toggle('mode-beginner', !isAdvanced);
    document.body.classList.toggle('mode-advanced', isAdvanced);

    // Advanced mode: default instructions collapsed at every stage
    if (isAdvanced) {
        const container = document.getElementById('progressiveInstructions');
        const arrow = document.getElementById('instructionsToggleArrow');
        if (container && !container.classList.contains('collapsed')) {
            container.classList.add('collapsed');
            instructionsCollapsed = true;
            if (arrow) arrow.textContent = '▸';
        }
        // Collapse tutorial by default in advanced mode
        if (typeof tutorialHasBeenShown !== 'undefined' && tutorialHasBeenShown && !tutorialCollapsed) {
            const tutorialOverlay = document.getElementById('tutorialOverlay');
            if (tutorialOverlay && !tutorialOverlay.classList.contains('hidden')) {
                collapseTutorial();
            }
        }
    } else {
        // Beginner mode: re-expand instructions
        const container = document.getElementById('progressiveInstructions');
        const arrow = document.getElementById('instructionsToggleArrow');
        if (container && container.classList.contains('collapsed')) {
            container.classList.remove('collapsed');
            instructionsCollapsed = false;
            if (arrow) arrow.textContent = '▾';
        }
    }



    // Update instruction text for target-selection step based on mode
    updateInstructionTextForMode();

    // Update palette icon visibility based on Direct Picker state
    updatePaletteIconVisibility();
}

function toggleAppMode() {
    setAppMode(appMode === 'advanced' ? 'beginner' : 'advanced');
}

// Update instruction step text based on mode
function updateInstructionTextForMode() {
    const stepText = document.getElementById('targetStepText');
    const stepNum = document.getElementById('targetStepNum');
    if (!stepText) return;

    if (appMode === 'beginner') {
        stepText.innerHTML = 'Time to choose your new colors — there\'s LOTS of ways to do this! The preview updates live as you make changes.';
    } else {
        stepText.innerHTML = 'Time to choose your new colors — there\'s LOTS of ways to do this! Hit <span class="btn-highlight">Apply this recolor</span> when you\'re done, or just turn on <span class="btn-highlight">Recolor Live</span> to have the preview update whenever you make a change.';
    }
}

// Handle Direct Color Picker section toggle (beginner mode)
function handleDirectPickerToggle(detailsEl) {
    if (!detailsEl) return;
    updatePaletteIconVisibility();
}

// Update palette icon visibility based on Direct Picker state
function updatePaletteIconVisibility() {
    const directPicker = document.getElementById('directPickerSection');
    const isOpen = directPicker && directPicker.open;
    const isBeginner = appMode === 'beginner';

    // Get all palette (color picker) buttons under target swatches
    document.querySelectorAll('.swatch-buttons').forEach(btnRow => {
        const pickerBtn = btnRow.querySelector('button:first-child');
        if (!pickerBtn) return;

        if (isBeginner) {
            if (isOpen) {
                pickerBtn.classList.remove('palette-icon-hidden');
                pickerBtn.classList.add('palette-icon-active');
            } else {
                pickerBtn.classList.add('palette-icon-hidden');
                pickerBtn.classList.remove('palette-icon-active');
            }
        } else {
            // Advanced mode: always visible, normal style
            pickerBtn.classList.remove('palette-icon-hidden');
            pickerBtn.classList.remove('palette-icon-active');
        }
    });
}

// Activate harmony mini-tool (show quick harmony overlay on the image)
function activateHarmonyMiniTool() {
    const quickHarmonyBar = document.getElementById('quickHarmonyBar');
    const qhPanel = document.getElementById('quickHarmonyPanel');
    const qhTab = document.getElementById('quickHarmonyTab');
    if (!quickHarmonyBar) return;

    quickHarmonyBar.classList.remove('hidden');
    if (qhPanel) qhPanel.classList.remove('hidden');
    if (qhTab) qhTab.classList.add('hidden');

    // Scroll so the canvas-wrapper is vertically centered in the viewport
    const canvasWrapper = document.getElementById('canvasWrapper');
    if (canvasWrapper) {
        const rect = canvasWrapper.getBoundingClientRect();
        const wrapperMidY = rect.top + rect.height / 2;
        const viewportMidY = window.innerHeight / 2;
        window.scrollBy({ top: wrapperMidY - viewportMidY, behavior: 'smooth' });
    }
}

function toggleInstructions() {
    instructionsCollapsed = !instructionsCollapsed;
    const container = document.getElementById('progressiveInstructions');
    const arrow = document.getElementById('instructionsToggleArrow');
    if (container) container.classList.toggle('collapsed', instructionsCollapsed);
    if (arrow) arrow.textContent = instructionsCollapsed ? '▸' : '▾';
}

function updateProgressiveInstructions(step) {
    const container = document.getElementById('progressiveInstructions');
    if (!container) return;
    container.querySelectorAll('.instruction-step').forEach(el => {
        const elStep = el.dataset.step;
        // When at target-selection or complete, show both step 4 and step 5/6
        if (step === 'target-selection' || step === 'complete') {
            el.classList.toggle('active', elStep === 'target-selection' || elStep === 'complete');
        } else {
            el.classList.toggle('active', elStep === step);
        }
    });
    // Advanced mode: auto-collapse instructions at every stage change
    if (appMode === 'advanced') {
        const arrow = document.getElementById('instructionsToggleArrow');
        if (!container.classList.contains('collapsed')) {
            container.classList.add('collapsed');
            instructionsCollapsed = true;
            if (arrow) arrow.textContent = '▸';
        }
    }
}

// Palette counts
let originCount = 5;
let targetCount = 5;

// Color tolerance for extraction merging (0 = no merging, higher = more merging)
let colorTolerance = 0;
const DEFAULT_COLOR_TOLERANCE = 0;

// Zoom and pan state (managed by Panzoom)
let zoomLevel = 1;
let panX = 0, panY = 0;  // kept for compatibility reads, synced from Panzoom
let panzoomInstance = null;
let draggingMarker = null;
let altHeld = false;
let draggedOriginIndex = null;
let draggedTargetIndex = null;    // For target swatch drag-and-drop
let targetConflict = null;        // {col: colIdx} if two targets in same slot

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
// Sticky Overlay Positioning
// ============================================

// Keeps left-side tools (zoom slider + harmony overlay) and right-side picker overlay
// centered in the visible portion of the canvas wrapper as the page scrolls.
function updateStickyOverlays() {
    const wrapper = document.getElementById('canvasWrapper');
    const toolsColumn = document.getElementById('canvasToolsColumn');
    const pickerOverlay = document.getElementById('pickerOverlay');
    if (!wrapper) return;

    const wrapperRect = wrapper.getBoundingClientRect();
    const viewportHeight = window.innerHeight;

    const visibleTop = Math.max(wrapperRect.top, 0);
    const visibleBottom = Math.min(wrapperRect.bottom, viewportHeight);
    const visibleHeight = visibleBottom - visibleTop;

    if (visibleHeight <= 0) return;

    const visibleCenterInWrapper = (visibleTop - wrapperRect.top + visibleBottom - wrapperRect.top) / 2;
    const wrapperH = wrapperRect.height;

    // Clamp so overlays (centered via translateY(-50%)) don't overflow the wrapper edges
    function clampCenter(el) {
        if (!el) return visibleCenterInWrapper;
        const elH = el.getBoundingClientRect().height || 0;
        const halfH = elH / 2;
        return Math.max(halfH, Math.min(wrapperH - halfH, visibleCenterInWrapper));
    }

    // Position tools column (zoom slider + harmony overlay) at visible center
    if (toolsColumn) {
        toolsColumn.style.top = clampCenter(toolsColumn) + 'px';
    }
    // Position picker overlay at visible center
    if (pickerOverlay) {
        pickerOverlay.style.top = clampCenter(pickerOverlay) + 'px';
    }
}

// ============================================
// Initialization
// ============================================

// Disable browser's automatic scroll restoration on reload
if ('scrollRestoration' in history) {
    history.scrollRestoration = 'manual';
}

document.addEventListener('DOMContentLoaded', function() {
    // Scroll to top on page load/reload (immediate + deferred to override any browser restore)
    window.scrollTo(0, 0);
    requestAnimationFrame(() => window.scrollTo(0, 0));

    // Initialize mode toggle UI
    setAppMode('beginner');

    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Create display canvas for 2D fallback rendering (used when WebGL is inactive)
    displayCanvas = document.createElement('canvas');
    displayCanvas.id = 'displayCanvas';
    displayCtx = displayCanvas.getContext('2d', { alpha: false });
    displayCtx.imageSmoothingEnabled = true;
    displayCtx.imageSmoothingQuality = 'high';

    // WebGL display canvas is created by initWebGL() on first recolor

    // Set initial workspace state - centered with no sidebar
    const workspace = document.getElementById('workspace');
    workspace.classList.add('centered-initial');
    uiStage = 'initial';
    updateProgressiveInstructions('initial');

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
        // Turn Apply button orange to indicate unapplied change
        const applyBtn = document.getElementById('toleranceApplyBtn');
        if (applyBtn) applyBtn.classList.add('tolerance-dirty');
    });

    // Sticky overlay event listeners (function defined at file scope)
    window.addEventListener('scroll', updateStickyOverlays, { passive: true });
    window.addEventListener('resize', updateStickyOverlays, { passive: true });

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

    // Ctrl toggle to temporarily hide/show picker overlay & zoom controls (peek underneath)
    let ctrlOverlaysHidden = false;

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Control' && !e.repeat) {
            ctrlOverlaysHidden = !ctrlOverlaysHidden;
            const pickerOverlay = document.getElementById('pickerOverlay');
            const zoomControls = document.getElementById('zoomControls');
            const zoomSlider = document.getElementById('zoomSliderContainer');
            const quickHarmony = document.getElementById('quickHarmonyBar');
            const tutorialOverlayEl = document.getElementById('tutorialOverlay');
            const tutorialTabEl = document.getElementById('tutorialTab');

            if (ctrlOverlaysHidden) {
                if (pickerOverlay && !pickerOverlay.classList.contains('hidden') && !pickerOverlay.dataset.tutorialHidden) {
                    pickerOverlay.style.opacity = '0';
                    pickerOverlay.style.pointerEvents = 'none';
                }
                if (zoomControls && !zoomControls.classList.contains('hidden')) {
                    zoomControls.style.opacity = '0';
                    zoomControls.style.pointerEvents = 'none';
                }
                if (zoomSlider && !zoomSlider.classList.contains('hidden')) {
                    zoomSlider.style.opacity = '0';
                    zoomSlider.style.pointerEvents = 'none';
                }
                if (quickHarmony && !quickHarmony.classList.contains('hidden')) {
                    quickHarmony.style.opacity = '0';
                    quickHarmony.style.pointerEvents = 'none';
                }
                // Temporarily hide tutorial panel (not collapse — no traditional picker shown)
                // Tutorial is in normal flow now, so use display:none to reclaim space
                if (tutorialOverlayEl && !tutorialOverlayEl.classList.contains('hidden')) {
                    tutorialOverlayEl.style.display = 'none';
                    tutorialCtrlHidden = true;
                }
                if (tutorialTabEl && !tutorialTabEl.classList.contains('hidden')) {
                    tutorialTabEl.style.display = 'none';
                }
            } else {
                if (pickerOverlay && !pickerOverlay.dataset.tutorialHidden) {
                    pickerOverlay.style.opacity = '';
                    pickerOverlay.style.pointerEvents = '';
                }
                if (zoomControls) { zoomControls.style.opacity = ''; zoomControls.style.pointerEvents = ''; }
                if (zoomSlider) { zoomSlider.style.opacity = ''; zoomSlider.style.pointerEvents = ''; }
                if (quickHarmony) { quickHarmony.style.opacity = ''; quickHarmony.style.pointerEvents = ''; }
                // Restore tutorial panel
                if (tutorialOverlayEl) {
                    tutorialOverlayEl.style.display = '';
                    tutorialCtrlHidden = false;
                }
                if (tutorialTabEl) {
                    tutorialTabEl.style.display = '';
                }
            }
        }
    });

    // Double-tap Shift to advance to the next color category (during color picking)
    let _lastShiftUpTime = 0;
    const DOUBLE_TAP_THRESHOLD = 400; // ms

    document.addEventListener('keyup', function(e) {
        if (e.key === 'Shift' && !e.repeat && pickerMode) {
            const now = Date.now();
            if (now - _lastShiftUpTime < DOUBLE_TAP_THRESHOLD) {
                _lastShiftUpTime = 0; // Reset so triple-tap doesn't fire again
                advancePickerCategory();
            } else {
                _lastShiftUpTime = now;
            }
        }
    });

    const canvasInner = document.getElementById('canvasInner');
    const canvasWrapper = document.getElementById('canvasWrapper');

    // Initialize Panzoom on canvasInner — ONLY for panning (translate).
    // Zoom is handled by us directly via canvas re-render + CSS sizing.
    // Panzoom scale is always locked at 1.
    panzoomInstance = Panzoom(canvasInner, {
        maxScale: 1,
        minScale: 1,
        startScale: 1,
        startX: 0,
        startY: 0,
        cursor: 'default',
        disablePan: true,   // We enable/disable per Alt+drag session
        disableZoom: true,  // We never use Panzoom zoom
        noBind: true,
        pinchAndPan: false,
    });

    // Ensure clean initial state
    panzoomInstance.pan(0, 0, { animate: false, force: true });

    // Sync pan position from Panzoom (scale is always 1 — we manage zoom ourselves)
    function syncFromPanzoom() {
        const pan = panzoomInstance.getPan();
        panX = pan.x;
        panY = pan.y;
    }

    // Update markers during pan, and clamp pan bounds
    canvasInner.addEventListener('panzoomchange', function() {
        syncFromPanzoom();
        constrainPan();
        updateMarkers();
    });

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

    // Pan functionality — Alt+drag when zoomed
    let isPanning = false;
    canvasWrapper.addEventListener('mousedown', function(e) {
        if (e.target.closest('.picker-marker') || e.target.closest('.zoom-slider-container') ||
            e.target.closest('.zoom-controls') || e.target.closest('.picker-overlay')) return;
        if (zoomLevel <= 1 && !isVerticallyCropped) return;
        if (pickerMode && !altHeld) return;
        if (!altHeld && pickerMode) return;

        // Enable pan for this drag session
        panzoomInstance.setOptions({ disablePan: false });
        isPanning = true;
        canvasWrapper.classList.add('panning');
        // Trigger Panzoom's pan start
        panzoomInstance.handleDown(e);
    });

    document.addEventListener('mousemove', function(e) {
        if (isPanning) {
            panzoomInstance.handleMove(e);
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

    document.addEventListener('mouseup', function(e) {
        if (isPanning) {
            panzoomInstance.handleUp(e);
            panzoomInstance.setOptions({ disablePan: true });
            isPanning = false;
            canvasWrapper.classList.remove('panning');
        }
        if (draggingMarker !== null) {
            const marker = document.querySelector(`.picker-marker[data-index="${draggingMarker}"]`);
            if (marker) marker.classList.remove('dragging');
            draggingMarker = null;
        }
    });

    // Alt+scroll to zoom — we manage zoom ourselves (canvas re-render),
    // Panzoom only handles pan (translate).
    // Zoom-to-cursor: keeps the point under the mouse fixed on screen.
    canvasWrapper.addEventListener('wheel', function(e) {
        if (e.altKey) {
            e.preventDefault();

            const wrapper = document.getElementById('canvasWrapper');
            const oldZoom = zoomLevel;

            // Calculate new zoom from scroll delta (gentle 7% per tick)
            const delta = e.deltaY > 0 ? -0.07 : 0.07;
            let newZoom = Math.max(1, Math.min(4, zoomLevel * (1 + delta)));

            // Snap to exactly 1x if we're very close — avoids partial-zoom limbo
            if (newZoom < 1.02) newZoom = 1;

            // Ensure wrapper height is locked and layout is in zoomed mode
            // BEFORE any pan/zoom math (do this once, on first zoom above 1x)
            if (!wrapper.classList.contains('zoomed') && newZoom > 1) {
                const rect = wrapper.getBoundingClientRect();
                wrapper.style.height = rect.height + 'px';
                wrapper.classList.add('zoomed', 'can-pan');
            }

            debugLog(`[zoom-wheel] ${oldZoom.toFixed(2)} → ${newZoom.toFixed(2)}`);

            if (newZoom <= 1) {
                // At 1x — reset pan but DON'T tear down zoomed layout yet.
                // Layout teardown is deferred so zooming back in is seamless.
                zoomLevel = 1;
                panX = 0;
                panY = 0;
                panzoomInstance.pan(0, 0, { animate: false, force: true });
            } else {
                // Zoom-to-cursor math:
                // Mouse position relative to wrapper top-left
                const wrapperRect = wrapper.getBoundingClientRect();
                const mouseX = e.clientX - wrapperRect.left;
                const mouseY = e.clientY - wrapperRect.top;

                // Current pan from Panzoom
                const currentPan = panzoomInstance.getPan();

                // Keep the content point under the cursor fixed:
                // newPan = mousePos - (mousePos - oldPan) * (newZoom / oldZoom)
                const ratio = newZoom / oldZoom;
                const newPanX = mouseX - (mouseX - currentPan.x) * ratio;
                const newPanY = mouseY - (mouseY - currentPan.y) * ratio;

                zoomLevel = newZoom;
                panX = newPanX;
                panY = newPanY;
                panzoomInstance.pan(newPanX, newPanY, { animate: false, force: true });
            }

            updateDisplayCanvas();
            constrainPan();
            updateMarkers();
            updateZoomDisplay();
            updateStickyOverlays();

            // Deferred 1x layout cleanup: if we're at 1x, schedule the layout
            // teardown for after scrolling stops (avoids layout thrashing during
            // continuous scroll-through-1x).
            if (zoomLevel <= 1) {
                if (window._zoomResetTimeout) clearTimeout(window._zoomResetTimeout);
                window._zoomResetTimeout = setTimeout(() => {
                    if (zoomLevel <= 1) {
                        if (zoomRenderTimeout) { clearTimeout(zoomRenderTimeout); zoomRenderTimeout = null; }
                        if (!isVerticallyCropped) {
                            wrapper.style.height = '';
                            wrapper.classList.remove('zoomed', 'can-pan');
                        }
                        renderAtCurrentZoom();
                        updateMarkers();
                    }
                }, 200);
            } else {
                // Cancel any pending 1x cleanup if we zoomed back above 1
                if (window._zoomResetTimeout) { clearTimeout(window._zoomResetTimeout); window._zoomResetTimeout = null; }
            }
        }
    }, { passive: false });
});

// ============================================
// File Handling
// ============================================

let originalFileName = 'image';  // Track original file name for download naming

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Extract filename without extension for download naming
    const name = file.name || 'image';
    const dotIdx = name.lastIndexOf('.');
    originalFileName = dotIdx > 0 ? name.substring(0, dotIdx) : name;

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

    // Swap to short logo
    const logoTall = document.getElementById('logoTall');
    const logoShort = document.getElementById('logoShort');
    if (logoTall) logoTall.style.display = 'none';
    if (logoShort) logoShort.style.display = '';

    // Progressive UI: transition from centered to image-loaded layout
    const workspace = document.getElementById('workspace');
    workspace.classList.remove('centered-initial');
    workspace.classList.add('image-loaded');
    // Show sidebar and Color Analysis panel
    document.getElementById('sidebar').classList.remove('hidden');
    document.getElementById('colorAnalysisPanel').classList.remove('hidden');
    document.getElementById('colorAnalysisOptionalPanel').classList.remove('hidden');
    // Show the Reset/Remove/Download buttons and resize handles
    document.getElementById('imageButtonGroup').classList.remove('hidden');
    document.querySelectorAll('.resize-handle').forEach(h => h.classList.remove('hidden'));
    uiStage = 'image-loaded';
    updateProgressiveInstructions('image-loaded');

    // Reset Panzoom to zoom=1, pan=0,0
    zoomLevel = 1;
    panX = 0;
    panY = 0;
    if (panzoomInstance) {
        panzoomInstance.reset({ animate: false });
    }
    const wrapper = document.getElementById('canvasWrapper');
    wrapper.style.height = '';
    wrapper.classList.remove('zoomed', 'can-pan');
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
    debugLog(`[image-load] ${width}x${height}, file=${originalFileName || 'unknown'}`);

    canvas.width = width;
    canvas.height = height;

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0, width, height);

    imageData = ctx.getImageData(0, 0, width, height);
    originalImageData = ctx.getImageData(0, 0, width, height);

    // Invalidate WebGL image texture and render state for new image
    invalidateImageTexture();
    _lastWebGLRenderType = null;
    _lastSimpleUniforms = null;
    _lastRBFUniforms = null;
    _webglDirty = false;

    // Auto-size the canvas area to fit the image
    autoSizeCanvasArea();

    // Set aspect-ratio on canvas wrapper so it naturally conforms to image proportions
    wrapper.style.aspectRatio = `${width} / ${height}`;
    wrapper.style.height = ''; // Clear any leftover manual height
    wrapper.style.minHeight = ''; // Let aspect-ratio control height

    // Update high-quality display
    updateDisplayCanvas();

    // Re-center overlays after layout change
    requestAnimationFrame(updateStickyOverlays);

    extractPalette();
    setStatus('Image loaded. Palette extracted.');
}

function extractPalette() {
    showLoading();

    setTimeout(() => {
        const colors = extractColorsKMeans(originalImageData, 20);
        rawColorDistribution = colors.map(c => ({ color: [...c.color], pct: c.pct })); // Store original
        fullColorDistribution = colors;

        // === OLD AUTO-ASSIGN CODE (commented out - picking is now mandatory) ===
        // originalPalette = colors.slice(0, originCount).map(c => [...c.color]);
        // colorPercentages = colors.slice(0, originCount).map(c => c.pct);
        // targetPalette = originalPalette.map(c => [...c]);
        //
        // // Initialize column mapping: each origin maps to its corresponding target column
        // originToColumn = [];
        // columnBypass = []; // Reset bypass states
        // for (let i = 0; i < originCount; i++) {
        //     if (i < targetCount) {
        //         originToColumn[i] = i;
        //     } else {
        //         originToColumn[i] = 'bank'; // Extra origins go to bank
        //     }
        // }
        // shouldKeepPickedMarkers = false;
        // renderColumnMapping();
        // updateHarmonyWheel();
        // updateGradientFromSelectedColor();
        // renderThemesSortedByMatch(); // Sort themes by match to extracted palette
        // === END OLD AUTO-ASSIGN CODE ===

        // Only render the distribution strip (extraction still needed for strip + Add Color dropdown)
        renderColorStrip();
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

    // Pre-compute LAB values for all colors
    const labColors = colors.map(c => RGB2LAB(c.color));

    const merged = [];
    const used = new Set();

    for (let i = 0; i < colors.length; i++) {
        if (used.has(i)) continue;

        const baseLab = labColors[i];
        let totalPct = colors[i].pct;
        // Accumulate weighted LAB sums for perceptually accurate averaging
        let labSum = [baseLab[0] * colors[i].pct, baseLab[1] * colors[i].pct, baseLab[2] * colors[i].pct];
        let count = 1;

        // Find all similar colors
        for (let j = i + 1; j < colors.length; j++) {
            if (used.has(j)) continue;

            const testLab = labColors[j];
            const dist = Math.sqrt(
                Math.pow(baseLab[0] - testLab[0], 2) +
                Math.pow(baseLab[1] - testLab[1], 2) +
                Math.pow(baseLab[2] - testLab[2], 2)
            );

            if (dist <= labThreshold) {
                used.add(j);
                totalPct += colors[j].pct;
                // Weighted sum in LAB space (perceptually uniform)
                labSum[0] += testLab[0] * colors[j].pct;
                labSum[1] += testLab[1] * colors[j].pct;
                labSum[2] += testLab[2] * colors[j].pct;
                count++;
            }
        }

        // Calculate weighted average color in LAB space, then convert back to RGB
        if (count > 1) {
            const avgLab = [labSum[0] / totalPct, labSum[1] / totalPct, labSum[2] / totalPct];
            const avgRgb = LAB2RGB(avgLab);
            const avgColor = [
                Math.round(Math.max(0, Math.min(255, avgRgb[0]))),
                Math.round(Math.max(0, Math.min(255, avgRgb[1]))),
                Math.round(Math.max(0, Math.min(255, avgRgb[2])))
            ];
            merged.push({ color: avgColor, pct: totalPct });
        } else {
            merged.push({ color: [...colors[i].color], pct: totalPct });
        }

        used.add(i);
    }

    // Re-sort by percentage
    merged.sort((a, b) => b.pct - a.pct);

    return merged;
}

// Reset tolerance to default and apply
function resetTolerance() {
    colorTolerance = DEFAULT_COLOR_TOLERANCE;
    document.getElementById('toleranceSlider').value = colorTolerance;
    document.getElementById('toleranceValue').textContent = colorTolerance;
    reExtractWithTolerance();
}

// Re-extract palette with current tolerance setting
function reExtractWithTolerance() {
    if (!originalImageData) {
        setStatus('Load an image first');
        return;
    }

    if (rawColorDistribution.length === 0) {
        setStatus('No raw color data. Reload image first.');
        return;
    }

    showLoading();
    setStatus('Re-extracting with tolerance...');

    setTimeout(() => {
        // Get current tolerance from slider
        colorTolerance = parseInt(document.getElementById('toleranceSlider').value);

        // Start from the raw k-means colors and apply tolerance merging
        const mergedColors = mergeColorsWithTolerance(
            rawColorDistribution.map(c => ({ color: [...c.color], pct: c.pct })),
            colorTolerance
        );

        // Update fullColorDistribution for the analysis strips only.
        // Origins/targets/column mapping are owned by the picker — don't touch them.
        fullColorDistribution = mergedColors;

        renderColorStrip();
        renderThemesSortedByMatch();
        hideLoading();

        // Remove orange dirty indicator from Apply button
        const applyBtn = document.getElementById('toleranceApplyBtn');
        if (applyBtn) applyBtn.classList.remove('tolerance-dirty');

        const colorCount = mergedColors.length;
        setStatus(`Re-extracted ${colorCount} colors with tolerance ${colorTolerance}`);
    }, 50);
}

// Render color strip with optional descale
function renderColorStrip() {
    const strip = document.getElementById('colorStrip');
    strip.innerHTML = '';

    const descaleEnabled = document.getElementById('descaleBgCheckbox').checked;

    // Filter out tiny colors FIRST so their percentages don't leave a gap
    let displayData = fullColorDistribution.filter(item => item.pct >= 0.5);

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
        // Normalize so visible segments sum to 100%
        const total = displayData.reduce((sum, item) => sum + item.pct, 0);
        displayData = displayData.map(item => ({
            ...item,
            displayPct: total > 0 ? (item.pct / total) * 100 : 0
        }));
    }

    for (const item of displayData) {
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

    // Skip if any target is still blank (null)
    if (targetPalette.some(c => c === null)) {
        return;
    }

    // Build effective palette: for bypassed columns, use original colors instead of targets
    const effectivePalette = targetPalette.map((color, i) => {
        if (columnBypass[i]) {
            // Find the first origin mapped to this column and use its original color
            for (let j = 0; j < originToColumn.length; j++) {
                if (originToColumn[j] === i && originalPalette[j]) {
                    return [...originalPalette[j]];
                }
            }
            // Fallback: use the target color itself (shouldn't happen normally)
            return color ? [...color] : [128, 128, 128];
        }
        return color ? [...color] : [128, 128, 128];
    });

    // Calculate color distribution of the recolored image based on effective palette
    const labTargets = effectivePalette.map(c => RGB2LAB(c));
    const counts = new Array(effectivePalette.length).fill(0);
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

    // Build distribution data using effective palette (originals for bypassed)
    const recoloredDistribution = effectivePalette.map((color, i) => ({
        color: [...color],
        pct: (counts[i] / totalSamples) * 100
    }));

    // Sort by percentage descending for display
    recoloredDistribution.sort((a, b) => b.pct - a.pct);

    // Filter out tiny colors and normalize so segments sum to 100%
    const visible = recoloredDistribution.filter(item => item.pct >= 0.5);
    const total = visible.reduce((sum, item) => sum + item.pct, 0);

    for (const item of visible) {
        const seg = document.createElement('div');
        seg.className = 'color-strip-segment';
        seg.style.background = rgbToHex(...item.color);
        seg.style.width = (total > 0 ? (item.pct / total) * 100 : 0) + '%';
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

    // Diagnostic: log target swatch colors — only when uiStage or target colors changed
    const selectable = (uiStage === 'target-selection' || uiStage === 'complete');
    const tgtKey = targetPalette.map((t, i) => t ? rgbToHex(...t) : 'null').join(',');
    if (renderColumnMapping._lastTgtKey !== tgtKey || renderColumnMapping._lastStage !== uiStage) {
        renderColumnMapping._lastTgtKey = tgtKey;
        renderColumnMapping._lastStage = uiStage;
        const tgtSummary = targetPalette.map((t, i) => t ? `col${i}:${rgbToHex(...t)}` : `col${i}:null`).join(', ');
        debugLog(`[render-swatches] uiStage=${uiStage}, selectable=${selectable}, targets=[${tgtSummary}]`);
    }

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
    const isBlankTargets = targetPalette.some(t => t === null);
    const hideSwatchButtons = (uiStage === 'colors-picked'); // Hide buttons until target selection
    // Targets are selectable only after the target selector has been engaged
    const targetsSelectable = (uiStage === 'target-selection' || uiStage === 'complete');

    for (let colIdx = 0; colIdx < targetCount; colIdx++) {
        const targetColumn = document.createElement('div');
        targetColumn.className = 'mapping-targets-column';
        targetColumn.dataset.columnIndex = colIdx;

        const targetDiv = document.createElement('div');
        targetDiv.className = 'column-target';

        const isBlank = targetPalette[colIdx] === null;

        const slot = document.createElement('div');
        let slotClass = 'color-slot';
        if (targetsSelectable && colIdx === selectedSlotIndex) slotClass += ' selected';
        if (isBlank) slotClass += ' blank-target';
        if (!targetsSelectable) slotClass += ' not-selectable';
        slot.className = slotClass;

        if (!isBlank) {
            const swatchHex = rgbToHex(...targetPalette[colIdx]);
            slot.style.background = swatchHex;
            slot.title = targetsSelectable
                ? swatchHex + ' (click to select, drag to swap)'
                : 'Engage Target Selector to modify';
        } else {
            slot.style.background = 'transparent';
            slot.title = targetsSelectable ? 'No target color assigned yet' : 'Engage Target Selector to assign colors';
        }
        if (targetsSelectable) {
            slot.onclick = () => selectSlot(colIdx);
        }

        // Make target swatch draggable when targets are selectable and swatch has a color
        if (targetsSelectable && !isBlank) {
            slot.draggable = true;
            slot.addEventListener('dragstart', (e) => {
                draggedTargetIndex = colIdx;
                slot.classList.add('dragging');
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/plain', 'target-' + colIdx);
            });
            slot.addEventListener('dragend', () => {
                slot.classList.remove('dragging');
                draggedTargetIndex = null;
                document.querySelectorAll('.column-target.drag-over-target').forEach(el => el.classList.remove('drag-over-target'));
            });
        }

        // Make this column a drop zone for target swatches
        if (targetsSelectable) {
            targetDiv.addEventListener('dragover', (e) => {
                if (draggedTargetIndex === null) return;
                e.preventDefault();
                targetDiv.classList.add('drag-over-target');
            });
            targetDiv.addEventListener('dragleave', () => {
                targetDiv.classList.remove('drag-over-target');
            });
            targetDiv.addEventListener('drop', (e) => {
                e.preventDefault();
                targetDiv.classList.remove('drag-over-target');
                if (draggedTargetIndex !== null && draggedTargetIndex !== colIdx) {
                    handleTargetSwap(draggedTargetIndex, colIdx);
                }
                draggedTargetIndex = null;
            });
        }

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

        // When bypassed, show all origin colors unconsolidated below the swatch
        let bypassOriginsRow = null;
        if (columnBypass[colIdx]) {
            const originsInCol = [];
            for (let i = 0; i < originToColumn.length; i++) {
                if (originToColumn[i] === colIdx && originalPalette[i]) {
                    originsInCol.push(i);
                }
            }
            if (originsInCol.length > 0) {
                bypassOriginsRow = document.createElement('div');
                bypassOriginsRow.className = 'bypass-origins-preview';
                originsInCol.forEach(oi => {
                    const mini = document.createElement('div');
                    mini.className = 'bypass-origin-mini';
                    mini.style.background = rgbToHex(...originalPalette[oi]);
                    mini.title = rgbToHex(...originalPalette[oi]) + ' - ' + (colorPercentages[oi]?.toFixed(1) || '?') + '%';
                    bypassOriginsRow.appendChild(mini);
                });
                // Show "locked" indicator on the swatch
                slot.style.background = 'repeating-linear-gradient(45deg, transparent, transparent 3px, rgba(0,0,0,0.05) 3px, rgba(0,0,0,0.05) 6px)';
                slot.style.border = '2px dashed var(--text-muted)';
            }
        }

        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.className = 'hidden-color-input';
        colorInput.id = 'colorPicker_' + colIdx;

        const hexLabel = document.createElement('div');
        hexLabel.className = 'swatch-hex';
        hexLabel.textContent = isBlank ? '—' : rgbToHex(...targetPalette[colIdx]);

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

        // Buttons - hidden until target selection stage
        const btnRow = document.createElement('div');
        btnRow.className = 'swatch-buttons' + (hideSwatchButtons ? ' target-hidden' : '');

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

        btnRow.appendChild(pickerBtn);
        btnRow.appendChild(bypassBtn);

        // Category label at top of target column
        const categoryLabel = document.createElement('div');
        categoryLabel.className = 'target-category-label';
        categoryLabel.textContent = getTargetCategoryLabel(colIdx);
        targetDiv.appendChild(categoryLabel);

        targetDiv.appendChild(slot);

        // Always reserve space for bypass origins row so buttons align across columns
        if (bypassOriginsRow) {
            targetDiv.appendChild(bypassOriginsRow);
        } else {
            const spacer = document.createElement('div');
            spacer.className = 'bypass-origins-spacer';
            targetDiv.appendChild(spacer);
        }

        targetDiv.appendChild(colorInput);
        targetDiv.appendChild(hexLabel);
        targetDiv.appendChild(pctLabel);
        targetDiv.appendChild(btnRow);

        // Create per-swatch color picker (hidden by default)
        const swatchPicker = createSwatchColorPicker(colIdx);
        targetDiv.appendChild(swatchPicker);

        targetColumn.appendChild(targetDiv);
        targetsRow.appendChild(targetColumn);
    }
    
    // Update hex input (skip if target is null/blank)
    if (targetPalette.length > 0 && targetPalette[selectedSlotIndex] !== null) {
        const hex = rgbToHex(...targetPalette[selectedSlotIndex]);
        document.getElementById('hexInput').value = hex;
        document.getElementById('hexPreview').style.background = hex;
    }

    // Refresh lock-a-color swatch pickers (always, so they're ready when user switches mode)
    if (typeof populateLockColorDropdowns === 'function') populateLockColorDropdowns();
    // Update harmony nudge swatch to reflect current selected target
    updateHarmonyNudgeSwatch();
    // Update palette icon visibility for beginner mode Direct Picker
    if (typeof updatePaletteIconVisibility === 'function') updatePaletteIconVisibility();
}

function toggleColumnBypass(colIdx) {
    columnBypass[colIdx] = !columnBypass[colIdx];
    debugLog(`[bypass-toggle] col=${colIdx} → ${columnBypass[colIdx] ? 'LOCKED' : 'unlocked'}`);
    renderColumnMapping();
    if (livePreviewEnabled) {
        autoRecolorImage();
    }
    // Record lock state change in recolor history
    addToRecolorHistory();
}

function handleTargetSwap(fromCol, toCol) {
    // Swap the target colors between two columns
    const tempColor = targetPalette[fromCol];
    targetPalette[fromCol] = targetPalette[toCol];
    targetPalette[toCol] = tempColor;

    // Swap target opacities
    const fromOpacity = targetOpacity[fromCol];
    const toOpacity = targetOpacity[toCol];
    if (fromOpacity !== undefined || toOpacity !== undefined) {
        if (toOpacity !== undefined) targetOpacity[fromCol] = toOpacity;
        else delete targetOpacity[fromCol];
        if (fromOpacity !== undefined) targetOpacity[toCol] = fromOpacity;
        else delete targetOpacity[toCol];
    }

    // Invalidate opacity cache since targets moved
    _opacityCache = null;

    // columnBypass stays with the column position — do NOT swap it

    // Re-render and recolor
    renderColumnMapping();
    if (livePreviewEnabled) {
        autoRecolorImage();
    }
    addToRecolorHistory();
    setStatus(`Swapped target colors between columns ${fromCol + 1} and ${toCol + 1}.`);
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
    wrapper.className = 'origin-swatch-wrapper';

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

    const currentOpacity = originOpacity[originIndex] !== undefined ? originOpacity[originIndex] : 100;

    // Opacity icon button — absolutely positioned to the right of the swatch
    const opBtn = document.createElement('button');
    opBtn.className = 'origin-opacity-btn';
    opBtn.title = 'Color opacity: ' + currentOpacity + '%';
    opBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>';
    if (currentOpacity < 100) opBtn.classList.add('has-custom');

    // Popup panel (positioned absolutely, initially hidden)
    const opPopup = document.createElement('div');
    opPopup.className = 'origin-opacity-popup';
    opPopup.id = `originOpPopup_${originIndex}`;

    const opSlider = document.createElement('input');
    opSlider.type = 'range';
    opSlider.className = 'origin-opacity-slider';
    opSlider.min = 0;
    opSlider.max = 100;
    opSlider.value = currentOpacity;
    opSlider.id = `originOpSlider_${originIndex}`;

    const opInput = document.createElement('input');
    opInput.type = 'number';
    opInput.className = 'origin-opacity-input';
    opInput.min = 0;
    opInput.max = 100;
    opInput.value = currentOpacity;
    opInput.id = `originOpInput_${originIndex}`;

    const opPct = document.createElement('span');
    opPct.className = 'origin-opacity-pct';
    opPct.textContent = '%';

    const opReset = document.createElement('button');
    opReset.className = 'origin-opacity-reset';
    opReset.textContent = '↺';
    opReset.title = 'Reset to 100%';

    function updateOpacity(v) {
        v = Math.max(0, Math.min(100, v));
        originOpacity[originIndex] = v;
        opSlider.value = v;
        opInput.value = v;
        opBtn.title = 'Color opacity: ' + v + '%';
        opBtn.classList.toggle('has-custom', v < 100);
        autoRecolorOpacityOnly();
    }

    opSlider.oninput = () => updateOpacity(parseInt(opSlider.value));
    opInput.oninput = () => {
        const v = parseInt(opInput.value);
        if (!isNaN(v)) updateOpacity(v);
    };
    opReset.onclick = (e) => {
        e.stopPropagation();
        updateOpacity(100);
    };

    // Toggle popup visibility — popup is a sibling of the swatch (not a child),
    // so we position it with fixed coordinates relative to the swatch
    opBtn.onclick = (e) => {
        e.stopPropagation();
        const isOpen = opPopup.classList.contains('open');
        document.querySelectorAll('.origin-opacity-popup.open').forEach(p => {
            p.classList.remove('open');
            p.style.position = '';
        });
        if (!isOpen) {
            const swatchRect = swatch.getBoundingClientRect();
            opPopup.style.position = 'fixed';
            opPopup.style.left = (swatchRect.left - 4) + 'px';
            opPopup.style.top = (swatchRect.top + swatchRect.height / 2) + 'px';
            opPopup.style.transform = 'translateX(-100%) translateY(-50%)';
            opPopup.style.right = 'auto';
            opPopup.classList.add('open');
        }
    };

    const opInputRow = document.createElement('div');
    opInputRow.className = 'origin-opacity-input-row';
    opInputRow.appendChild(opInput);
    opInputRow.appendChild(opPct);

    opPopup.appendChild(opSlider);
    opPopup.appendChild(opInputRow);
    opPopup.appendChild(opReset);

    // Build DOM: button inside swatch, popup as sibling (avoids drag conflict)
    swatch.appendChild(opBtn);
    wrapper.appendChild(swatch);
    wrapper.appendChild(opPopup);

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
    if (!c1 || !c2) return false;
    return c1[0] === c2[0] && c1[1] === c2[1] && c1[2] === c2[2];
}

// Get extracted colors not currently in the origin palette
function getUnusedExtractedColors() {
    // Return ALL extracted colors (per user spec: Add Color shows all extracted colors, not just unused)
    if (!fullColorDistribution || fullColorDistribution.length === 0) return [];
    return fullColorDistribution;
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
    updateHarmonyNudgeSwatch();
}

function openColorPicker(index) {
    const picker = document.getElementById('colorPicker_' + index);
    if (picker) picker.click();
}

// Per-swatch color picker state
let activeSwatchPicker = null;
let swatchPickerState = {};
let _isSwatchGradientDragging = false; // Global flag to prevent picker close during gradient drag

function createSwatchColorPicker(colIdx) {
    const picker = document.createElement('div');
    picker.className = 'swatch-color-picker';
    picker.id = `swatchPicker_${colIdx}`;

    // Get current color (use gray if blank)
    const currentColor = targetPalette[colIdx] || [128, 128, 128];
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
        e.preventDefault(); // Prevent text selection while dragging
        isDraggingSwatchGradient = true;
        _isSwatchGradientDragging = true;
        updateSwatchGradientFromMouse(e, colIdx);
    });

    document.addEventListener('mousemove', (e) => {
        if (isDraggingSwatchGradient && activeSwatchPicker === colIdx) {
            e.preventDefault(); // Prevent text selection while dragging
            updateSwatchGradientFromMouse(e, colIdx);
        }
    });

    document.addEventListener('mouseup', () => {
        if (isDraggingSwatchGradient) {
            isDraggingSwatchGradient = false;
            // Delay clearing the global flag so the click handler sees it
            setTimeout(() => { _isSwatchGradientDragging = false; }, 50);
        }
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

    // Opacity slider row
    const opacityRow = document.createElement('div');
    opacityRow.className = 'swatch-picker-opacity-row';
    const opacityLabel = document.createElement('span');
    opacityLabel.className = 'swatch-picker-opacity-label';
    opacityLabel.textContent = 'Opacity:';

    const opacitySlider = document.createElement('input');
    opacitySlider.type = 'range';
    opacitySlider.className = 'swatch-picker-opacity-slider';
    opacitySlider.min = 0;
    opacitySlider.max = 100;
    opacitySlider.value = targetOpacity[colIdx] !== undefined ? targetOpacity[colIdx] : 100;

    const opacityValueDisplay = document.createElement('span');
    opacityValueDisplay.className = 'swatch-picker-opacity-value';
    opacityValueDisplay.textContent = (targetOpacity[colIdx] !== undefined ? targetOpacity[colIdx] : 100) + '%';

    opacitySlider.oninput = () => {
        opacityValueDisplay.textContent = opacitySlider.value + '%';
    };

    const opacityReset = document.createElement('button');
    opacityReset.className = 'swatch-picker-opacity-reset';
    opacityReset.innerHTML = '↻';
    opacityReset.title = 'Reset to 100%';
    opacityReset.onclick = (e) => {
        e.stopPropagation();
        opacitySlider.value = 100;
        opacityValueDisplay.textContent = '100%';
    };

    opacityRow.appendChild(opacityLabel);
    opacityRow.appendChild(opacitySlider);
    opacityRow.appendChild(opacityValueDisplay);
    opacityRow.appendChild(opacityReset);

    // Apply button
    const applyBtn = document.createElement('button');
    applyBtn.className = 'swatch-picker-apply';
    applyBtn.textContent = 'Apply';
    applyBtn.onclick = () => {
        const state = swatchPickerState[colIdx];
        const rgb = hsvToRgb(state.h, state.s, state.v);
        targetPalette[colIdx] = rgb;
        targetOpacity[colIdx] = parseInt(opacitySlider.value);
        closeAllSwatchPickers();
        renderColumnMapping();
        autoRecolorImage();
        setStatus(`Color ${colIdx + 1} set to ${rgbToHex(...rgb)} at ${opacitySlider.value}% opacity`);
    };

    picker.appendChild(gradient);
    picker.appendChild(hueSlider);
    picker.appendChild(previewRow);
    picker.appendChild(opacityRow);
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

// Close pickers when clicking outside (but not during gradient drag)
document.addEventListener('click', (e) => {
    if (_isSwatchGradientDragging) return; // Don't close during gradient drag
    if (!e.target.closest('.swatch-color-picker') && !e.target.closest('.swatch-buttons')) {
        closeAllSwatchPickers();
    }
    // Close origin opacity popups when clicking outside
    if (!e.target.closest('.origin-opacity-popup') && !e.target.closest('.origin-opacity-btn')) {
        document.querySelectorAll('.origin-opacity-popup.open').forEach(p => {
            p.classList.remove('open');
            p.style.position = '';
        });
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
        // Skip bypassed columns during shuffle
        if (columnBypass[i]) continue;
        let j;
        do {
            j = Math.floor(Math.random() * (i + 1));
        } while (columnBypass[j] && j !== i); // avoid swapping into a bypassed slot
        if (!columnBypass[j]) {
            [targetPalette[i], targetPalette[j]] = [targetPalette[j], targetPalette[i]];
        }
    }
    renderColumnMapping();
    setStatus('Target colors shuffled');
    autoRecolorImage();
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
    if (!rgb) return; // Blank target - skip gradient update
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

    // Check if any targets are still blank
    if (targetPalette.some(t => t === null)) {
        setStatus('Please assign target colors first.');
        return;
    }

    showLoading();
    setStatus('Applying recolor...');

    setTimeout(() => {
        recolorImage();
        hideLoading();
        // Un-gray the recolored distribution strip
        document.getElementById('recoloredStrip').classList.remove('grayed-out');
        // Ensure full UI is revealed
        revealFullUI();
        // Add to recolor history
        addToRecolorHistory();
        setStatus('Recolor applied!');
    }, 50);
}

let _autoRecolorHistoryTimer = null;
function autoRecolorImage() {
    // Only recolor if an image is loaded and has been previously recolored
    if (!originalImageData) {
        return;
    }
    // Only auto-recolor if live preview is enabled
    if (!livePreviewEnabled) {
        debugLog('[auto-recolor] skipped — live preview OFF');
        return;
    }
    debugLog('[auto-recolor] triggered, invalidating opacity cache');
    // Invalidate opacity cache — palette/mapping changed, need full recompute
    _opacityCache = null;
    if (recolorImageOpacityFast._hitCount) recolorImageOpacityFast._hitCount = 0;
    // Silently recolor without loading UI (but still async for UI responsiveness)
    setTimeout(() => recolorImage(), 10);
    // Debounced history add — records after 1s of no further changes
    if (_autoRecolorHistoryTimer) clearTimeout(_autoRecolorHistoryTimer);
    _autoRecolorHistoryTimer = setTimeout(() => {
        addToRecolorHistory();
        _autoRecolorHistoryTimer = null;
    }, 1000);
}

// Fast opacity-only recolor: uses cached palette data + rAF throttle
function autoRecolorOpacityOnly() {
    if (!originalImageData || !livePreviewEnabled) return;

    // Use rAF to coalesce rapid slider events into one paint per frame
    if (_opacityRafPending) return;
    _opacityRafPending = true;
    requestAnimationFrame(() => {
        _opacityRafPending = false;
        recolorImageOpacityFast();
        // Debounced history
        if (_autoRecolorHistoryTimer) clearTimeout(_autoRecolorHistoryTimer);
        _autoRecolorHistoryTimer = setTimeout(() => {
            addToRecolorHistory();
            _autoRecolorHistoryTimer = null;
        }, 1000);
    });
}

function toggleLivePreview(enabled) {
    debugLog(`[live-preview] ${enabled ? 'ON' : 'OFF'}`);
    livePreviewEnabled = enabled;
    const applyBtn = document.getElementById('applyRecolorBtn');
    const applyText = document.getElementById('applyRecolorText');
    if (applyBtn && applyText) {
        if (enabled) {
            applyBtn.classList.remove('btn-primary');
            applyBtn.classList.add('btn-live-inactive');
            applyBtn.disabled = true;
            applyText.textContent = 'Apply this recolor (Live is Active)';
        } else {
            applyBtn.classList.add('btn-primary');
            applyBtn.classList.remove('btn-live-inactive');
            applyBtn.disabled = false;
            applyText.textContent = 'Apply this recolor';
        }
    }
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
    debugRenderStart();

    // Use setTimeout to let the loading indicator render
    setTimeout(() => {
        try {
            doRecolorImage();
            debugRenderEnd(true);
        } catch (e) {
            console.error('Recolor failed:', e);
            debugLog('EXCEPTION: ' + (e.message || e), 'error');
            debugRenderEnd(false);
            setStatus('Recolor failed — try a different algorithm or palette');
        }
        hideLoading();
    }, 50);
}

// Fast opacity-only recolor: recomputes only the palette diff using cached full-opacity
// target LAB values, then feeds it to the same WebGL/CPU pipeline. Skips the expensive
// RBF matrix/grid computation or re-building palette from scratch.
function recolorImageOpacityFast() {
    const k = originalPalette.length;
    if (k === 0 || !originalImageData) return;

    // Build or reuse cache of full-opacity target LAB values
    if (!_opacityCache || _opacityCache.k !== k || _opacityCache.algorithm !== selectedAlgorithm) {
        // Cache miss — need full recolor first to populate cache
        debugLog(`[opacity-fast] cache MISS (k=${k}, algo=${selectedAlgorithm}), falling back to full recolor`);
        recolorImage();
        return;
    }

    const { oldLab, bgLab } = _opacityCache;

    // Read target palette LIVE (not from cache) to guarantee we always use the
    // current targetPalette values.  Only oldLab and bgLab are stable enough to cache.
    const liveTargetLab = {};
    for (let col = 0; col < targetPalette.length; col++) {
        if (targetPalette[col] !== null && targetPalette[col] !== undefined) {
            liveTargetLab[col] = RGB2LAB(targetPalette[col]);
        }
    }

    // Log cache state — only on first call + every 60th to avoid flooding
    if (!recolorImageOpacityFast._hitCount) recolorImageOpacityFast._hitCount = 0;
    recolorImageOpacityFast._hitCount++;
    if (recolorImageOpacityFast._hitCount === 1 || recolorImageOpacityFast._hitCount % 60 === 0) {
        const targetSummary = Object.entries(liveTargetLab).map(([col, lab]) => {
            const curTP = targetPalette[col];
            const curHex = curTP ? rgbToHex(...curTP) : 'null';
            return `col${col}:${curHex}→L${lab[0].toFixed(0)}`;
        }).join(', ');
        debugLog(`[opacity-fast] cache HIT #${recolorImageOpacityFast._hitCount}, algo=${selectedAlgorithm}, targets=[${targetSummary}]`);
    }
    // Intermediate hits logged only to console (not render log) for minimal noise
    // Use browser DevTools console if you need per-tick visibility

    // Recompute newLab with current opacity values (instant — just array math)
    // Opacity simulates synthetic transparency: blends target color over active background
    const newLab = [];
    for (let i = 0; i < k; i++) {
        const col = originToColumn[i];
        if (col === 'locked' || col === 'bank' || typeof col !== 'number' || col >= targetPalette.length) {
            newLab.push(oldLab[i]);
        } else if (columnBypass[col]) {
            newLab.push(oldLab[i]);
        } else if (targetPalette[col] === null || !liveTargetLab[col]) {
            newLab.push(oldLab[i]);
        } else {
            const oOpacity = (originOpacity[i] !== undefined ? originOpacity[i] : 100) / 100;
            const tOpacity = (targetOpacity[col] !== undefined ? targetOpacity[col] : 100) / 100;
            const effectiveOpacity = oOpacity * tOpacity;

            if (effectiveOpacity >= 1.0) {
                newLab.push(liveTargetLab[col]);
            } else if (effectiveOpacity <= 0.0) {
                // Fully transparent: show background color
                newLab.push([...bgLab]);
            } else {
                // Blend target color over background at effectiveOpacity
                newLab.push([
                    bgLab[0] + (liveTargetLab[col][0] - bgLab[0]) * effectiveOpacity,
                    bgLab[1] + (liveTargetLab[col][1] - bgLab[1]) * effectiveOpacity,
                    bgLab[2] + (liveTargetLab[col][2] - bgLab[2]) * effectiveOpacity
                ]);
            }
        }
    }

    const diffLab = oldLab.map((old, i) => [
        newLab[i][0] - old[0],
        newLab[i][1] - old[1],
        newLab[i][2] - old[2]
    ]);

    const width = canvas.width;
    const height = canvas.height;

    if (selectedAlgorithm === 'rbf') {
        // For RBF, we need the full grid recomputation — fall back to full recolor
        // but still benefit from rAF throttle
        doRecolorRBF();
    } else {
        // Simple algorithm: just need oldLab + diffLab — skip palette rebuild
        const luminosity = 0;

        // Debug: compare opacity-fast diffLab vs last full-recolor diffLab
        // Only log on first tick of each opacity drag session (not every frame)
        // to reduce render log noise. Drift is expected during opacity changes.
        if (_lastSimpleUniforms && _lastSimpleUniforms.diffLabFlat && recolorImageOpacityFast._hitCount === 1) {
            const prev = _lastSimpleUniforms.diffLabFlat;
            let maxDelta = 0;
            let maxIdx = -1;
            for (let i = 0; i < k; i++) {
                for (let c = 0; c < 3; c++) {
                    const delta = Math.abs(diffLab[i][c] - prev[i*3 + c]);
                    if (delta > maxDelta) { maxDelta = delta; maxIdx = i; }
                }
            }
            if (maxDelta > 0.5) {
                const col = originToColumn[maxIdx];
                const curTP = col >= 0 && col < targetPalette.length && targetPalette[col] ? rgbToHex(...targetPalette[col]) : '?';
                debugLog(`[opacity-fast-drift] first-tick maxΔ=${maxDelta.toFixed(2)} at origin#${maxIdx}→col${col} target=${curTP}`);
            }
        }

        if (initWebGL() && doRecolorSimpleWebGL(width, height, k, oldLab, diffLab, luminosity)) {
            return;
        }
        doRecolorSimpleCPU(width, height, k, oldLab, diffLab, luminosity);
    }
}

function doRecolorImage() {
    // Dispatch to selected algorithm
    debugLog('doRecolorImage() dispatching to: ' + selectedAlgorithm);
    if (selectedAlgorithm === 'rbf') {
        doRecolorRBF();
    } else {
        doRecolorSimple();
    }
}

// Get the active background color in LAB for synthetic opacity layering.
// Column 0 = CATEGORY_BACKGROUND. If bypassed (locked), use the representative
// original background color; otherwise use the target background color.
function getActiveBackgroundLab(oldLab) {
    const bgCol = 0; // CATEGORY_BACKGROUND is always column 0

    // If background column is bypassed (locked), use average of origin colors in that column
    if (columnBypass[bgCol]) {
        let count = 0;
        let sumLab = [0, 0, 0];
        for (let j = 0; j < originToColumn.length; j++) {
            if (originToColumn[j] === bgCol && originalPalette[j]) {
                const lab = oldLab ? oldLab[j] : RGB2LAB(originalPalette[j]);
                sumLab[0] += lab[0];
                sumLab[1] += lab[1];
                sumLab[2] += lab[2];
                count++;
            }
        }
        if (count > 0) {
            return [sumLab[0] / count, sumLab[1] / count, sumLab[2] / count];
        }
        // Fallback: first origin in column 0
        if (originalPalette[0]) return oldLab ? oldLab[0] : RGB2LAB(originalPalette[0]);
    }

    // Not bypassed: use the target color for the background column
    if (targetPalette[bgCol] !== null && targetPalette[bgCol] !== undefined) {
        return RGB2LAB(targetPalette[bgCol]);
    }

    // Fallback: use original palette first color
    if (originalPalette[0]) return oldLab ? oldLab[0] : RGB2LAB(originalPalette[0]);

    // Ultimate fallback: white
    return RGB2LAB([255, 255, 255]);
}

// Simple weighted nearest-neighbor algorithm (v14 default)
function doRecolorSimple() {
    const width = canvas.width;
    const height = canvas.height;
    const k = originalPalette.length;

    if (k === 0) return;
    debugLog('doRecolorSimple() — k=' + k + ', image=' + width + 'x' + height);

    const oldLab = originalPalette.map(c => RGB2LAB(c));
    const newLab = [];

    // Cache full-opacity target LAB values for fast opacity-only rerenders
    const fullTargetLab = {};
    for (let col = 0; col < targetPalette.length; col++) {
        if (targetPalette[col] !== null && targetPalette[col] !== undefined) {
            fullTargetLab[col] = RGB2LAB(targetPalette[col]);
        }
    }
    const bgLab = getActiveBackgroundLab(oldLab);
    _opacityCache = { oldLab, fullTargetLab, bgLab, k, algorithm: 'simple' };

    // Log cache contents for debugging target color accuracy
    const cacheTargets = Object.entries(fullTargetLab).map(([col, lab]) => {
        const tp = targetPalette[col];
        return `col${col}:${tp ? rgbToHex(...tp) : 'null'}→L${lab[0].toFixed(1)}a${lab[1].toFixed(1)}b${lab[2].toFixed(1)}`;
    }).join(', ');
    debugLog(`[opacity-cache-built] targets=[${cacheTargets}], bg=L${bgLab[0].toFixed(1)}`);

    // Build origin-to-target mapping based on columns, respecting opacity
    // Opacity simulates synthetic transparency: blends target color over active background
    for (let i = 0; i < k; i++) {
        const col = originToColumn[i];
        if (col === 'locked' || col === 'bank' || typeof col !== 'number' || col >= targetPalette.length) {
            newLab.push(oldLab[i]);
        } else if (columnBypass[col]) {
            newLab.push(oldLab[i]);
        } else if (targetPalette[col] === null) {
            newLab.push(oldLab[i]);
        } else {
            const oOpacity = (originOpacity[i] !== undefined ? originOpacity[i] : 100) / 100;
            const tOpacity = (targetOpacity[col] !== undefined ? targetOpacity[col] : 100) / 100;
            const effectiveOpacity = oOpacity * tOpacity;

            if (effectiveOpacity >= 1.0) {
                newLab.push(fullTargetLab[col]);
            } else if (effectiveOpacity <= 0.0) {
                // Fully transparent: show background color
                newLab.push([...bgLab]);
            } else {
                // Blend target color over background at effectiveOpacity
                newLab.push([
                    bgLab[0] + (fullTargetLab[col][0] - bgLab[0]) * effectiveOpacity,
                    bgLab[1] + (fullTargetLab[col][1] - bgLab[1]) * effectiveOpacity,
                    bgLab[2] + (fullTargetLab[col][2] - bgLab[2]) * effectiveOpacity
                ]);
            }
        }
    }

    const diffLab = oldLab.map((old, i) => [
        newLab[i][0] - old[0],
        newLab[i][1] - old[1],
        newLab[i][2] - old[2]
    ]);

    // Luminosity is now a separate post-processing step (applyLuminosityPostProcess)
    const luminosity = 0;

    // Try WebGL first
    if (initWebGL() && doRecolorSimpleWebGL(width, height, k, oldLab, diffLab, luminosity)) {
        return;
    }

    // CPU fallback
    console.log('Using CPU fallback for simple recolor');
    doRecolorSimpleCPU(width, height, k, oldLab, diffLab, luminosity);
}

// WebGL implementation of simple recolor — renders directly to visible display canvas
function doRecolorSimpleWebGL(width, height, k, oldLab, diffLab, luminosity) {
    try {
        const startTime = performance.now();

        // Upload palette data as uniform arrays
        const oldLabFlat = new Float32Array(60); // 20 * 3
        const diffLabFlat = new Float32Array(60);
        for (let i = 0; i < 20; i++) {
            if (i < k) {
                oldLabFlat[i*3] = oldLab[i][0]; oldLabFlat[i*3+1] = oldLab[i][1]; oldLabFlat[i*3+2] = oldLab[i][2];
                diffLabFlat[i*3] = diffLab[i][0]; diffLabFlat[i*3+1] = diffLab[i][1]; diffLabFlat[i*3+2] = diffLab[i][2];
            }
        }

        // Cache uniforms for re-render on zoom and for deferred CPU sync
        _lastSimpleUniforms = { oldLabFlat, diffLabFlat, k, luminosity };
        _lastWebGLRenderType = 'simple';
        _lastRBFUniforms = null;

        // Render to display canvas at current display size
        renderWebGLToDisplay('simple');

        // Mark CPU-side imageData as stale
        _webglDirty = true;

        const elapsed = performance.now() - startTime;
        console.log(`WebGL simple recolor: ${elapsed.toFixed(1)}ms`);

        renderRecoloredStripDeferred();
        return true;
    } catch (e) {
        console.warn('WebGL simple recolor failed:', e);
        return false;
    }
}

// Core WebGL-to-display renderer — sets up the display canvas at zoom-appropriate
// resolution, renders the recolor shader, and updates CSS sizing for zoom/pan.
// Called after every recolor AND after every zoom change.
function renderWebGLToDisplay(type) {
    if (!gl || !webglCanvas) return;

    const wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) return;
    const wrapperRect = wrapper.getBoundingClientRect();
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;
    if (imgWidth === 0 || imgHeight === 0) return;
    const aspectRatio = imgWidth / imgHeight;

    const baseWidth = wrapperRect.width;
    const baseHeight = baseWidth / aspectRatio;

    // Guard: mid-layout-transition
    if (baseWidth < 1 || baseHeight < 1) return;

    // Always render at full image resolution so the shader processes every
    // source pixel without downsampling.  CSS sizing handles display scaling.
    // This prevents cross-color bleed at text edges and fine detail boundaries.
    const renderWidth = imgWidth;
    const renderHeight = imgHeight;

    // Display size on screen = base * zoom
    const displayWidth = Math.max(1, Math.round(baseWidth * zoomLevel));
    const displayHeight = Math.max(1, Math.round(baseHeight * zoomLevel));

    // Set WebGL canvas pixel buffer to full image resolution
    debugLog(`[renderWebGL] pixels=${renderWidth}x${renderHeight}, css=${displayWidth}x${displayHeight}, wrapper=${baseWidth.toFixed(0)}x${baseHeight.toFixed(0)}, zoom=${zoomLevel}`);
    webglCanvas.width = renderWidth;
    webglCanvas.height = renderHeight;
    gl.viewport(0, 0, renderWidth, renderHeight);

    // Render the appropriate shader using cached uniform locations
    const renderType = type || _lastWebGLRenderType;
    if (renderType === 'simple' && _lastSimpleUniforms && _simpleUniformLocs) {
        const u = _lastSimpleUniforms;
        const loc = _simpleUniformLocs;
        gl.useProgram(simpleRecolorProgram);
        setupWebGLBuffers(gl, simpleRecolorProgram);
        const tex = ensureImageTexture();
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(loc.u_image, 0);
        gl.uniform3fv(loc.u_oldLab, u.oldLabFlat);
        gl.uniform3fv(loc.u_diffLab, u.diffLabFlat);
        gl.uniform1i(loc.u_paletteSize, u.k);
        gl.uniform1f(loc.u_blendSharpness, 2.0);
        gl.uniform1f(loc.u_luminosity, u.luminosity);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    } else if (renderType === 'rbf' && _lastRBFUniforms && _rbfUniformLocs) {
        const u = _lastRBFUniforms;
        const loc = _rbfUniformLocs;
        gl.useProgram(rbfRecolorProgram);
        setupWebGLBuffers(gl, rbfRecolorProgram);
        const tex = ensureImageTexture();
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(loc.u_image, 0);
        const lutTexture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, lutTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, u.lutWidth, u.lutHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, u.lutData);
        gl.uniform1i(loc.u_lut, 1);
        gl.uniform1f(loc.u_lutSize, u.lutSize);
        gl.uniform1f(loc.u_luminosity, u.luminosity);
        // Bypass correction uniforms
        gl.uniform3fv(loc.u_bypassedRGB, u.bypassedFlat || new Float32Array(60));
        gl.uniform1i(loc.u_bypassedCount, u.bypassedCount || 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.deleteTexture(lutTexture);
    }

    // CSS size = actual zoomed display size (browser scales down from full-res pixel buffer)
    webglCanvas.style.width = displayWidth + 'px';
    webglCanvas.style.height = displayHeight + 'px';
    webglCanvas.style.transform = 'scale(1)';
    webglCanvas.style.transformOrigin = '0 0';

    baseDisplayWidth = displayWidth;
    baseDisplayHeight = displayHeight;
    lastRenderedZoom = zoomLevel;

    // Show WebGL canvas, hide both CPU canvases
    const canvasInner = document.getElementById('canvasInner');
    const firstInsertion = canvasInner && !webglCanvas.parentNode;
    if (firstInsertion) {
        canvasInner.insertBefore(webglCanvas, canvas);
    }
    webglCanvas.style.display = 'block';
    if (displayCanvas) displayCanvas.style.display = 'none';
    canvas.style.display = 'none';

    // After DOM changes (canvas swap, sidebar reveal, etc.), the wrapper may
    // reflow to a different size.  Re-measure and correct CSS sizing once
    // the layout has settled — both immediately after paint and after a short
    // delay for sidebar transitions.
    const _resyncCSS = () => {
        if (!webglCanvas || webglCanvas.style.display === 'none') return;
        const w = document.getElementById('canvasWrapper');
        if (!w) return;
        const rect = w.getBoundingClientRect();
        const bw = rect.width;
        const bh = bw / (imgWidth / imgHeight);
        if (bw < 1 || bh < 1) return;
        const dw = Math.max(1, Math.round(bw * zoomLevel));
        const dh = Math.max(1, Math.round(bh * zoomLevel));
        if (webglCanvas.style.width !== dw + 'px' || webglCanvas.style.height !== dh + 'px') {
            debugLog(`[resyncCSS] correcting css ${webglCanvas.style.width}x${webglCanvas.style.height} → ${dw}x${dh}, wrapper=${bw.toFixed(0)}`, 'warn');
            webglCanvas.style.width = dw + 'px';
            webglCanvas.style.height = dh + 'px';
            baseDisplayWidth = dw;
            baseDisplayHeight = dh;
        }
    };
    // Next frame (catches immediate reflows from canvas swap)
    requestAnimationFrame(_resyncCSS);
    // Deferred (catches sidebar/panel transitions that settle after the swap)
    setTimeout(_resyncCSS, 250);
    // Extra safety: some layout shifts (sidebar reveal, collapsible expansion) may
    // not settle until well after 250ms — catch those too
    setTimeout(_resyncCSS, 600);
}

// Deferred strip rebuild — syncs CPU pixels then rebuilds the strip.
// Uses a short debounce to avoid repeated syncs during rapid updates.
let _stripRebuildTimeout = null;
function renderRecoloredStripDeferred() {
    if (_stripRebuildTimeout) clearTimeout(_stripRebuildTimeout);
    _stripRebuildTimeout = setTimeout(() => {
        syncWebGLToCPU();
        renderRecoloredStrip();
        _stripRebuildTimeout = null;
    }, 200);
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
    _lastWebGLRenderType = null; // Using CPU path, not WebGL
    _webglDirty = false;
    updateDisplayCanvas();
    renderRecoloredStrip();
}

// RBF (Radial Basis Function) algorithm with grid precomputation (from v2)
// Better for smooth gradients in photos
function doRecolorRBF() {
    const RBF_param_coff = 5;
    const ngrid = 16;
    const width = canvas.width;
    const height = canvas.height;
    const k = originalPalette.length;

    if (k === 0) return;

    debugLog('doRecolorRBF() called — k=' + k + ', ngrid=' + ngrid + ', image=' + width + 'x' + height);

    // Wrap entire RBF computation — if anything fails, fall back to Simple
    try {
        return _doRecolorRBFInner(RBF_param_coff, ngrid, width, height, k);
    } catch (e) {
        console.error('RBF recolor failed, falling back to Simple:', e);
        debugLog('RBF EXCEPTION: ' + (e.message || e), 'error');
        debugLog('Falling back to Simple algorithm', 'warn');
        setStatus('RBF failed — using Simple algorithm instead');
        doRecolorSimple();
    }
}

function _doRecolorRBFInner(RBF_param_coff, ngrid, width, height, k) {

    debugLog('Building LAB grid (' + ((ngrid+1)**3) + ' entries)...');
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

    // Build origin-to-target mapping based on columns, respecting opacity
    const oldLab = originalPalette.map(c => RGB2LAB(c));
    const newLab = [];

    // Cache full-opacity target LAB values for fast opacity-only rerenders
    const fullTargetLab = {};
    for (let col = 0; col < targetPalette.length; col++) {
        if (targetPalette[col] !== null && targetPalette[col] !== undefined) {
            fullTargetLab[col] = RGB2LAB(targetPalette[col]);
        }
    }
    const bgLab = getActiveBackgroundLab(oldLab);
    _opacityCache = { oldLab, fullTargetLab, bgLab, k, algorithm: 'rbf' };

    // Opacity simulates synthetic transparency: blends target color over active background
    for (let i = 0; i < k; i++) {
        const col = originToColumn[i];
        if (col === 'locked' || col === 'bank' || typeof col !== 'number' || col >= targetPalette.length) {
            newLab.push(oldLab[i]);
        } else if (columnBypass[col]) {
            newLab.push(oldLab[i]);
        } else if (targetPalette[col] === null) {
            newLab.push(oldLab[i]);
        } else {
            const oOpacity = (originOpacity[i] !== undefined ? originOpacity[i] : 100) / 100;
            const tOpacity = (targetOpacity[col] !== undefined ? targetOpacity[col] : 100) / 100;
            const effectiveOpacity = oOpacity * tOpacity;

            if (effectiveOpacity >= 1.0) {
                newLab.push(fullTargetLab[col]);
            } else if (effectiveOpacity <= 0.0) {
                // Fully transparent: show background color
                newLab.push([...bgLab]);
            } else {
                // Blend target color over background at effectiveOpacity
                newLab.push([
                    bgLab[0] + (fullTargetLab[col][0] - bgLab[0]) * effectiveOpacity,
                    bgLab[1] + (fullTargetLab[col][1] - bgLab[1]) * effectiveOpacity,
                    bgLab[2] + (fullTargetLab[col][2] - bgLab[2]) * effectiveOpacity
                ]);
            }
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

    debugLog('LAB grid built. Building RBF matrix (' + k + 'x' + k + ')...');
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

    debugLog('RBF matrix built. Checking for degenerate palette...');
    // Check for degenerate palette (duplicate/near-duplicate origins cause SVD to hang)
    for (let i = 0; i < k; i++) {
        for (let j = i + 1; j < k; j++) {
            const d2 = Math.pow(oldLab[i][0] - oldLab[j][0], 2) +
                       Math.pow(oldLab[i][1] - oldLab[j][1], 2) +
                       Math.pow(oldLab[i][2] - oldLab[j][2], 2);
            if (d2 < 1.0) {
                // Near-duplicate origins — RBF matrix will be singular
                debugLog('Near-duplicate palette colors at indices ' + i + ',' + j + ' (dist²=' + d2.toFixed(4) + ') — falling back to Simple', 'warn');
                console.warn('RBF: near-duplicate palette colors detected, falling back to Simple');
                setStatus('RBF unavailable — palette has near-duplicate colors. Using Simple.');
                doRecolorSimple();
                return;
            }
        }
    }

    // Tikhonov regularization: add small epsilon to diagonal to ensure
    // the matrix is well-conditioned and numeric.svd converges reliably.
    // Without this, near-singular matrices can cause SVD to hang in an infinite loop.
    const regularizationEpsilon = 1e-6;
    for (let i = 0; i < k; i++) {
        rbfMatrix[i][i] += regularizationEpsilon;
    }
    debugLog('Matrix regularized (eps=' + regularizationEpsilon + '). Computing pinv (SVD) for ' + k + 'x' + k + ' matrix...');

    // Validate matrix before SVD
    let matrixHasNaN = false, matrixHasInf = false;
    for (let mi = 0; mi < k && !matrixHasNaN && !matrixHasInf; mi++) {
        for (let mj = 0; mj < k; mj++) {
            if (isNaN(rbfMatrix[mi][mj])) { matrixHasNaN = true; break; }
            if (!isFinite(rbfMatrix[mi][mj])) { matrixHasInf = true; break; }
        }
    }
    if (matrixHasNaN || matrixHasInf) {
        debugLog('RBF matrix contains ' + (matrixHasNaN ? 'NaN' : 'Infinity') + ' — cannot compute SVD', 'error');
        debugLog('avgDist=' + avgDist + ', param=' + param, 'error');
        debugLog('Diagonal sample: ' + rbfMatrix.slice(0, 4).map((r, i) => r[i].toString()).join(', '), 'error');
        console.error('RBF matrix has invalid values, falling back to Simple');
        setStatus('RBF failed — matrix has invalid values. Using Simple.');
        doRecolorSimple();
        return;
    }

    debugLog('Matrix validated OK. Diagonal[0]=' + rbfMatrix[0][0].toFixed(6) + ', off-diag sample=' + rbfMatrix[0][1].toFixed(6));

    let rbfInv;
    try {
        rbfInv = pinv(rbfMatrix);
    } catch (e) {
        debugLog('pinv/SVD threw exception: ' + (e.message || e), 'error');
        debugLog('Exception stack: ' + (e.stack ? e.stack.split('\\n').slice(0, 3).join(' | ') : 'no stack'), 'error');

        // Retry with stronger regularization before giving up
        const retryEpsilon = 1e-3;
        debugLog('Retrying SVD with stronger regularization (eps=' + retryEpsilon + ')...', 'warn');
        for (let ri = 0; ri < k; ri++) {
            rbfMatrix[ri][ri] += retryEpsilon - regularizationEpsilon;
        }
        try {
            rbfInv = pinv(rbfMatrix);
            debugLog('SVD succeeded with stronger regularization', 'info');
        } catch (e2) {
            debugLog('SVD retry also failed: ' + (e2.message || e2), 'error');
            console.error('RBF matrix inversion failed (SVD error), falling back to Simple:', e2);
            setStatus('RBF failed for this palette — using Simple algorithm instead');
            doRecolorSimple();
            return;
        }
    }

    debugLog('pinv returned. Validating result...');
    // Verify pinv result is valid (not NaN)
    if (!rbfInv || !rbfInv[0] || isNaN(rbfInv[0][0])) {
        debugLog('pinv result is NaN — matrix is ill-conditioned', 'error');
        console.error('RBF matrix inversion produced NaN, falling back to Simple');
        setStatus('RBF failed for this palette — using Simple algorithm instead');
        doRecolorSimple();
        return;
    }

    debugLog('pinv completed successfully. Computing grid transformation (' + gridSize + ' entries)...');

    // Calculate grid transformation (CPU - fast for 4913 entries)
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

    // Luminosity is now a separate post-processing step (applyLuminosityPostProcess)
    const luminosity = 0;

    // Collect bypassed palette colors for per-pixel correction.
    // RBF is a global color space transform — the LUT can bleed changes into
    // bypassed regions.  After the LUT lookup, pixels close to a bypassed
    // palette color are blended back toward their original value.
    const bypassedRGB = [];
    for (let i = 0; i < k; i++) {
        const col = originToColumn[i];
        // Collect all origin colors whose diffLab is [0,0,0] — they should stay put.
        // This includes: bypassed columns, locked/bank entries, unassigned entries,
        // and entries mapped to null target slots.
        if (col === 'locked' || col === 'bank' || typeof col !== 'number' || col >= targetPalette.length) {
            bypassedRGB.push(originalPalette[i]);
        } else if (columnBypass[col]) {
            bypassedRGB.push(originalPalette[i]);
        } else if (targetPalette[col] === null) {
            bypassedRGB.push(originalPalette[i]);
        }
    }

    debugLog('Grid transformation complete. Starting per-pixel recolor...');
    // Try WebGL for the per-pixel trilinear interpolation (the slow part)
    if (initWebGL() && doRecolorRBFWebGL(width, height, ngrid, gridRGB, luminosity, bypassedRGB)) {
        debugLog('WebGL RBF render complete');
        return;
    }

    // CPU fallback
    debugLog('WebGL unavailable, using CPU fallback for RBF recolor', 'warn');
    console.log('Using CPU fallback for RBF recolor');
    doRecolorRBFCPU(width, height, ngrid, gridRGB, luminosity, bypassedRGB);
    debugLog('CPU RBF render complete');
}

// WebGL implementation of RBF recolor (trilinear interpolation on GPU)
// Renders directly to visible display canvas — no readPixels
function doRecolorRBFWebGL(width, height, ngrid, gridRGB, luminosity, bypassedRGB) {
    try {
        const startTime = performance.now();
        const lutSize = ngrid + 1;
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

        // Pack bypassed palette colors as flat Float32Array (RGB 0-1 range)
        const bypassedFlat = new Float32Array(60); // 20 * 3 max
        const bypassedCount = Math.min(bypassedRGB ? bypassedRGB.length : 0, 20);
        for (let i = 0; i < bypassedCount; i++) {
            bypassedFlat[i * 3]     = bypassedRGB[i][0] / 255;
            bypassedFlat[i * 3 + 1] = bypassedRGB[i][1] / 255;
            bypassedFlat[i * 3 + 2] = bypassedRGB[i][2] / 255;
        }

        // Cache uniforms for re-render on zoom and deferred CPU sync
        _lastRBFUniforms = { lutData, lutWidth, lutHeight, lutSize, luminosity, bypassedFlat, bypassedCount };
        _lastWebGLRenderType = 'rbf';
        _lastSimpleUniforms = null;

        // Render to display canvas at current display size
        renderWebGLToDisplay('rbf');

        // Mark CPU-side imageData as stale
        _webglDirty = true;

        const elapsed = performance.now() - startTime;
        console.log(`WebGL RBF recolor: ${elapsed.toFixed(1)}ms`);

        renderRecoloredStripDeferred();
        return true;
    } catch (e) {
        console.warn('WebGL RBF recolor failed:', e);
        return false;
    }
}

// CPU fallback for RBF recolor
function doRecolorRBFCPU(width, height, ngrid, gridRGB, luminosity, bypassedRGB) {
    const startTime = performance.now();
    const newData = ctx.createImageData(width, height);
    const ntmp = ngrid + 1;
    const ntmpsqr = ntmp * ntmp;

    // Pre-normalize bypassed colors to 0-1 range for distance calculation
    const bypassedNorm = (bypassedRGB || []).map(c => [c[0] / 255, c[1] / 255, c[2] / 255]);
    const sigma2x2 = 2.0 * 0.12 * 0.12; // Gaussian sigma matching WebGL shader

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

        // Bypass correction: blend back toward original for pixels near bypassed colors
        if (bypassedNorm.length > 0) {
            const rn = r / 255, gn = g / 255, bn = b / 255;
            let maxKeep = 0;
            for (let j = 0; j < bypassedNorm.length; j++) {
                const dr = rn - bypassedNorm[j][0];
                const dg = gn - bypassedNorm[j][1];
                const db = bn - bypassedNorm[j][2];
                const d2 = dr * dr + dg * dg + db * db;
                const keep = Math.exp(-d2 / sigma2x2);
                if (keep > maxKeep) maxKeep = keep;
            }
            if (maxKeep > 0.001) {
                newR = newR + (r - newR) * maxKeep;
                newG = newG + (g - newG) * maxKeep;
                newB = newB + (b - newB) * maxKeep;
            }
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
    _lastWebGLRenderType = null; // Using CPU path, not WebGL
    _webglDirty = false;
    updateDisplayCanvas();
    renderRecoloredStrip();
}

// ============================================
// Color Picker Mode
// ============================================

// Update the Pick Colors button text and style after colors have been picked
function updatePickColorsButtonState(colorsPicked) {
    const sidebarBtn = document.getElementById('sidebarPickColorsBtn');
    const overlayBtn = document.getElementById('pickerToggleBtn');

    if (colorsPicked) {
        if (sidebarBtn) {
            sidebarBtn.innerHTML = '<span class="step-circle step-circle-btn">2</span> <span>🎯</span> Edit Origin Colors';
            sidebarBtn.classList.add('colors-picked');
        }
        if (overlayBtn) {
            const textSpan = overlayBtn.querySelector('.picker-toggle-text');
            if (textSpan) textSpan.innerHTML = 'Edit Origin<br><span class="picker-toggle-subtitle">Colors</span>';
            overlayBtn.classList.add('colors-picked');
        }
    } else {
        if (sidebarBtn) {
            sidebarBtn.innerHTML = '<span class="step-circle step-circle-btn">2</span> <span>🎯</span> Pick Colors';
            sidebarBtn.classList.remove('colors-picked');
        }
        if (overlayBtn) {
            const textSpan = overlayBtn.querySelector('.picker-toggle-text');
            if (textSpan) textSpan.textContent = 'Pick Colors';
            overlayBtn.classList.remove('colors-picked');
        }
    }
}

function togglePickerMode() {
    pickerMode = !pickerMode;
    const btn = document.getElementById('pickerToggleBtn');
    const canvasInner = document.getElementById('canvasInner');
    const canvasWrapper = document.getElementById('canvasWrapper');
    const categorySelector = document.getElementById('pickerCategorySelector');
    const pickerPanHint = document.getElementById('pickerPanHint');
    const pickerCtrlHint = document.getElementById('pickerCtrlHint');
    const sidebarPickBtn = document.getElementById('sidebarPickColorsBtn');

    if (pickerMode) {
        // Preserve picked colors - don't clear them when re-entering picker mode
        // User can use "Clear Selections" button if they want to start over
        shouldKeepPickedMarkers = false; // Allow markers to be managed normally

        btn.classList.add('active');
        if (sidebarPickBtn) {
            sidebarPickBtn.classList.add('active');
            sidebarPickBtn.innerHTML = '<span class="step-circle step-circle-btn">2</span> <span>🎯</span> Pick Colors (tool active)';
        }
        canvasInner.classList.add('picking-mode');
        canvasWrapper.classList.add('picking-mode');

        // Show picker hints (only if tutorial is not active)
        // Tutorial has its own hints

        // Rebuild markers for any existing picked colors
        document.querySelectorAll('.picker-marker').forEach(m => m.remove());
        pickedPositions.forEach((_, i) => createMarker(i));

        // Show and populate category selector
        categorySelector.classList.remove('hidden');
        updatePickerCategoryOptions();

        // Tutorial logic: always default to expanded when picker activates
        if (!tutorialHasBeenShown) {
            tutorialHasBeenShown = true;
            tutorialStep = 0;
        }
        showTutorialOverlay();
        // Hide picker hints since tutorial shows them
        if (pickerPanHint) pickerPanHint.classList.add('hidden');
        if (pickerCtrlHint) pickerCtrlHint.classList.add('hidden');

        if (pickedColors.length > 0) {
            setStatus('Picker mode: ' + pickedColors.length + ' colors already selected. Click to add more.');
        } else {
            setStatus('Picker mode: click to pick colors');
        }
    } else {
        btn.classList.remove('active');
        if (sidebarPickBtn) {
            sidebarPickBtn.classList.remove('active');
            // Text will be restored by updatePickColorsButtonState below or by caller
            if (pickedColors.length > 0) {
                sidebarPickBtn.innerHTML = '<span class="step-circle step-circle-btn">2</span> <span>🎯</span> Edit Origin Colors';
            } else {
                sidebarPickBtn.innerHTML = '<span class="step-circle step-circle-btn">2</span> <span>🎯</span> Pick Colors';
            }
        }
        canvasInner.classList.remove('picking-mode');
        canvasWrapper.classList.remove('picking-mode');
        categorySelector.classList.add('hidden');
        // Hide picker hints
        if (pickerPanHint) pickerPanHint.classList.add('hidden');
        if (pickerCtrlHint) pickerCtrlHint.classList.add('hidden');
        // Hide markers — they'll be rebuilt from pickedPositions when picker is re-engaged
        document.querySelectorAll('.picker-marker').forEach(m => m.remove());

        // Hide the overlay sub-elements (swatch list, apply, clear) when picker is deactivated
        // They'll come back when the picker is re-engaged
        document.getElementById('pickerSwatchesList').classList.add('hidden');
        document.getElementById('pickerApplyBtn').classList.add('hidden');
        document.getElementById('pickerClearBtn').classList.add('hidden');

        // Hide tutorial overlay and tab when picker deactivated
        hideTutorialOverlay();
        const tutorialTab = document.getElementById('tutorialTab');
        if (tutorialTab) tutorialTab.classList.add('hidden');
    }

    // Only update the overlay list if picker is active (otherwise we just hid everything above)
    if (pickerMode) {
        updatePickerOverlay();
    }
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

    // Add Accent 1, 2, 3, etc. — always show at least 4 accent options
    // so the user can always pick new categories even if current targetCount is low
    const maxAccent = Math.max(targetCount, 5);
    for (let i = 1; i < maxAccent; i++) {
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

    // Render newest colors first (reverse order) so most recent appears at top
    for (let i = pickedColors.length - 1; i >= 0; i--) {
        const color = pickedColors[i];
        const item = document.createElement('div');
        item.className = 'picker-swatch-item';
        item.dataset.colorIndex = i; // Store real index for cloning in tutorial

        // Category label on the left (B, L, A1, A2, etc.) - CLICKABLE to cycle
        const categoryLabel = document.createElement('div');
        categoryLabel.className = 'picker-swatch-category clickable';
        const cat = pickedCategories[i] !== undefined ? pickedCategories[i] : CATEGORY_BACKGROUND;
        categoryLabel.textContent = getTargetCategoryLabel(cat, true);
        categoryLabel.title = 'Click to change category';
        categoryLabel.onclick = ((idx) => (e) => {
            e.stopPropagation();
            cyclePickedColorCategory(idx);
        })(i);

        const colorDiv = document.createElement('div');
        colorDiv.className = 'picker-swatch-color';
        colorDiv.style.background = rgbToHex(...color);

        const infoDiv = document.createElement('div');
        infoDiv.className = 'picker-swatch-info';
        infoDiv.innerHTML = rgbToHex(...color);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'picker-swatch-delete';
        deleteBtn.innerHTML = '×';
        deleteBtn.onclick = ((idx) => (e) => {
            e.stopPropagation();
            removePickedColor(idx);
        })(i);

        item.appendChild(categoryLabel);
        item.appendChild(colorDiv);
        item.appendChild(infoDiv);
        item.appendChild(deleteBtn);
        list.appendChild(item);
    }
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

// Returns whichever canvas is currently visible to the user
function getVisibleCanvas() {
    if (webglCanvas && webglCanvas.style.display !== 'none' && webglCanvas.parentNode) return webglCanvas;
    if (displayCanvas && displayCanvas.style.display !== 'none') return displayCanvas;
    return canvas;
}

function getCanvasCoords(e) {
    // Use whichever canvas is currently visible for positioning
    const visibleCanvas = getVisibleCanvas();
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

    // Markers are children of canvasInner (which Panzoom transforms).
    // Position them in canvasInner's LOCAL coordinate space using the
    // display canvas CSS size (not screen/getBoundingClientRect size).
    const vc = getVisibleCanvas();
    const localWidth = parseFloat(vc.style.width) || vc.offsetWidth;
    const localHeight = parseFloat(vc.style.height) || vc.offsetHeight;
    const displayX = (pos.x / canvas.width) * localWidth;
    const displayY = (pos.y / canvas.height) * localHeight;

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
    // Position markers in canvasInner's LOCAL coordinate space.
    // Use the display canvas CSS size (not screen rect) since Panzoom transforms canvasInner.
    const vc = getVisibleCanvas();
    const localWidth = parseFloat(vc.style.width) || vc.offsetWidth;
    const localHeight = parseFloat(vc.style.height) || vc.offsetHeight;
    pickedPositions.forEach((pos, i) => {
        const marker = document.querySelector(`.picker-marker[data-index="${i}"]`);
        if (marker) {
            const displayX = (pos.x / canvas.width) * localWidth;
            const displayY = (pos.y / canvas.height) * localHeight;
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

    // Determine if we're re-picking after already having advanced past "Look's Good!"
    const alreadyAdvanced = (uiStage === 'target-selection' || uiStage === 'complete');
    const oldTargetCount = alreadyAdvanced ? targetPalette.length : 0;
    const oldTargetPalette = alreadyAdvanced ? [...targetPalette] : [];

    originalPalette = newOriginalPalette;
    colorPercentages = pcts;
    originToColumn = newOriginToColumn;
    columnBypass = []; // Reset bypass states

    if (alreadyAdvanced) {
        // Preserve existing targets — grow or shrink to match new target count
        const newTargetPalette = [];
        for (let catIdx = 0; catIdx < targetCount; catIdx++) {
            if (catIdx < oldTargetCount && oldTargetPalette[catIdx] !== null) {
                newTargetPalette.push([...oldTargetPalette[catIdx]]);
            } else {
                newTargetPalette.push(null);
            }
        }
        targetPalette = newTargetPalette;
    } else {
        // First time — start blank
        targetPalette = [];
        for (let catIdx = 0; catIdx < targetCount; catIdx++) {
            targetPalette.push(null);
        }
    }

    // Default selectedSlotIndex to first slot (clamp if target count shrank)
    if (selectedSlotIndex >= targetCount) selectedSlotIndex = 0;

    // Update data attribute for responsive styling
    document.getElementById('columnMappingContainer').setAttribute('data-target-count', targetCount);

    // Remove markers — data is preserved in pickedPositions/pickedColors arrays
    // Markers will reappear when the picker is re-engaged
    document.querySelectorAll('.picker-marker').forEach(m => m.remove());
    shouldKeepPickedMarkers = false;

    // Turn off picker mode UI completely but keep markers visible
    pickerMode = false;
    const pickerBtn = document.getElementById('pickerToggleBtn');
    pickerBtn.classList.remove('active');
    const sidebarPickBtn = document.getElementById('sidebarPickColorsBtn');
    if (sidebarPickBtn) sidebarPickBtn.classList.remove('active');
    document.getElementById('canvasInner').classList.remove('picking-mode');
    document.getElementById('canvasWrapper').classList.remove('picking-mode');
    document.getElementById('pickerCategorySelector').classList.add('hidden');
    const pickerPanHint = document.getElementById('pickerPanHint');
    if (pickerPanHint) pickerPanHint.classList.add('hidden');
    const pickerCtrlHint = document.getElementById('pickerCtrlHint');
    if (pickerCtrlHint) pickerCtrlHint.classList.add('hidden');

    // Hide the picker overlay swatch list and buttons (picker is no longer engaged)
    document.getElementById('pickerSwatchesList').classList.add('hidden');
    document.getElementById('pickerApplyBtn').classList.add('hidden');
    document.getElementById('pickerClearBtn').classList.add('hidden');

    // Hide the tutorial overlay and tab entirely (picker is done)
    hideTutorialOverlay();

    // Collapse the picker instructions in the sidebar (colors have been picked)
    const pickerInstructions = document.getElementById('pickerInstructions');
    if (pickerInstructions) pickerInstructions.classList.add('hidden');

    // Update Pick Colors button to reflect that colors have been picked
    updatePickColorsButtonState(true);

    if (alreadyAdvanced) {
        // Already past "Look's Good!" — stay at current stage, just update the mapping
        // Expand the origin collapsible so user can see what changed
        const originCollapsible = document.getElementById('originCollapsible');
        if (originCollapsible) originCollapsible.setAttribute('open', '');

        renderColumnMapping();

        // Fill any new blank targets with origin colors (same logic as activateTargetSelection)
        for (let colIdx = 0; colIdx < targetCount; colIdx++) {
            if (targetPalette[colIdx] === null) {
                let foundColor = null;
                for (let i = 0; i < originCount; i++) {
                    if (originToColumn[i] === colIdx) {
                        foundColor = [...originalPalette[i]];
                        break;
                    }
                }
                targetPalette[colIdx] = foundColor || [128, 128, 128];
            }
        }
        renderColumnMapping();

        // Auto-recolor if live preview is on
        const liveToggle = document.getElementById('livePreviewToggle');
        if (liveToggle && liveToggle.checked) {
            recolorImage();
        }

        const lockedCount = lockedColors.length;
        const lockedMsg = lockedCount > 0 ? ` (${lockedCount} locked)` : '';
        setStatus('Updated origin colors' + lockedMsg + '. Targets preserved.');
    } else {
        // First time through — show Palette Mapping panel and "Look's Good!" button
        document.getElementById('paletteMappingPanel').classList.remove('hidden');
        document.getElementById('targetSelectorBtn').classList.remove('hidden');
        uiStage = 'colors-picked';
        updateProgressiveInstructions('colors-picked');
        renderColumnMapping();

        const lockedCount = lockedColors.length;
        const lockedMsg = lockedCount > 0 ? ` (${lockedCount} locked)` : '';
        setStatus('Applied ' + originCount + ' picked colors grouped by category' + lockedMsg + '. Now choose your targets.');
    }
}

// ============================================
// Target Selection Activation (Progressive UI)
// ============================================

function activateTargetSelection() {
    // If picker/tutorial is still active, deactivate it cleanly first
    if (pickerMode) {
        togglePickerMode(); // This hides tutorial, markers, picker sub-elements
    } else {
        // Even if picker isn't active, ensure tutorial is fully hidden
        hideTutorialOverlay();
        const tutorialTab = document.getElementById('tutorialTab');
        if (tutorialTab) tutorialTab.classList.add('hidden');
    }

    uiStage = 'target-selection';
    updateProgressiveInstructions('target-selection');

    // Hide the Target Selector button itself
    document.getElementById('targetSelectorBtn').classList.add('hidden');

    // Show Target Choice panel
    document.getElementById('targetChoicePanel').classList.remove('hidden');

    // Collapse the Color Analysis panel to save vertical space
    const colorAnalysis = document.getElementById('colorAnalysisOptionalPanel');
    if (colorAnalysis) colorAnalysis.removeAttribute('open');

    // Collapse the Origin section to save vertical space
    const originCollapsible = document.getElementById('originCollapsible');
    if (originCollapsible) originCollapsible.removeAttribute('open');

    // Hide early import from Color Analysis (full config section is now visible below)
    const earlyImport = document.getElementById('earlyImportSection');
    if (earlyImport) earlyImport.classList.add('hidden');

    // Show shuffle and live preview toggle
    document.getElementById('shuffleBtn').classList.remove('hidden');
    document.getElementById('livePreviewToggleWrapper').classList.remove('hidden');

    // Show tool legend below image preview
    const toolLegend = document.getElementById('toolLegendPanel');
    if (toolLegend) toolLegend.classList.remove('hidden');

    // Show quick harmony overlay on image — default to COLLAPSED (tab visible)
    const quickHarmonyBar = document.getElementById('quickHarmonyBar');
    if (quickHarmonyBar) {
        quickHarmonyBar.classList.remove('hidden');
        const qhPanel = document.getElementById('quickHarmonyPanel');
        const qhTab = document.getElementById('quickHarmonyTab');
        // Default collapsed: panel hidden, tab visible
        if (qhPanel) qhPanel.classList.add('hidden');
        if (qhTab) qhTab.classList.remove('hidden');
    }


    // Advanced mode: collapse instructions
    if (appMode === 'advanced') {
        const container = document.getElementById('progressiveInstructions');
        const arrow = document.getElementById('instructionsToggleArrow');
        if (container && !container.classList.contains('collapsed')) {
            container.classList.add('collapsed');
            instructionsCollapsed = true;
            if (arrow) arrow.textContent = '▸';
        }
    }

    // Update palette icon visibility for the mode
    updatePaletteIconVisibility();

    // Fill blank targets with origin colors as starting point (so user has something to work with)
    for (let colIdx = 0; colIdx < targetCount; colIdx++) {
        if (targetPalette[colIdx] === null) {
            // Find first origin mapped to this column and use its color
            let foundColor = null;
            for (let i = 0; i < originCount; i++) {
                if (originToColumn[i] === colIdx) {
                    foundColor = [...originalPalette[i]];
                    break;
                }
            }
            targetPalette[colIdx] = foundColor || [128, 128, 128];
        }
    }

    // Re-render with buttons now visible and targets populated
    renderColumnMapping();
    updateHarmonyWheel();
    updateGradientFromSelectedColor();
    renderThemesSortedByMatch();

    setStatus('Target selection active. Choose your target colors using the tools below.');
}

function revealFullUI() {
    // Called after any action that should reveal everything (import config, etc.)
    // This is a one-way escalation — once revealed, panels never re-hide
    if (uiStage === 'complete') return;
    uiStage = 'complete';
    updateProgressiveInstructions('complete');

    // Show all panels
    document.getElementById('sidebar').classList.remove('hidden');
    document.getElementById('colorAnalysisPanel').classList.remove('hidden');
    document.getElementById('colorAnalysisOptionalPanel').classList.remove('hidden');
    document.getElementById('paletteMappingPanel').classList.remove('hidden');
    document.getElementById('targetChoicePanel').classList.remove('hidden');

    // Hide the Target Selector button (no longer needed)
    document.getElementById('targetSelectorBtn').classList.add('hidden');

    // Don't force origin collapsible open — leave it as-is (collapsed by activateTargetSelection)

    // Hide early import from Color Analysis (full config section is now visible below)
    const earlyImport = document.getElementById('earlyImportSection');
    if (earlyImport) earlyImport.classList.add('hidden');

    // Collapse the picker instructions (colors have already been picked or loaded)
    const pickerInstructions = document.getElementById('pickerInstructions');
    if (pickerInstructions) pickerInstructions.classList.add('hidden');

    // Update Pick Colors button to reflect that colors have been picked
    updatePickColorsButtonState(true);

    // Show shuffle row and live preview toggle
    document.getElementById('shuffleBtn').classList.remove('hidden');
    document.getElementById('livePreviewToggleWrapper').classList.remove('hidden');

    // Show tool legend below image preview
    const toolLegend = document.getElementById('toolLegendPanel');
    if (toolLegend) toolLegend.classList.remove('hidden');

    // Show quick harmony overlay on image — default collapsed
    const quickHarmonyBar = document.getElementById('quickHarmonyBar');
    if (quickHarmonyBar) {
        quickHarmonyBar.classList.remove('hidden');
        const qhPanel = document.getElementById('quickHarmonyPanel');
        const qhTab = document.getElementById('quickHarmonyTab');
        if (qhPanel) qhPanel.classList.add('hidden');
        if (qhTab) qhTab.classList.remove('hidden');
    }

    // Advanced mode: collapse instructions
    if (appMode === 'advanced') {
        const instrContainer = document.getElementById('progressiveInstructions');
        const instrArrow = document.getElementById('instructionsToggleArrow');
        if (instrContainer && !instrContainer.classList.contains('collapsed')) {
            instrContainer.classList.add('collapsed');
            instructionsCollapsed = true;
            if (instrArrow) instrArrow.textContent = '▸';
        }
    }

    // Update palette icon visibility
    updatePaletteIconVisibility();

    // Un-gray the recolored strip
    document.getElementById('recoloredStrip').classList.remove('grayed-out');

    // Ensure workspace is in image-loaded layout
    const workspace = document.getElementById('workspace');
    workspace.classList.remove('centered-initial');
    workspace.classList.add('image-loaded');

    // Re-center overlays now that panels have been revealed (layout may have changed)
    requestAnimationFrame(updateStickyOverlays);
}

// ============================================
// Zoom and Pan Functions
// ============================================

function resetZoom() {
    debugLog(`[zoom-reset] from ${zoomLevel.toFixed(2)} → 1.00`);
    zoomLevel = 1;
    panX = 0;
    panY = 0;
    isVerticallyCropped = false;
    if (panzoomInstance) {
        panzoomInstance.pan(0, 0, { animate: false, force: true });
    }
    // Cancel pending debounced render and any deferred 1x cleanup
    if (zoomRenderTimeout) { clearTimeout(zoomRenderTimeout); zoomRenderTimeout = null; }
    if (window._zoomResetTimeout) { clearTimeout(window._zoomResetTimeout); window._zoomResetTimeout = null; }

    const wrapper = document.getElementById('canvasWrapper');
    const canvasArea = document.getElementById('canvasArea');

    // Clear ALL manual resize overrides on canvasArea and wrapper
    // (same as autoSizeCanvasArea — let CSS flex + aspect-ratio handle sizing)
    canvasArea.style.width = '';
    canvasArea.style.height = '';
    canvasArea.style.flex = '';

    // Restore wrapper aspect-ratio to match the image's natural proportions
    if (canvas && canvas.width > 0 && canvas.height > 0) {
        wrapper.style.aspectRatio = `${canvas.width} / ${canvas.height}`;
    }
    wrapper.style.height = '';
    wrapper.style.minHeight = '';

    wrapper.classList.remove('zoomed', 'can-pan');

    // Re-render at zoom 1 with the restored dimensions
    renderAtCurrentZoom();
    updateMarkers();
    updateZoomDisplay();
    updateStickyOverlays();
}

function setZoomFromSlider(value) {
    let newZoom = Math.max(1, Math.min(4, parseInt(value) / 100));
    const oldZoom = zoomLevel;
    debugLog(`[zoom-slider] ${oldZoom.toFixed(2)} → ${newZoom.toFixed(2)}`);
    const wrapper = document.getElementById('canvasWrapper');

    // Snap to exactly 1x if close
    if (newZoom < 1.02) newZoom = 1;

    // Ensure wrapper is in zoomed layout mode before any pan math
    if (!wrapper.classList.contains('zoomed') && newZoom > 1) {
        const rect = wrapper.getBoundingClientRect();
        wrapper.style.height = rect.height + 'px';
        wrapper.classList.add('zoomed', 'can-pan');
    }

    if (newZoom <= 1) {
        // Reset to 1x — keep layout stable, defer teardown
        zoomLevel = 1;
        panX = 0;
        panY = 0;
        if (panzoomInstance) {
            panzoomInstance.pan(0, 0, { animate: false, force: true });
        }
    } else {

        zoomLevel = newZoom;

        // Zoom-to-center-of-viewport math:
        const wrapperRect = wrapper.getBoundingClientRect();
        const centerX = wrapperRect.width / 2;
        const centerY = wrapperRect.height / 2;

        // Current pan from Panzoom
        const currentPan = panzoomInstance.getPan();

        // Keep the viewport center point stable
        const ratio = newZoom / oldZoom;
        const newPanX = centerX - (centerX - currentPan.x) * ratio;
        const newPanY = centerY - (centerY - currentPan.y) * ratio;

        panX = newPanX;
        panY = newPanY;
        panzoomInstance.pan(newPanX, newPanY, { animate: false, force: true });
    }

    updateDisplayCanvas();
    constrainPan();
    updateMarkers();
    updateZoomDisplay();
    updateStickyOverlays();

    // Deferred 1x layout cleanup (same as wheel handler)
    if (zoomLevel <= 1) {
        if (window._zoomResetTimeout) clearTimeout(window._zoomResetTimeout);
        window._zoomResetTimeout = setTimeout(() => {
            if (zoomLevel <= 1) {
                if (zoomRenderTimeout) { clearTimeout(zoomRenderTimeout); zoomRenderTimeout = null; }
                if (!isVerticallyCropped) {
                    wrapper.style.height = '';
                    wrapper.classList.remove('zoomed', 'can-pan');
                }
                renderAtCurrentZoom();
                updateMarkers();
            }
        }, 200);
    } else {
        if (window._zoomResetTimeout) { clearTimeout(window._zoomResetTimeout); window._zoomResetTimeout = null; }
    }
}

// Clamp pan so at least 20% of the canvas stays visible in the wrapper.
// Called after any zoom or pan change.
let _constrainingPan = false;
function constrainPan() {
    if (_constrainingPan) return; // Prevent recursive calls from panzoomchange
    if (!panzoomInstance || (zoomLevel <= 1 && !isVerticallyCropped)) return;

    const wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) return;
    const wrapperRect = wrapper.getBoundingClientRect();
    const wW = wrapperRect.width;
    const wH = wrapperRect.height;

    // Current canvas display size
    const vc = getVisibleCanvas();
    const cW = parseFloat(vc.style.width) || vc.offsetWidth;
    const cH = parseFloat(vc.style.height) || vc.offsetHeight;

    // During temporary CSS scale, actual visual size may differ from style.width
    // Use the zoomed size directly
    const canvasW = Math.max(cW, wW * zoomLevel / lastRenderedZoom);
    const canvasH = Math.max(cH, wH * zoomLevel / lastRenderedZoom);

    const currentPan = panzoomInstance.getPan();
    let px = currentPan.x;
    let py = currentPan.y;

    // Clamp so image edges can never go past the viewing window edges.
    // - Image left edge must not go right of wrapper left edge (px <= 0)
    // - Image right edge must not go left of wrapper right edge (px + canvasW >= wW)
    // - Same logic for vertical axis
    // If the image is smaller than the wrapper in a dimension, center it instead.
    let minX, maxX, minY, maxY;
    if (canvasW >= wW) {
        minX = wW - canvasW;  // most it can go left (negative)
        maxX = 0;              // can't go right of origin
    } else {
        // Image narrower than wrapper — center it, no horizontal panning
        const centered = (wW - canvasW) / 2;
        minX = centered;
        maxX = centered;
    }
    if (canvasH >= wH) {
        minY = wH - canvasH;  // most it can go up (negative)
        maxY = 0;              // can't go below origin
    } else {
        // Image shorter than wrapper — center it, no vertical panning
        const centered = (wH - canvasH) / 2;
        minY = centered;
        maxY = centered;
    }

    let clamped = false;
    if (px < minX) { px = minX; clamped = true; }
    if (px > maxX) { px = maxX; clamped = true; }
    if (py < minY) { py = minY; clamped = true; }
    if (py > maxY) { py = maxY; clamped = true; }

    if (clamped) {
        _constrainingPan = true;
        panX = px;
        panY = py;
        panzoomInstance.pan(px, py, { animate: false, force: true });
        _constrainingPan = false;
    }
}

// Legacy compatibility — Panzoom handles transforms
function updateCanvasTransform(zoomChanged = false) {
    // Panzoom manages the CSS transform on canvasInner.
    // This function is kept for any remaining call sites.
    const wrapper = document.getElementById('canvasWrapper');
    if (zoomLevel > 1 || isVerticallyCropped) {
        wrapper.classList.add('zoomed', 'can-pan');
    } else {
        wrapper.classList.remove('zoomed', 'can-pan');
    }
    if (zoomChanged) {
        updateDisplayCanvas();
    }
    updateMarkers();
}

// Display canvas rendering — WebGL direct or 2D fallback
// Track base dimensions for zoom calculations
let baseDisplayWidth = 0;
let baseDisplayHeight = 0;
let lastRenderedZoom = 1;
let zoomRenderTimeout = null;

function updateDisplayCanvas() {
    if (!canvas || !useHighQualityDisplay) return;
    // Need either imageData (CPU path) or WebGL uniforms (GPU path)
    if (!imageData && !_lastWebGLRenderType) return;

    const canvasInner = document.getElementById('canvasInner');
    const wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) return;

    const wrapperRect = wrapper.getBoundingClientRect();
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;
    if (imgWidth === 0 || imgHeight === 0) return;
    const aspectRatio = imgWidth / imgHeight;

    const baseWidth = wrapperRect.width;
    const baseHeight = baseWidth / aspectRatio;

    if (zoomLevel === 1 && !isVerticallyCropped) {
        wrapper.style.height = '';
        baseDisplayWidth = baseWidth;
        baseDisplayHeight = baseHeight;
    } else {
        if (!wrapper.style.height || wrapper.style.height === '') {
            wrapper.style.height = wrapperRect.height + 'px';
        }
    }

    // If WebGL is active with cached uniforms, just update CSS sizing.
    // The pixel buffer is always at full image resolution — no re-render needed on zoom.
    if (webglInitialized && _lastWebGLRenderType) {
        const displayWidth = Math.max(1, Math.round(baseWidth * zoomLevel));
        const displayHeight = Math.max(1, Math.round(baseHeight * zoomLevel));
        webglCanvas.style.width = displayWidth + 'px';
        webglCanvas.style.height = displayHeight + 'px';
        webglCanvas.style.transform = 'scale(1)';
        webglCanvas.style.transformOrigin = '0 0';

        baseDisplayWidth = displayWidth;
        baseDisplayHeight = displayHeight;
        lastRenderedZoom = zoomLevel;

        // Show WebGL canvas, hide CPU canvases
        if (canvasInner && !webglCanvas.parentNode) {
            canvasInner.insertBefore(webglCanvas, canvas);
        }
        webglCanvas.style.display = 'block';
        displayCanvas.style.display = 'none';
        canvas.style.display = 'none';
    } else {
        // CPU fallback path — use 2D context
        const cssScale = zoomLevel / lastRenderedZoom;
        displayCanvas.style.transform = `scale(${cssScale})`;
        displayCanvas.style.transformOrigin = '0 0';

        if (zoomRenderTimeout) clearTimeout(zoomRenderTimeout);
        zoomRenderTimeout = setTimeout(() => {
            renderAtCurrentZoom();
        }, 150);

        // Show 2D display canvas, hide WebGL + original
        if (canvasInner && !displayCanvas.parentNode) {
            canvasInner.insertBefore(displayCanvas, canvas);
        }
        displayCanvas.style.display = 'block';
        if (webglCanvas) webglCanvas.style.display = 'none';
        canvas.style.display = 'none';
    }
}

// Render at the actual zoomed pixel size for crisp, pixel-perfect display.
// WebGL path: pixel buffer is already full-res — just update CSS sizing (no shader re-render).
// CPU fallback: draws hidden canvas at full image resolution, CSS handles display scaling.
function renderAtCurrentZoom() {
    if (!canvas) return;

    const wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) return;
    const wrapperRect = wrapper.getBoundingClientRect();
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;
    if (imgWidth === 0 || imgHeight === 0) return;
    const aspectRatio = imgWidth / imgHeight;

    const baseWidth = wrapperRect.width;
    const baseHeight = baseWidth / aspectRatio;
    if (baseWidth < 1 || baseHeight < 1) return;

    const displayWidth = Math.max(1, Math.round(baseWidth * zoomLevel));
    const displayHeight = Math.max(1, Math.round(baseHeight * zoomLevel));

    // WebGL path: pixel buffer is already at full image resolution.
    // Just update CSS sizing — no need to re-render the shader.
    if (webglInitialized && _lastWebGLRenderType) {
        debugLog(`[renderAtZoom] css=${displayWidth}x${displayHeight}, wrapper=${baseWidth.toFixed(0)}, zoom=${zoomLevel}`);
        webglCanvas.style.width = displayWidth + 'px';
        webglCanvas.style.height = displayHeight + 'px';
        webglCanvas.style.transform = 'scale(1)';
        webglCanvas.style.transformOrigin = '0 0';

        baseDisplayWidth = displayWidth;
        baseDisplayHeight = displayHeight;
        lastRenderedZoom = zoomLevel;

        // Show WebGL canvas, hide CPU canvases
        const canvasInner = document.getElementById('canvasInner');
        if (canvasInner && !webglCanvas.parentNode) {
            canvasInner.insertBefore(webglCanvas, canvas);
        }
        webglCanvas.style.display = 'block';
        displayCanvas.style.display = 'none';
        canvas.style.display = 'none';
        return;
    }

    // CPU fallback path: render at full image resolution, CSS handles display scaling.
    // This prevents downsampling artifacts just like the WebGL path.
    if (!imageData) return;
    debugLog(`[renderAtZoom-CPU] image=${imgWidth}x${imgHeight}, css=${displayWidth}x${displayHeight}, zoom=${zoomLevel}`);

    if (!displayCtx) {
        displayCtx = displayCanvas.getContext('2d', { alpha: false });
    }

    // Set pixel buffer to full image resolution (matching WebGL approach)
    displayCanvas.width = imgWidth;
    displayCanvas.height = imgHeight;

    displayCtx.imageSmoothingEnabled = true;
    displayCtx.imageSmoothingQuality = 'high';
    displayCtx.drawImage(canvas, 0, 0, imgWidth, imgHeight);

    // CSS handles display scaling
    displayCanvas.style.width = displayWidth + 'px';
    displayCanvas.style.height = displayHeight + 'px';
    displayCanvas.style.transform = 'scale(1)';
    displayCanvas.style.transformOrigin = '0 0';

    baseDisplayWidth = displayWidth;
    baseDisplayHeight = displayHeight;
    lastRenderedZoom = zoomLevel;

    // Show 2D display canvas, hide WebGL + original
    const canvasInner = document.getElementById('canvasInner');
    if (canvasInner && !displayCanvas.parentNode) {
        canvasInner.insertBefore(displayCanvas, canvas);
    }
    displayCanvas.style.display = 'block';
    if (webglCanvas) webglCanvas.style.display = 'none';
    canvas.style.display = 'none';
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
let isVerticallyCropped = false; // true when wrapper is shorter than image natural height

function initResizeHandles() {
    const canvasArea = document.getElementById('canvasArea');
    const resizeRight = document.getElementById('resizeRight');
    const resizeBottom = document.getElementById('resizeBottom');
    const resizeCorner = document.getElementById('resizeCorner');

    if (!resizeRight || !resizeBottom || !resizeCorner) return;

    const startResize = (type) => (e) => {
        e.preventDefault(); // Prevent text selection during drag
        isResizing = true;
        resizeType = type;
        resizeStartX = e.clientX;
        resizeStartY = e.clientY;
        resizeStartWidth = canvasArea.offsetWidth;
        resizeStartHeight = canvasArea.offsetHeight;
        document.body.style.cursor = type === 'corner' ? 'nwse-resize' : (type === 'right' ? 'ew-resize' : 'ns-resize');
        document.body.style.userSelect = 'none';
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

        // Allow height adjustment — shrinking clips the image (overflow hidden + panning)
        if (resizeType === 'bottom' || resizeType === 'corner') {
            const minH = 150; // minimum box height
            const newHeight = Math.max(minH, resizeStartHeight + dy);
            canvasArea.style.height = newHeight + 'px';
            // Override aspect-ratio so the explicit height takes effect
            const wrapper = document.getElementById('canvasWrapper');
            wrapper.style.aspectRatio = 'auto';
            wrapper.style.height = '100%';
        }
        // Don't update display during drag - too slow
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizeType = null;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.querySelectorAll('.resize-handle').forEach(h => h.classList.remove('active'));
            // Check if vertically cropped (wrapper shorter than image natural height)
            checkVerticallyCropped();
            // Re-render after resize is complete
            updateDisplayCanvas();
            updateMarkers();
            updateStickyOverlays();
        }
    });

    // Use ResizeObserver on canvas wrapper to re-render when size changes
    // This fires for window resize, flex layout changes, and manual resize handle drags
    const wrapperObs = document.getElementById('canvasWrapper');
    if (wrapperObs && typeof ResizeObserver !== 'undefined') {
        let resizeObserverTimeout = null;
        const ro = new ResizeObserver(() => {
            // Debounce to avoid excessive re-renders during drag
            if (resizeObserverTimeout) clearTimeout(resizeObserverTimeout);
            resizeObserverTimeout = setTimeout(() => {
                if (canvas && imageData) {
                    updateDisplayCanvas();
                    updateMarkers();
                }
                updateStickyOverlays();
            }, 50);
        });
        ro.observe(wrapperObs);
    } else {
        // Fallback for older browsers
        window.addEventListener('resize', () => {
            updateDisplayCanvas();
            updateMarkers();
            updateStickyOverlays();
        });
    }
}

function checkVerticallyCropped() {
    if (!canvas || !canvas.width || !canvas.height) {
        isVerticallyCropped = false;
        return;
    }
    const wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) { isVerticallyCropped = false; return; }
    const wrapperRect = wrapper.getBoundingClientRect();
    // Natural image height at current wrapper width (unzoomed)
    const naturalHeight = wrapperRect.width / (canvas.width / canvas.height);
    // Cropped if wrapper is shorter than the image's natural display height (with small tolerance)
    isVerticallyCropped = (wrapperRect.height < naturalHeight - 2);
    if (isVerticallyCropped) {
        wrapper.classList.add('zoomed', 'can-pan');
    } else if (zoomLevel <= 1) {
        wrapper.classList.remove('zoomed', 'can-pan');
    }
}

function autoSizeCanvasArea() {
    // Auto-size the canvas area to fit the image when not zoomed
    if (!canvas || !originalImageData || zoomLevel > 1) return;

    const canvasArea = document.getElementById('canvasArea');
    const wrapper = document.getElementById('canvasWrapper');

    // Clear any manual resize overrides — let CSS flex + aspect-ratio handle sizing
    canvasArea.style.width = '';
    canvasArea.style.height = '';
    canvasArea.style.flex = '';

    // Restore aspect-ratio if it was overridden by manual resize
    if (canvas.width > 0 && canvas.height > 0) {
        wrapper.style.aspectRatio = `${canvas.width} / ${canvas.height}`;
        wrapper.style.height = '';
        wrapper.style.minHeight = '';
    }
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
    // Apply theme immediately on click
    applyTheme();
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
    if (!baseColor) {
        setStatus('Select a target color first');
        return;
    }
    const [h, s, l] = rgbToHsl(...baseColor);
    const harmonyType = _currentHarmonyType;

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

    targetPalette.forEach((color, i) => {
        if (columnBypass[i]) return;  // Respect locked/bypassed columns
        // If target is blank, use default saturation/lightness
        const origS = color ? rgbToHsl(...color)[1] : 60;
        const origL = color ? rgbToHsl(...color)[2] : 50;
        targetPalette[i] = hslToRgb(newHues[i], origS, origL);
    });

    renderColumnMapping();
    setStatus('Adjusted colors to ' + harmonyType + ' harmony');
    updateHarmonyWheel();
    autoRecolorImage();
}

function updateHarmonyBaseHueDisplay(value) {
    document.getElementById('harmonyBaseHueValue').textContent = value + '°';
    updateHarmonyWheel();
}

// Harmony Tutorial slide navigation
let _harmonyTutorialSlide = 0;
function harmonyTutorialNav(dir) {
    const slides = document.querySelectorAll('#harmonyTutorialSlides .harmony-tutorial-slide');
    if (!slides.length) return;
    _harmonyTutorialSlide = Math.max(0, Math.min(slides.length - 1, _harmonyTutorialSlide + dir));
    slides.forEach((s, i) => s.classList.toggle('active', i === _harmonyTutorialSlide));
    document.getElementById('harmonyTutorialPrev').disabled = _harmonyTutorialSlide === 0;
    document.getElementById('harmonyTutorialNext').disabled = _harmonyTutorialSlide === slides.length - 1;
}

// Generate palette from the harmony controls (base hue, sat range, lit range)
function generateHarmonyFromControls() {
    if (targetPalette.length === 0) {
        setStatus('Load an image first');
        return;
    }

    const harmonyType = _currentHarmonyType;
    const n = targetPalette.length;

    const baseHue = parseInt(document.getElementById('harmonyBaseHue').value);
    const satMin = parseInt(document.getElementById('harmonySatMin').value);
    const satMax = parseInt(document.getElementById('harmonySatMax').value);
    const litMin = parseInt(document.getElementById('harmonyLitMin').value);
    const litMax = parseInt(document.getElementById('harmonyLitMax').value);

    let hues = [];
    switch (harmonyType) {
        case 'complementary':
            hues = Array(n).fill(0).map((_, i) => i % 2 === 0 ? baseHue : (baseHue + 180) % 360);
            break;
        case 'analogous':
            const spread = 30;
            hues = Array(n).fill(0).map((_, i) => (baseHue + (i - Math.floor(n/2)) * spread + 360) % 360);
            break;
        case 'triadic':
            hues = Array(n).fill(0).map((_, i) => (baseHue + (i % 3) * 120) % 360);
            break;
        case 'split':
            hues = Array(n).fill(0).map((_, i) => {
                const angles = [0, 150, 210];
                return (baseHue + angles[i % 3]) % 360;
            });
            break;
        case 'tetradic':
            hues = Array(n).fill(0).map((_, i) => (baseHue + (i % 4) * 90) % 360);
            break;
    }

    targetPalette.forEach((color, i) => {
        if (columnBypass[i]) return;
        const effectiveSatMin = Math.min(satMin, satMax);
        const effectiveSatMax = Math.max(satMin, satMax);
        const effectiveLitMin = Math.min(litMin, litMax);
        const effectiveLitMax = Math.max(litMin, litMax);
        const sat = effectiveSatMin + Math.random() * (effectiveSatMax - effectiveSatMin);
        const lit = effectiveLitMin + Math.random() * (effectiveLitMax - effectiveLitMin);
        targetPalette[i] = hslToRgb(hues[i], sat, lit);
    });

    renderColumnMapping();
    setStatus('Generated ' + harmonyType + ' harmony (base hue ' + baseHue + '°)');
    updateHarmonyWheel();
    autoRecolorImage();
}

function randomizeHarmony() {
    if (targetPalette.length === 0) {
        setStatus('Load an image first');
        return;
    }

    const harmonyType = _currentHarmonyType;
    const n = targetPalette.length;

    // Random base hue - also sync the slider
    const baseHue = Math.random() * 360;
    const hueSlider = document.getElementById('harmonyBaseHue');
    if (hueSlider) {
        hueSlider.value = Math.round(baseHue);
        updateHarmonyBaseHueDisplay(Math.round(baseHue));
    }

    // Generate hues from the selected harmony model
    let hues = [];
    switch (harmonyType) {
        case 'complementary':
            hues = Array(n).fill(0).map((_, i) => {
                return i % 2 === 0 ? baseHue : (baseHue + 180) % 360;
            });
            break;
        case 'analogous':
            const spread = 30;
            hues = Array(n).fill(0).map((_, i) => {
                const offset = (i - Math.floor(n / 2)) * spread;
                return (baseHue + offset + 360) % 360;
            });
            break;
        case 'triadic':
            hues = Array(n).fill(0).map((_, i) => {
                return (baseHue + (i % 3) * 120) % 360;
            });
            break;
        case 'split':
            hues = Array(n).fill(0).map((_, i) => {
                const angles = [0, 150, 210];
                return (baseHue + angles[i % 3]) % 360;
            });
            break;
        case 'tetradic':
            hues = Array(n).fill(0).map((_, i) => {
                return (baseHue + (i % 4) * 90) % 360;
            });
            break;
    }

    // Generate varied but pleasing saturation/lightness per slot
    targetPalette.forEach((color, i) => {
        if (columnBypass[i]) return;       // respect locked/bypassed columns

        // Vary saturation 45–90, lightness 35–65 for a balanced palette
        const sat = 45 + Math.random() * 45;
        const lit = 35 + Math.random() * 30;
        targetPalette[i] = hslToRgb(hues[i], sat, lit);
    });

    renderColumnMapping();
    setStatus('Randomized ' + harmonyType + ' harmony (base hue ' + Math.round(baseHue) + '°)');
    updateHarmonyWheel();
    autoRecolorImage();
}

// ============================================
// Harmony Mode Toggle (Lock a Harmony / Lock a Color)
// ============================================

function setHarmonyMode(mode) {
    harmonyMode = mode;
    const isColor = mode === 'color';

    // Sidebar toggle
    document.getElementById('harmonyModeSwitch').classList.toggle('color-mode', isColor);
    document.getElementById('harmonyModeHarmonyLabel').classList.toggle('active', !isColor);
    document.getElementById('harmonyModeColorLabel').classList.toggle('active', isColor);
    document.getElementById('harmonyPanelHarmony').classList.toggle('hidden', isColor);
    document.getElementById('harmonyPanelColor').classList.toggle('hidden', !isColor);

    // Quick bar toggle
    const qSwitch = document.getElementById('quickModeSwitch');
    const qHLabel = document.getElementById('quickModeHarmonyLabel');
    const qCLabel = document.getElementById('quickModeColorLabel');
    const qHControls = document.getElementById('quickHarmonyControls');
    const qCControls = document.getElementById('quickColorControls');
    if (qSwitch) qSwitch.classList.toggle('color-mode', isColor);
    if (qHLabel) qHLabel.classList.toggle('active', !isColor);
    if (qCLabel) qCLabel.classList.toggle('active', isColor);
    if (qHControls) qHControls.classList.toggle('hidden', isColor);
    if (qCControls) qCControls.classList.toggle('hidden', !isColor);

    // Hide result labels when switching
    const resultLabel = document.getElementById('lockColorResultLabel');
    if (resultLabel) resultLabel.classList.add('hidden');
    const quickResult = document.getElementById('quickHarmonyResult');
    if (quickResult) quickResult.classList.add('hidden');

    // Restore harmony type dropdowns from cached value (CSS hidden/show can reset selects in some browsers)
    const sidebar = document.getElementById('harmonyType');
    const quick = document.getElementById('quickHarmonyType');
    if (sidebar) sidebar.value = _currentHarmonyType;
    if (quick) quick.value = _currentHarmonyType;
    updateHarmonyRandomizeLabel();

    // Show nudge hint only in Harmony mode
    const nudgeHint = document.getElementById('harmonyNudgeHint');
    if (nudgeHint) nudgeHint.classList.toggle('hidden', isColor);

    // Populate the lock-a-color dropdowns when switching to color mode
    if (isColor) populateLockColorDropdowns();
}

function toggleHarmonyMode() {
    setHarmonyMode(harmonyMode === 'harmony' ? 'color' : 'harmony');
}

// Sync all harmony type selects (sidebar + quick bar)
function syncHarmonySelects(value) {
    _currentHarmonyType = value;
    const sidebar = document.getElementById('harmonyType');
    const quick = document.getElementById('quickHarmonyType');
    if (sidebar) sidebar.value = value;
    if (quick) quick.value = value;
    updateHarmonyWheel();
    updateHarmonyRandomizeLabel();
}

function updateHarmonyRandomizeLabel() {
    const label = document.getElementById('harmonyRandomizeLabel');
    if (!label) return;
    // Use cached harmony type to build the label text
    const names = { complementary: 'Complementary', analogous: 'Analogous', triadic: 'Triadic', split: 'Split-Complementary', tetradic: 'Tetradic (Square)' };
    label.textContent = names[_currentHarmonyType] || _currentHarmonyType;
}

function updateHarmonyNudgeSwatch() {
    const swatches = [
        document.getElementById('harmonyNudgeSwatch'),
        document.getElementById('quickNudgeSwatch')
    ];
    const color = targetPalette[selectedSlotIndex];
    swatches.forEach(swatch => {
        if (!swatch) return;
        if (color) {
            swatch.style.background = rgbToHex(...color);
            swatch.style.display = 'inline-block';
            swatch.title = 'Based on selected target: ' + rgbToHex(...color);
        } else {
            swatch.style.display = 'none';
        }
    });
}

// Toggle quick harmony overlay collapse/expand
function toggleQuickHarmonyCollapse() {
    const panel = document.getElementById('quickHarmonyPanel');
    const tab = document.getElementById('quickHarmonyTab');
    if (!panel || !tab) return;
    const isCollapsed = panel.classList.contains('hidden');
    if (isCollapsed) {
        panel.classList.remove('hidden');
        tab.classList.add('hidden');
    } else {
        panel.classList.add('hidden');
        tab.classList.remove('hidden');
    }
}

// Quick bar randomize (Lock a Harmony mode)
function quickRandomizeHarmony() {
    // Sync quick bar dropdown to sidebar before running
    const quickSelect = document.getElementById('quickHarmonyType');
    if (quickSelect) syncHarmonySelects(quickSelect.value);
    randomizeHarmony();
}

function quickGenerateHarmony() {
    // Sync quick bar dropdown to sidebar before running
    const quickSelect = document.getElementById('quickHarmonyType');
    if (quickSelect) syncHarmonySelects(quickSelect.value);
    generateHarmonyFromControls();
}

// Populate the "Lock a Color" swatch pickers with current target colors
function populateLockColorDropdowns() {
    const sidebarPicker = document.getElementById('lockColorSwatchPicker');
    const quickWrapper = document.getElementById('quickLockColorSelectWrapper');

    // Separate unlocked and locked/bypassed targets
    const unlocked = [];
    const lockedTargets = [];
    for (let i = 0; i < targetCount; i++) {
        if (!targetPalette[i]) continue;
        if (columnBypass[i]) {
            lockedTargets.push(i);
        } else {
            unlocked.push(i);
        }
    }

    // Gather locked/bank origin colors (from the color bank — not assigned to any target column)
    const bankOrigins = [];
    for (let i = 0; i < originToColumn.length; i++) {
        if ((originToColumn[i] === 'locked' || originToColumn[i] === 'bank') && originalPalette[i]) {
            bankOrigins.push(i);
        }
    }

    // Combine bypassed targets + bank origins into the "locked" section
    // Use string keys for bank origins: 'origin_5', numeric for target columns
    const allSelectable = [...unlocked.map(i => i), ...lockedTargets.map(i => i), ...bankOrigins.map(i => 'origin_' + i)];

    // Default selection to first selectable if current selection is invalid
    if (!allSelectable.includes(lockColorSelectedIdx) && allSelectable.length > 0) {
        lockColorSelectedIdx = allSelectable[0];
    }

    // Helper: get color for any selectable index (target column or bank origin)
    function getSelectableColor(selIdx) {
        if (typeof selIdx === 'string' && selIdx.startsWith('origin_')) {
            const oi = parseInt(selIdx.replace('origin_', ''));
            return originalPalette[oi] ? rgbToHex(...originalPalette[oi]) : '#888888';
        } else {
            // Target column - for bypassed, show the original color mapped to that column
            if (columnBypass[selIdx]) {
                for (let j = 0; j < originToColumn.length; j++) {
                    if (originToColumn[j] === selIdx && originalPalette[j]) {
                        return rgbToHex(...originalPalette[j]);
                    }
                }
            }
            return targetPalette[selIdx] ? rgbToHex(...targetPalette[selIdx]) : '#888888';
        }
    }

    // Helper: get label for any selectable index
    function getSelectableLabel(selIdx, short = false) {
        if (typeof selIdx === 'string' && selIdx.startsWith('origin_')) {
            const oi = parseInt(selIdx.replace('origin_', ''));
            return short ? ('O' + (oi + 1)) : ('Bank Color ' + (oi + 1));
        }
        return getTargetCategoryLabel(selIdx, short);
    }

    // Helper: check if a selectable index is a locked item (bypassed target or bank origin)
    function isLockedItem(selIdx) {
        if (typeof selIdx === 'string' && selIdx.startsWith('origin_')) return true;
        return lockedTargets.includes(selIdx);
    }

    // Helper: update the quick bar 🔒 button to show the selected locked color's swatch
    function updateQuickLockedBtnAppearance(selIdx) {
        if (!quickWrapper) return;
        const addBtn = quickWrapper.parentElement?.querySelector('.quick-locked-add-btn');
        if (!addBtn) return;

        if (isLockedItem(selIdx)) {
            // A locked item is selected — show its color swatch with L indicator
            const color = getSelectableColor(selIdx);
            addBtn.innerHTML = '<span class="quick-locked-L">L</span>';
            addBtn.style.background = color;
            addBtn.style.borderStyle = 'solid';
            addBtn.style.borderColor = 'var(--accent)';
            addBtn.classList.add('has-selection');
        } else {
            // An unlocked target is selected — revert to 🔒 placeholder
            addBtn.innerHTML = '🔒';
            addBtn.style.background = '';
            addBtn.style.borderStyle = 'dashed';
            addBtn.style.borderColor = '';
            addBtn.classList.remove('has-selection');
        }
    }

    // Helper to sync selection across both pickers
    function syncSelection(idx) {
        lockColorSelectedIdx = idx;
        const idxStr = String(idx);
        if (sidebarPicker) {
            sidebarPicker.querySelectorAll('.lock-color-option, .lock-color-locked-chip').forEach(el => {
                el.classList.toggle('selected', el.dataset.idx === idxStr);
            });
        }
        if (quickWrapper) {
            quickWrapper.querySelectorAll('.quick-lock-color-chip').forEach(el => {
                el.classList.toggle('selected', el.dataset.idx === idxStr);
            });
            // Also update inside the locked dropdown
            const lockedDropdown = quickWrapper.parentElement?.querySelector('.quick-locked-dropdown');
            if (lockedDropdown) {
                lockedDropdown.querySelectorAll('.lock-color-option').forEach(el => {
                    el.classList.toggle('selected', el.dataset.idx === idxStr);
                });
            }
            // Update the locked button appearance
            updateQuickLockedBtnAppearance(idx);
        }
    }

    // Combined "locked" items for the locked section: bypassed targets + bank origins
    const lockedItems = [
        ...lockedTargets.map(i => ({ selIdx: i, type: 'target' })),
        ...bankOrigins.map(i => ({ selIdx: 'origin_' + i, type: 'bank' }))
    ];

    // =============================================
    // SIDEBAR: list-style swatch picker
    // =============================================
    if (sidebarPicker) {
        sidebarPicker.innerHTML = '';

        // --- Unlocked targets: individual rows with swatch + label + hex ---
        unlocked.forEach(i => {
            const hex = rgbToHex(...targetPalette[i]);
            const label = getTargetCategoryLabel(i);

            const option = document.createElement('div');
            option.className = 'lock-color-option' + (i === lockColorSelectedIdx ? ' selected' : '');
            option.dataset.idx = String(i);
            option.onclick = () => syncSelection(i);

            const swatch = document.createElement('div');
            swatch.className = 'lock-color-swatch';
            swatch.style.background = hex;

            const info = document.createElement('div');
            info.className = 'lock-color-info';
            info.innerHTML = `${label}<br>${hex}`;

            option.appendChild(swatch);
            option.appendChild(info);
            sidebarPicker.appendChild(option);
        });

        // --- Locked/bypassed targets + bank origins: compact horizontal row ---
        if (lockedItems.length > 0) {
            const lockedSection = document.createElement('div');
            lockedSection.className = 'lock-color-locked-section';

            const lockedLbl = document.createElement('div');
            lockedLbl.className = 'lock-color-locked-label';
            lockedLbl.textContent = '🔒 Locked';
            lockedSection.appendChild(lockedLbl);

            const lockedRow = document.createElement('div');
            lockedRow.className = 'lock-color-locked-row';

            lockedItems.forEach(item => {
                const selIdx = item.selIdx;
                const displayColor = getSelectableColor(selIdx);
                const shortLabel = getSelectableLabel(selIdx, true);
                const fullLabel = getSelectableLabel(selIdx, false);

                const chip = document.createElement('div');
                chip.className = 'lock-color-locked-chip' + (selIdx === lockColorSelectedIdx ? ' selected' : '');
                chip.dataset.idx = String(selIdx);
                chip.title = fullLabel + ' — ' + displayColor + ' (locked)';
                chip.onclick = () => syncSelection(selIdx);

                const chipSwatch = document.createElement('div');
                chipSwatch.className = 'lock-color-locked-chip-swatch';
                chipSwatch.style.background = displayColor;

                const chipLabel = document.createElement('span');
                chipLabel.className = 'lock-color-locked-chip-label';
                chipLabel.textContent = shortLabel;

                chip.appendChild(chipSwatch);
                chip.appendChild(chipLabel);
                lockedRow.appendChild(chip);
            });

            lockedSection.appendChild(lockedRow);
            sidebarPicker.appendChild(lockedSection);
        }
    }

    // =============================================
    // QUICK BAR: compact chips + locked dropdown
    // =============================================
    if (quickWrapper) {
        quickWrapper.innerHTML = '';

        // --- Unlocked targets: live-updating color chips ---
        unlocked.forEach(i => {
            const hex = rgbToHex(...targetPalette[i]);
            const label = getTargetCategoryLabel(i);

            const chip = document.createElement('div');
            chip.className = 'quick-lock-color-chip' + (i === lockColorSelectedIdx ? ' selected' : '');
            chip.style.background = hex;
            chip.dataset.idx = String(i);
            chip.title = label + ' — ' + hex;
            chip.onclick = () => syncSelection(i);
            quickWrapper.appendChild(chip);
        });

        // --- Locked targets + bank origins: "🔒" dropdown button (like Add Color bank) ---
        if (lockedItems.length > 0) {
            // Remove any previous locked dropdown from parent
            const existingDropdownWrapper = quickWrapper.parentElement?.querySelector('.quick-locked-btn-wrapper');
            if (existingDropdownWrapper) existingDropdownWrapper.remove();

            const btnWrapper = document.createElement('div');
            btnWrapper.className = 'quick-locked-btn-wrapper';

            const addBtn = document.createElement('button');
            addBtn.className = 'quick-locked-add-btn';
            addBtn.innerHTML = '🔒';
            addBtn.title = 'Select a locked/bank color to base harmony on';

            const dropdown = document.createElement('div');
            dropdown.className = 'quick-locked-dropdown';

            lockedItems.forEach(item => {
                const selIdx = item.selIdx;
                const displayColor = getSelectableColor(selIdx);
                const label = getSelectableLabel(selIdx, false);

                const option = document.createElement('div');
                option.className = 'lock-color-option' + (selIdx === lockColorSelectedIdx ? ' selected' : '');
                option.dataset.idx = String(selIdx);
                option.onclick = (e) => {
                    e.stopPropagation();
                    syncSelection(selIdx);
                    dropdown.classList.remove('visible');
                };

                const swatch = document.createElement('div');
                swatch.className = 'lock-color-swatch';
                swatch.style.background = displayColor;

                const info = document.createElement('div');
                info.className = 'lock-color-info';
                info.innerHTML = `${label}<br>${displayColor}`;

                option.appendChild(swatch);
                option.appendChild(info);
                dropdown.appendChild(option);
            });

            addBtn.onclick = (e) => {
                e.stopPropagation();
                document.querySelectorAll('.quick-locked-dropdown.visible').forEach(d => d.classList.remove('visible'));
                dropdown.classList.toggle('visible');
            };

            // Close on outside click
            document.addEventListener('click', (e) => {
                if (!btnWrapper.contains(e.target)) {
                    dropdown.classList.remove('visible');
                }
            }, { once: false });

            btnWrapper.appendChild(addBtn);
            btnWrapper.appendChild(dropdown);

            // Insert after the quickWrapper's parent controls div
            quickWrapper.after(btnWrapper);

            // If a locked item is currently selected, show its swatch on the button
            if (isLockedItem(lockColorSelectedIdx)) {
                const color = getSelectableColor(lockColorSelectedIdx);
                addBtn.innerHTML = '<span class="quick-locked-L">L</span>';
                addBtn.style.background = color;
                addBtn.style.borderStyle = 'solid';
                addBtn.style.borderColor = 'var(--accent)';
                addBtn.classList.add('has-selection');
            }
        } else {
            // No locked items — remove any leftover dropdown
            const existingDropdownWrapper = quickWrapper.parentElement?.querySelector('.quick-locked-btn-wrapper');
            if (existingDropdownWrapper) existingDropdownWrapper.remove();
        }
    }
}

// "Lock a Color" randomize: lock chosen target/bank color, generate compatible harmony around it
function randomizeLockColor() {
    if (targetPalette.length === 0) {
        setStatus('Load an image first');
        return;
    }

    const selIdx = lockColorSelectedIdx;

    // Resolve the locked color: could be a target column index or a bank origin string
    let lockedColor;
    let lockedLabel;
    let lockedTargetIdx = null; // Only set if the locked color IS a target column

    if (typeof selIdx === 'string' && selIdx.startsWith('origin_')) {
        const oi = parseInt(selIdx.replace('origin_', ''));
        lockedColor = originalPalette[oi];
        lockedLabel = 'Bank Color ' + (oi + 1);
    } else {
        lockedTargetIdx = selIdx;
        lockedColor = targetPalette[selIdx];
        lockedLabel = getTargetCategoryLabel(selIdx);
    }

    if (!lockedColor) {
        setStatus('Selected color not found');
        return;
    }

    const [lockedH, lockedS, lockedL] = rgbToHsl(...lockedColor);

    // Pick a random harmony type
    const harmonyTypes = ['complementary', 'analogous', 'triadic', 'split', 'tetradic'];
    const chosenHarmony = harmonyTypes[Math.floor(Math.random() * harmonyTypes.length)];

    // Get the distinct hue offsets for this harmony type (relative to the anchor hue)
    let harmonyOffsets;
    switch (chosenHarmony) {
        case 'complementary':
            harmonyOffsets = [0, 180];
            break;
        case 'analogous':
            harmonyOffsets = [0, -30, 30, -60, 60];
            break;
        case 'triadic':
            harmonyOffsets = [0, 120, 240];
            break;
        case 'split':
            harmonyOffsets = [0, 150, 210];
            break;
        case 'tetradic':
            harmonyOffsets = [0, 90, 180, 270];
            break;
    }

    // Collect the target slot indices that will be changed (not bypassed, not the locked target)
    const changeable = [];
    for (let i = 0; i < targetPalette.length; i++) {
        if (columnBypass[i]) continue;
        if (i === lockedTargetIdx) continue;
        changeable.push(i);
    }

    // Assign each changeable slot a random harmony offset
    // If the locked color IS a target, it implicitly occupies offset 0.
    // For bank origins, offset 0 is the bank color's hue (not on the target palette).
    // Either way, distribute the OTHER offsets to changeable slots.
    // Remove offset 0 from the pool when the locked color is a target (it's already placed).
    // Keep offset 0 in the pool when it's a bank origin (a target can share the anchor hue).
    const availableOffsets = (lockedTargetIdx !== null)
        ? harmonyOffsets.filter(o => o !== 0)   // remove anchor offset; the locked target holds it
        : [...harmonyOffsets];                   // bank origin: all offsets available for targets

    changeable.forEach(i => {
        // Pick a random offset from the harmony, add slight variation for visual interest
        const baseOffset = availableOffsets[Math.floor(Math.random() * availableOffsets.length)];
        const jitter = (Math.random() - 0.5) * 10; // ±5° hue jitter for natural variation
        const newHue = (lockedH + baseOffset + jitter + 360) % 360;
        const sat = 40 + Math.random() * 45;
        const lit = 30 + Math.random() * 35;
        targetPalette[i] = hslToRgb(newHue, sat, lit);
    });

    renderColumnMapping();
    updateHarmonyWheel();
    autoRecolorImage();

    const harmonyLabel = chosenHarmony.charAt(0).toUpperCase() + chosenHarmony.slice(1);
    const statusMsg = harmonyLabel + ' harmony around ' + lockedLabel;
    setStatus(statusMsg);

    // Show the result label in sidebar
    const resultLabel = document.getElementById('lockColorResultLabel');
    if (resultLabel) {
        resultLabel.textContent = '↳ ' + harmonyLabel;
        resultLabel.classList.remove('hidden');
    }

    // Show in quick bar
    const quickResult = document.getElementById('quickHarmonyResult');
    if (quickResult) {
        quickResult.textContent = '↳ ' + harmonyLabel;
        quickResult.classList.remove('hidden');
    }
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
            if (!color) return; // Skip blank targets
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
    
    // If a bank origin is selected as the lock color, draw its reference marker on the wheel
    if (harmonyMode === 'color' && typeof lockColorSelectedIdx === 'string' && lockColorSelectedIdx.startsWith('origin_')) {
        const oi = parseInt(lockColorSelectedIdx.replace('origin_', ''));
        const bankColor = originalPalette[oi];
        if (bankColor) {
            const [bh] = rgbToHsl(...bankColor);
            const markerDistance = (outerRadius + innerRadius) / 2;
            const angle = (bh - 90) * Math.PI / 180;

            // Draw a reference line (dashed style)
            ctx2.save();
            ctx2.setLineDash([3, 3]);
            ctx2.strokeStyle = 'rgba(255,255,255,0.5)';
            ctx2.lineWidth = 1.5;
            ctx2.beginPath();
            ctx2.moveTo(centerX, centerY);
            ctx2.lineTo(
                centerX + markerDistance * Math.cos(angle),
                centerY + markerDistance * Math.sin(angle)
            );
            ctx2.stroke();
            ctx2.restore();

            // Draw the bank color marker dot
            const dotX = centerX + markerDistance * Math.cos(angle);
            const dotY = centerY + markerDistance * Math.sin(angle);
            ctx2.beginPath();
            ctx2.arc(dotX, dotY, 6, 0, Math.PI * 2);
            ctx2.fillStyle = rgbToHex(...bankColor);
            ctx2.fill();
            ctx2.strokeStyle = '#fff';
            ctx2.lineWidth = 2;
            ctx2.stroke();

            // Draw lock icon indicator (small circle with outline)
            ctx2.beginPath();
            ctx2.arc(dotX, dotY, 3, 0, Math.PI * 2);
            ctx2.fillStyle = '#fff';
            ctx2.fill();
        }
    }

    // Draw selected color in center — show locked bank color if that's the selection
    let centerColor = null;
    if (harmonyMode === 'color' && typeof lockColorSelectedIdx === 'string' && lockColorSelectedIdx.startsWith('origin_')) {
        const oi = parseInt(lockColorSelectedIdx.replace('origin_', ''));
        if (originalPalette[oi]) centerColor = originalPalette[oi];
    } else if (targetPalette.length > 0 && targetPalette[selectedSlotIndex] !== null) {
        centerColor = targetPalette[selectedSlotIndex];
    }
    if (centerColor) {
        ctx2.beginPath();
        ctx2.arc(centerX, centerY, innerRadius - 8, 0, Math.PI * 2);
        ctx2.fillStyle = rgbToHex(...centerColor);
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
        if (!color) return; // Skip blank targets
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
            // Don't call selectSlot() here — it re-renders and destroys this dot mid-drag.
            // Instead, update selectedSlotIndex directly and update dot highlights in-place.
            selectedSlotIndex = i;
            container.querySelectorAll('.harmony-dot').forEach(d => {
                d.classList.toggle('selected', parseInt(d.dataset.index) === i);
            });
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
    
    // Update target color with new hue, preserving saturation and lightness.
    // If saturation is very low (grey/near-grey), bump it up so the hue change
    // is actually visible — otherwise dragging a grey dot does nothing.
    const color = targetPalette[harmonyDragging];
    const [_, s, l] = rgbToHsl(...color);
    const effectiveS = Math.max(s, 40);
    const effectiveL = (l < 10 || l > 90) ? Math.min(Math.max(l, 30), 70) : l;
    const newRgb = hslToRgb(angle, effectiveS, effectiveL);
    targetPalette[harmonyDragging] = newRgb;
    
    // Update UI
    updateHarmonyWheel();
    renderColumnMapping();
});

document.addEventListener('mouseup', () => {
    if (harmonyDragging !== null) {
        document.querySelectorAll('.harmony-dot').forEach(d => d.classList.remove('dragging'));
        const draggedIdx = harmonyDragging;
        harmonyDragging = null;
        // Now that dragging is done, do the full re-render to sync everything
        selectSlot(draggedIdx);
        autoRecolorImage();
    }
});

// ============================================
// Utility Functions
// ============================================

function resetImage() {
    if (!originalImageData) return;

    // Reset the canvas to the original image
    ctx.putImageData(originalImageData, 0, 0);
    imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Clear WebGL render state so display falls back to showing original
    _lastWebGLRenderType = null;
    _lastSimpleUniforms = null;
    _lastRBFUniforms = null;
    _webglDirty = false;

    updateDisplayCanvas();

    // Clear all target swatch selections back to null (blank)
    for (let i = 0; i < targetPalette.length; i++) {
        targetPalette[i] = null;
    }

    // Reset all column bypass states
    columnBypass = new Array(targetCount).fill(false);

    // Reset all origin opacities if they exist
    if (typeof originOpacity !== 'undefined') {
        for (let key in originOpacity) {
            originOpacity[key] = 100;
        }
    }

    // Reset target opacities if they exist
    if (typeof targetOpacity !== 'undefined') {
        for (let key in targetOpacity) {
            targetOpacity[key] = 100;
        }
    }

    // Reset luminosity slider
    const lumSlider = document.getElementById('luminositySlider');
    const lumValue = document.getElementById('luminosityValue');
    if (lumSlider) lumSlider.value = 0;
    if (lumValue) lumValue.textContent = '0';

    // Invalidate opacity cache
    _opacityCache = null;

    // Uncheck live preview so the original image stays
    const liveToggle = document.getElementById('livePreviewToggle');
    if (liveToggle && liveToggle.checked) {
        liveToggle.checked = false;
        toggleLivePreview(false);
    }

    // Re-render the mapping to show blank targets
    renderColumnMapping();

    setStatus('Image reset to original — all target colors cleared');
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
    rawColorDistribution = [];
    originToColumn = [];
    columnBypass = [];

    // Reset picker state
    pickedColors = [];
    pickedPositions = [];
    pickedCategories = [];
    pickerMode = false;
    shouldKeepPickedMarkers = false;
    document.querySelectorAll('.picker-marker').forEach(m => m.remove());

    // Dismiss tutorial overlay and tab (they float over the canvas area)
    hideTutorialOverlay();

    // Reset Pick Colors button back to its initial "Pick Colors" text
    updatePickColorsButtonState(false);

    // Reset counts
    originCount = 5;
    targetCount = 5;
    document.getElementById('originCountDisplay').value = 5;
    document.getElementById('targetCountDisplay').value = 5;

    // Reset zoom and Panzoom
    zoomLevel = 1;
    panX = 0;
    panY = 0;
    if (panzoomInstance) {
        panzoomInstance.reset({ animate: false });
    }
    const wrapperRm = document.getElementById('canvasWrapper');
    wrapperRm.style.height = '';
    wrapperRm.classList.remove('zoomed', 'can-pan');

    // Hide UI elements
    document.getElementById('pickerOverlay').classList.add('hidden');
    document.getElementById('zoomControls').classList.add('hidden');
    document.getElementById('zoomSliderContainer').classList.add('hidden');
    document.getElementById('toleranceContainer').classList.add('hidden');
    document.getElementById('originOverflowBank').classList.remove('visible');

    // Clear mapping display
    document.getElementById('mappingColumns').innerHTML = '';
    document.getElementById('mappingTargetsRow').innerHTML = '';
    document.getElementById('colorStrip').innerHTML = '';

    // Restore tall logo
    const logoTall = document.getElementById('logoTall');
    const logoShort = document.getElementById('logoShort');
    if (logoTall) logoTall.style.display = '';
    if (logoShort) logoShort.style.display = 'none';

    // Progressive UI: reset back to initial centered state
    const workspace = document.getElementById('workspace');
    workspace.classList.remove('image-loaded');
    workspace.classList.add('centered-initial');
    document.getElementById('sidebar').classList.add('hidden');
    document.getElementById('colorAnalysisPanel').classList.remove('hidden'); // ready for next image
    document.getElementById('colorAnalysisOptionalPanel').classList.add('hidden');
    // Reset color analysis to expanded state for next image
    document.getElementById('colorAnalysisOptionalPanel').setAttribute('open', '');
    document.getElementById('paletteMappingPanel').classList.add('hidden');
    document.getElementById('targetSelectorBtn').classList.add('hidden');
    document.getElementById('targetChoicePanel').classList.add('hidden');
    document.getElementById('pickerPanHint').classList.add('hidden');
    document.getElementById('pickerCtrlHint').classList.add('hidden');
    // Hide the image action buttons and resize handles
    document.getElementById('imageButtonGroup').classList.add('hidden');
    document.querySelectorAll('.resize-handle').forEach(h => h.classList.add('hidden'));
    // Reset recolored strip to grayed-out
    document.getElementById('recoloredStrip').classList.add('grayed-out');
    // Reset shuffle and live toggle to hidden
    document.getElementById('shuffleBtn').classList.add('hidden');
    document.getElementById('livePreviewToggleWrapper').classList.add('hidden');
    // Hide tool legend and restore picker instructions and early import
    const toolLegend = document.getElementById('toolLegendPanel');
    if (toolLegend) toolLegend.classList.add('hidden');
    // Hide quick harmony overlay
    const quickHarmonyOverlay = document.getElementById('quickHarmonyBar');
    if (quickHarmonyOverlay) quickHarmonyOverlay.classList.add('hidden');
    const pickerInstructions = document.getElementById('pickerInstructions');
    if (pickerInstructions) pickerInstructions.classList.remove('hidden');
    const earlyImport = document.getElementById('earlyImportSection');
    if (earlyImport) earlyImport.classList.remove('hidden');
    uiStage = 'initial';
    updateProgressiveInstructions('initial');

    // Show upload zone again
    document.getElementById('uploadZone').classList.remove('hidden');
    canvas.style.display = 'none';
    // Hide display canvases too
    if (displayCanvas) {
        displayCanvas.style.display = 'none';
    }
    if (webglCanvas) {
        webglCanvas.style.display = 'none';
    }

    // Reset canvas area inline styles
    const canvasArea = document.getElementById('canvasArea');
    canvasArea.style.width = '';
    canvasArea.style.height = '';
    canvasArea.style.flex = '';
    // Clear aspect-ratio and wrapper styles
    const wrapperEl = document.getElementById('canvasWrapper');
    wrapperEl.style.aspectRatio = '';
    wrapperEl.style.height = '';
    wrapperEl.style.minHeight = '';

    // Reset file input so the same file can be re-selected
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';

    setStatus('Image removed. Upload a new image to start.');
}

function downloadImage() {
    if (!imageData && !_lastWebGLRenderType) {
        setStatus('No image to download');
        return;
    }
    debugLog(`[image-download] ${canvas.width}x${canvas.height}, webglDirty=${_webglDirty}`);

    // Sync WebGL pixels to CPU canvas if needed
    syncWebGLToCPU();

    // Use the current recolor history count as the suffix number
    const recolorNum = recolorHistory.length || 1;
    const baseName = originalFileName || 'image';
    const fileName = `${baseName}_Recolor_${recolorNum}.png`;

    const link = document.createElement('a');
    link.download = fileName;
    link.href = canvas.toDataURL('image/png');
    link.click();
    setStatus(`Image downloaded as ${fileName}`);
}

// ============================================
// Debug Console Logging System
// ============================================
// Always visible in advanced mode. Shows render pipeline trace.
// Collapsible header, record/stop/copy controls.

let _debugLog = [];
let _debugRenderStart = null;
let _debugRecording = false; // Default: recording off

function debugLog(message, level = 'info') {
    if (!_debugRecording) return;
    const elapsed = _debugRenderStart ? ((performance.now() - _debugRenderStart) / 1000).toFixed(3) : '—';
    _debugLog.push({ time: elapsed, message, level });
    renderDebugConsole();
}

function debugRenderStart() {
    if (!_debugRecording) return;
    // Don't clear log — accumulate across renders.
    // Log only clears on page reload or explicit Clear button.
    _debugRenderStart = performance.now();
    const imgSize = canvas ? canvas.width + 'x' + canvas.height : 'none';
    debugLog('── Render started ── algo=' + selectedAlgorithm + ', palette=' + originalPalette.length + ', image=' + imgSize + ', webgl=' + webglInitialized);
}

function debugRenderEnd(success) {
    if (!_debugRecording) return;
    const elapsed = _debugRenderStart ? ((performance.now() - _debugRenderStart) / 1000).toFixed(3) + 's' : '?';
    if (success) {
        debugLog('── Render complete ── ' + elapsed, 'success');
    } else {
        debugLog('── Render FAILED ── ' + elapsed, 'error');
    }
    _debugRenderStart = null;
}

function toggleDebugConsole() {
    const el = document.getElementById('debugConsole');
    if (el) el.classList.toggle('collapsed');
}

function toggleDebugRecording() {
    _debugRecording = !_debugRecording;
    const btn = document.getElementById('debugRecordBtn');
    if (btn) {
        btn.classList.toggle('active', _debugRecording);
        btn.title = _debugRecording ? 'Recording' : 'Paused';
    }
    if (_debugRecording) {
        debugLog('Recording started', 'info');
        // Log current state snapshot so user sees something useful immediately
        debugLog(`State: algo=${selectedAlgorithm}, webgl=${webglInitialized}, renderType=${_lastWebGLRenderType}, image=${canvas ? canvas.width + 'x' + canvas.height : 'none'}`);
        debugLog('Trigger a recolor (switch algorithm, change target, etc.) to capture render trace');
    }
}

function clearDebugLog() {
    _debugLog = [];
    renderDebugConsole();
}

function copyDebugLog() {
    const text = _debugLog.map(e => '[' + e.time + 's] ' + e.message).join('\n');
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector('.debug-btn-copy');
        if (btn) {
            const orig = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = orig; }, 800);
        }
    });
}

function renderDebugConsole() {
    const body = document.getElementById('debugConsoleBody');
    if (!body) return;
    if (_debugLog.length === 0) {
        body.innerHTML = '<div class="log-entry"><span class="log-time">[—]</span> <span class="log-info">Waiting for render...</span></div>';
        return;
    }
    body.innerHTML = _debugLog.map(entry => {
        const cls = 'log-' + entry.level;
        return '<div class="log-entry"><span class="log-time">[' + entry.time + 's]</span> <span class="' + cls + '">' +
            entry.message.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</span></div>';
    }).join('');
    body.scrollTop = body.scrollHeight;
}

// ============================================
// Eyedropper Diagnostic Probe (Full-Page Mode)
// ============================================
// When active, ALL clicks on the page are intercepted via a document-level
// capture listener + a visual overlay with crosshair cursor. The probe
// samples the element beneath the cursor and logs color, element identity,
// and CSS cascade info. For canvas pixels within the recolor image, it also
// performs the full mapping chain analysis.
// The only interactive element during probe mode is the Probe button itself.
// Scrolling still works normally (wheel events are not blocked).

let _eyedropperActive = false;
let _eyedropperOverlay = null;

function _eyedropperCaptureClick(e) {
    if (!_eyedropperActive) return;

    // Allow Probe button clicks through to toggle off
    const btn = document.getElementById('debugEyedropperBtn');
    if (btn && (e.target === btn || btn.contains(e.target))) {
        return; // Let the onclick handler fire normally
    }

    // Block ALL other clicks
    e.stopPropagation();
    e.preventDefault();

    eyedropperSample(e);
}

function toggleDebugEyedropper() {
    _eyedropperActive = !_eyedropperActive;
    const btn = document.getElementById('debugEyedropperBtn');
    if (btn) btn.classList.toggle('active', _eyedropperActive);

    if (_eyedropperActive) {
        // Expand the debug console so user can see results
        const dc = document.getElementById('debugConsole');
        if (dc && dc.classList.contains('collapsed')) dc.classList.remove('collapsed');

        // Auto-enable recording if off
        if (!_debugRecording) toggleDebugRecording();

        // Add document-level capture listeners to intercept ALL clicks
        document.addEventListener('click', _eyedropperCaptureClick, true);
        document.addEventListener('mousedown', _eyedropperBlockEvent, true);
        document.addEventListener('mouseup', _eyedropperBlockEvent, true);
        document.addEventListener('keydown', _eyedropperKeyHandler);

        // Set crosshair cursor on entire page
        document.body.classList.add('probe-active');

        // Add visual overlay border indicator (pointer-events: none so
        // scrolling works normally through it)
        if (!_eyedropperOverlay) {
            _eyedropperOverlay = document.createElement('div');
            _eyedropperOverlay.id = 'eyedropperOverlay';
        }
        document.body.appendChild(_eyedropperOverlay);

        debugLog('[probe] Activated — click anywhere on the page to inspect. Click Probe again to exit.', 'info');
        setStatus('Probe active — click anywhere to inspect element/pixel');
    } else {
        document.removeEventListener('click', _eyedropperCaptureClick, true);
        document.removeEventListener('mousedown', _eyedropperBlockEvent, true);
        document.removeEventListener('mouseup', _eyedropperBlockEvent, true);
        document.removeEventListener('keydown', _eyedropperKeyHandler);

        document.body.classList.remove('probe-active');

        if (_eyedropperOverlay && _eyedropperOverlay.parentNode) {
            _eyedropperOverlay.parentNode.removeChild(_eyedropperOverlay);
        }
        debugLog('[probe] Deactivated', 'info');
        setStatus('Probe deactivated');
    }
}

function _eyedropperKeyHandler(e) {
    if (!_eyedropperActive) return;
    if (e.key === 'Escape') {
        toggleDebugEyedropper();
    }
}

function _eyedropperBlockEvent(e) {
    if (!_eyedropperActive) return;
    const btn = document.getElementById('debugEyedropperBtn');
    if (btn && (e.target === btn || btn.contains(e.target))) return;
    e.stopPropagation();
    e.preventDefault();
}

function eyedropperSample(e) {
    if (!_eyedropperActive) return;

    const realTarget = e.target;
    if (!realTarget) {
        debugLog('[probe] No element at click position', 'warn');
        return;
    }

    // --- Determine if we clicked on the recolor image canvas ---
    const isCanvasPixel = realTarget.tagName === 'CANVAS' ||
        (realTarget.id === 'canvasInner' || realTarget.closest('#canvasInner'));

    if (isCanvasPixel && canvas && canvas.width > 0) {
        eyedropperSampleCanvas(e);
        return;
    }

    // --- General DOM element probe ---
    eyedropperSampleElement(realTarget, e);
}

function eyedropperSampleElement(el, e) {
    const computed = window.getComputedStyle(el);
    const bgColor = computed.backgroundColor;
    const color = computed.color;
    const tag = el.tagName.toLowerCase();
    const id = el.id ? `#${el.id}` : '';
    const cls = el.className && typeof el.className === 'string'
        ? '.' + el.className.trim().split(/\s+/).join('.')
        : '';
    const selector = `${tag}${id}${cls}`;

    // Inline style background (if any)
    const inlineBg = el.style.background || el.style.backgroundColor || '';

    // Walk up to find the nearest visible background
    let bgSource = '(transparent)';
    let bgEl = el;
    while (bgEl) {
        const cs = window.getComputedStyle(bgEl);
        const bg = cs.backgroundColor;
        if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
            const bgTag = bgEl.tagName.toLowerCase();
            const bgId = bgEl.id ? `#${bgEl.id}` : '';
            bgSource = `${bg} from ${bgTag}${bgId}`;
            break;
        }
        bgEl = bgEl.parentElement;
    }

    // Try to read hex from text content (for swatch labels)
    const textContent = el.textContent ? el.textContent.trim().substring(0, 30) : '';

    debugLog(`[probe] ─── Element Probe ───`, 'info');
    debugLog(`[probe]   element: <${selector}>`, 'info');
    debugLog(`[probe]   text: "${textContent}"`, 'info');
    debugLog(`[probe]   computed bg: ${bgColor}`, 'info');
    debugLog(`[probe]   computed color: ${color}`, 'info');
    if (inlineBg) {
        debugLog(`[probe]   inline style bg: ${inlineBg}`, 'info');
    }
    debugLog(`[probe]   effective bg: ${bgSource}`, 'info');
    debugLog(`[probe]   position: (${e.clientX}, ${e.clientY}) viewport`, 'info');
    debugLog(`[probe] ─── End Probe ───`, 'info');
}

function eyedropperSampleCanvas(e) {
    // Get image-space coordinates
    const coords = getCanvasCoords(e);
    const { x, y } = coords;

    // Bounds check
    if (x < 0 || y < 0 || !canvas || x >= canvas.width || y >= canvas.height) {
        debugLog(`[probe] Canvas out of bounds: (${x}, ${y})`, 'warn');
        return;
    }

    // 1. Read ORIGINAL pixel from hidden canvas
    const ctx2d = canvas.getContext('2d');
    const origPixel = ctx2d.getImageData(x, y, 1, 1).data;
    const origRGB = [origPixel[0], origPixel[1], origPixel[2]];
    const origHex = rgbToHex(...origRGB);
    const origLab = RGB2LAB(origRGB);

    // 2. Read RECOLORED pixel from visible canvas
    let recoloredRGB = null;
    let recoloredHex = '?';
    const vc = getVisibleCanvas();

    if (vc === webglCanvas && gl) {
        const glX = x;
        const glY = webglCanvas.height - 1 - y;
        const pixel = new Uint8Array(4);
        gl.readPixels(glX, glY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
        recoloredRGB = [pixel[0], pixel[1], pixel[2]];
        recoloredHex = rgbToHex(...recoloredRGB);
    } else if (vc === displayCanvas) {
        const dCtx = displayCanvas.getContext('2d');
        const dpx = dCtx.getImageData(x, y, 1, 1).data;
        recoloredRGB = [dpx[0], dpx[1], dpx[2]];
        recoloredHex = rgbToHex(...recoloredRGB);
    }

    // 3. Find nearest origin palette color
    let nearestOrigin = -1;
    let nearestDist = Infinity;
    const oldLabArr = [];
    for (let i = 0; i < originalPalette.length; i++) {
        const oLab = RGB2LAB(originalPalette[i]);
        oldLabArr.push(oLab);
        const dL = origLab[0] - oLab[0];
        const dA = origLab[1] - oLab[1];
        const dB = origLab[2] - oLab[2];
        const dist = Math.sqrt(dL*dL + dA*dA + dB*dB);
        if (dist < nearestDist) {
            nearestDist = dist;
            nearestOrigin = i;
        }
    }

    // 4. Trace the mapping chain
    let mappingInfo = '';
    if (nearestOrigin >= 0) {
        const col = originToColumn[nearestOrigin];
        const bypass = columnBypass[col] ? ' BYPASS' : '';
        const oOpacity = originOpacity[nearestOrigin] !== undefined ? originOpacity[nearestOrigin] : 100;
        const tOpacity = col !== undefined && col !== 'locked' && col !== 'bank' && targetOpacity[col] !== undefined ? targetOpacity[col] : 100;
        const effOpacity = (oOpacity / 100) * (tOpacity / 100);

        let targetHex = '?';
        if (typeof col === 'number' && col < targetPalette.length && targetPalette[col] !== null) {
            targetHex = rgbToHex(...targetPalette[col]);
        }

        mappingInfo = `origin#${nearestOrigin}→col${col}${bypass}, ` +
            `originColor=${rgbToHex(...originalPalette[nearestOrigin])}, ` +
            `targetColor=${targetHex}, ` +
            `opacity=(o:${oOpacity}%,t:${tOpacity}%,eff:${(effOpacity*100).toFixed(0)}%)`;

        // 5. Compute expected recolored value via Simple algorithm weights
        if (typeof col === 'number' && !columnBypass[col] && targetPalette[col] !== null) {
            const k = originalPalette.length;
            const blendSharpness = 2.0;
            let distances = [];
            let minDist = Infinity;
            for (let j = 0; j < k; j++) {
                const dL = origLab[0] - oldLabArr[j][0];
                const dA = origLab[1] - oldLabArr[j][1];
                const dB = origLab[2] - oldLabArr[j][2];
                const d = Math.sqrt(dL*dL + dA*dA + dB*dB);
                distances.push(d);
                if (d < minDist) minDist = d;
            }

            let totalWeight = 0;
            let weights = [];
            for (let j = 0; j < k; j++) {
                const relDist = distances[j] / Math.max(minDist, 1.0);
                const w = Math.exp(-blendSharpness * (relDist - 1.0));
                weights.push(w);
                totalWeight += w;
            }

            let dL = 0, dA = 0, dB = 0;
            if (totalWeight > 0) {
                for (let j = 0; j < k; j++) {
                    const nw = weights[j] / totalWeight;
                    const colJ = originToColumn[j];
                    if (typeof colJ === 'number' && colJ < targetPalette.length && targetPalette[colJ] !== null) {
                        const tLabJ = RGB2LAB(targetPalette[colJ]);
                        dL += nw * (tLabJ[0] - oldLabArr[j][0]);
                        dA += nw * (tLabJ[1] - oldLabArr[j][1]);
                        dB += nw * (tLabJ[2] - oldLabArr[j][2]);
                    }
                }
            }

            const expectedLab = [origLab[0] + dL, origLab[1] + dA, origLab[2] + dB];
            const expectedRGB = LAB2RGB(expectedLab);
            const expectedHex = rgbToHex(...expectedRGB);

            // Top 3 weights
            const indexed = weights.map((w, i) => ({ i, w: w / totalWeight }));
            indexed.sort((a, b) => b.w - a.w);
            const top3 = indexed.slice(0, 3).map(e => {
                const c = originToColumn[e.i];
                const th = typeof c === 'number' && c < targetPalette.length && targetPalette[c] !== null ? rgbToHex(...targetPalette[c]) : '?';
                return `o#${e.i}→col${c}(${th}) w=${(e.w*100).toFixed(1)}%`;
            }).join(', ');

            mappingInfo += `, expected=${expectedHex}, weights=[${top3}]`;
        }
    }

    // 6. Detect what rendered this pixel (WebGL vs CPU, render type)
    const renderSource = _lastWebGLRenderType
        ? `WebGL/${_lastWebGLRenderType}`
        : (vc === displayCanvas ? 'CPU/2D' : 'original');

    // Log everything
    debugLog(`[probe] ─── Pixel Probe (${x}, ${y}) ───`, 'info');
    debugLog(`[probe]   original: ${origHex} (L=${origLab[0].toFixed(1)} a=${origLab[1].toFixed(1)} b=${origLab[2].toFixed(1)})`, 'info');
    debugLog(`[probe]   rendered: ${recoloredHex} (source: ${renderSource}, ΔE=${nearestDist.toFixed(1)} to nearest origin)`, 'info');
    if (mappingInfo) {
        debugLog(`[probe]   mapping: ${mappingInfo}`, 'info');
    }
    debugLog(`[probe] ─── End Probe ───`, 'info');
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
    const oldAlgo = selectedAlgorithm;
    selectedAlgorithm = selectedAlgorithm === 'simple' ? 'rbf' : 'simple';
    debugLog(`[algorithm-switch] ${oldAlgo} → ${selectedAlgorithm}`);
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

function resetLuminositySlider() {
    const slider = document.getElementById('luminositySlider');
    const value = document.getElementById('luminosityValue');
    if (slider) slider.value = 0;
    if (value) value.textContent = '0';
}

// Apply luminosity as a post-processing step, respecting locked colors
function applyLuminosityPostProcess() {
    if (!canvas || !originalImageData) {
        setStatus('Load an image and apply recolor first');
        return;
    }

    const luminosity = parseInt(document.getElementById('luminositySlider').value);
    if (luminosity === 0) {
        setStatus('Luminosity is at 0 — no change applied');
        return;
    }
    debugLog(`[luminosity-apply] value=${luminosity}`);

    // Sync WebGL pixels to CPU before manipulating them
    syncWebGLToCPU();

    showLoading();
    setStatus('Applying luminosity post-processing...');

    setTimeout(() => {
        const width = canvas.width;
        const height = canvas.height;
        const factor = 1 + (luminosity / 100);

        // Build a set of LAB colors that are locked (bank + bypassed columns)
        const lockedLabs = [];
        for (let i = 0; i < originalPalette.length; i++) {
            const col = originToColumn[i];
            const isLocked = (col === 'locked' || col === 'bank' || typeof col !== 'number' || col >= targetPalette.length);
            const isBypassed = (typeof col === 'number' && columnBypass[col]);
            if (isLocked || isBypassed) {
                lockedLabs.push(RGB2LAB(originalPalette[i]));
            }
        }

        // Get current canvas pixel data (the recolored image)
        const currentData = ctx.getImageData(0, 0, width, height);
        const data = currentData.data;

        // For each pixel, check if it's closest to a locked color — if so, skip luminosity
        const origData = originalImageData.data;
        const step = 4; // process every pixel

        for (let idx = 0; idx < data.length; idx += step) {
            // Use original pixel color to determine if this pixel belongs to a locked origin
            const origR = origData[idx];
            const origG = origData[idx + 1];
            const origB = origData[idx + 2];

            let skipLuminosity = false;

            if (lockedLabs.length > 0) {
                const pixLab = RGB2LAB([origR, origG, origB]);

                // Find which origin color this pixel is closest to
                let minDist = Infinity;
                let isClosestLocked = false;

                // Check all origin colors
                for (let i = 0; i < originalPalette.length; i++) {
                    const oLab = RGB2LAB(originalPalette[i]);
                    const dist = Math.pow(pixLab[0] - oLab[0], 2) +
                                 Math.pow(pixLab[1] - oLab[1], 2) +
                                 Math.pow(pixLab[2] - oLab[2], 2);
                    if (dist < minDist) {
                        minDist = dist;
                        const col = originToColumn[i];
                        const locked = (col === 'locked' || col === 'bank' || typeof col !== 'number' || col >= targetPalette.length);
                        const bypassed = (typeof col === 'number' && columnBypass[col]);
                        isClosestLocked = locked || bypassed;
                    }
                }

                skipLuminosity = isClosestLocked;
            }

            if (!skipLuminosity) {
                data[idx] = Math.max(0, Math.min(255, data[idx] * factor));
                data[idx + 1] = Math.max(0, Math.min(255, data[idx + 1] * factor));
                data[idx + 2] = Math.max(0, Math.min(255, data[idx + 2] * factor));
            }
        }

        ctx.putImageData(currentData, 0, 0);
        imageData = ctx.getImageData(0, 0, width, height);
        // Luminosity is a CPU post-process — display must use 2D canvas now
        _lastWebGLRenderType = null;
        _webglDirty = false;
        updateDisplayCanvas();
        hideLoading();
        setStatus('Luminosity applied (locked colors preserved)');
    }, 50);
}

// Debug function - call from console: debugMapping()
window.debugMapping = function() {
    console.log('=== Debug Mapping ===');
    console.log('originCount:', originCount);
    console.log('targetCount:', targetCount);
    console.log('originToColumn:', JSON.stringify(originToColumn));
    console.log('originalPalette length:', originalPalette.length);
    console.log('targetPalette length:', targetPalette.length);
    console.log('targetPalette:', targetPalette.map(c => c ? rgbToHex(...c) : 'blank'));

    // Show which origins map to which targets
    for (let i = 0; i < originCount; i++) {
        const col = originToColumn[i];
        const originColor = rgbToHex(...originalPalette[i]);
        if (col === 'bank' || col === 'locked') {
            console.log(`  Origin ${i} (${originColor}) -> ${col} (no change)`);
        } else if (typeof col === 'number' && col < targetPalette.length) {
            const tc = targetPalette[col];
            console.log(`  Origin ${i} (${originColor}) -> Target ${col} (${tc ? rgbToHex(...tc) : 'blank'})`);
        } else {
            console.log(`  Origin ${i} (${originColor}) -> INVALID col=${col}`);
        }
    }
    return {originToColumn, originCount, targetCount, originalPalette, targetPalette};
};

// ============================================
// Configuration Save/Load System
// ============================================

let savedConfigs = []; // kept for backward compat with imports
let configCounter = 0;
let recolorHistory = []; // tracks every previewed recolor transformation
let currentViewedConfigId = null; // ID of the config currently being viewed

function buildConfigSnapshot() {
    return {
        id: Date.now() + Math.random(),
        name: `Recolor ${recolorHistory.length + 1}`,
        timestamp: new Date().toISOString(),
        savedForExport: false,
        originCount,
        targetCount,
        originalPalette: originalPalette.map(c => [...c]),
        targetPalette: targetPalette.map(c => c ? [...c] : [128, 128, 128]),
        colorPercentages: [...colorPercentages],
        originToColumn: [...originToColumn],
        columnBypass: {...columnBypass},
        algorithm: selectedAlgorithm,
        luminosity: parseInt(document.getElementById('luminositySlider').value),
        originOpacity: {...originOpacity},
        targetOpacity: {...targetOpacity},
        pickedColors: pickedColors.map(c => [...c]),
        pickedPositions: pickedPositions.map(p => ({ x: p.x, y: p.y, color: [...p.color] })),
        pickedCategories: [...pickedCategories]
    };
}

function addToRecolorHistory() {
    if (!originalImageData) return;
    if (targetPalette.some(t => t === null)) return;
    const snapshot = buildConfigSnapshot();
    snapshot.name = `Recolor ${recolorHistory.length + 1}`;
    recolorHistory.push(snapshot);
    currentViewedConfigId = snapshot.id;
    renderConfigList();
}

function saveCurrentConfig() {
    if (!originalImageData) {
        setStatus('Load an image first before saving config');
        return;
    }

    // If we're viewing a history entry, mark it as saved
    if (currentViewedConfigId) {
        const entry = recolorHistory.find(c => c.id === currentViewedConfigId);
        if (entry && !entry.savedForExport) {
            entry.savedForExport = true;
            renderConfigList();
            setStatus(`Saved current configuration for export`);
            return;
        }
    }

    // Otherwise create a new saved entry
    const snapshot = buildConfigSnapshot();
    snapshot.savedForExport = true;
    snapshot.name = `Recolor ${recolorHistory.length + 1}`;
    recolorHistory.push(snapshot);
    currentViewedConfigId = snapshot.id;
    renderConfigList();
    setStatus(`Saved current configuration for export`);
}

function toggleSaveForExport(configId) {
    const entry = recolorHistory.find(c => c.id === configId);
    if (entry) {
        entry.savedForExport = !entry.savedForExport;
        renderConfigList();
    }
}

function validateAndRepairConfig(config) {
    const warnings = [];

    // Ensure required arrays exist
    if (!config.originalPalette || !Array.isArray(config.originalPalette)) {
        warnings.push('Missing originalPalette');
        config.originalPalette = [];
    }
    if (!config.targetPalette || !Array.isArray(config.targetPalette)) {
        warnings.push('Missing targetPalette');
        config.targetPalette = [];
    }
    if (!config.originToColumn || !Array.isArray(config.originToColumn)) {
        warnings.push('Missing originToColumn');
        config.originToColumn = [];
    }
    if (!config.colorPercentages || !Array.isArray(config.colorPercentages)) {
        warnings.push('Missing colorPercentages');
        config.colorPercentages = [];
    }

    // Validate originalPalette entries are valid RGB arrays
    for (let i = 0; i < config.originalPalette.length; i++) {
        const c = config.originalPalette[i];
        if (!Array.isArray(c) || c.length < 3 || c.some(v => typeof v !== 'number' || isNaN(v))) {
            warnings.push('Invalid originalPalette[' + i + ']: ' + JSON.stringify(c));
            config.originalPalette[i] = [128, 128, 128];
        }
    }

    // Validate targetPalette entries
    for (let i = 0; i < config.targetPalette.length; i++) {
        const c = config.targetPalette[i];
        if (c === null || c === undefined) continue; // null means unmapped, that's OK
        if (!Array.isArray(c) || c.length < 3 || c.some(v => typeof v !== 'number' || isNaN(v))) {
            warnings.push('Invalid targetPalette[' + i + ']: ' + JSON.stringify(c));
            config.targetPalette[i] = [128, 128, 128];
        }
    }

    // Ensure counts match array lengths
    if (config.originCount !== config.originalPalette.length) {
        warnings.push('originCount (' + config.originCount + ') !== originalPalette.length (' + config.originalPalette.length + ')');
        config.originCount = config.originalPalette.length;
    }
    if (config.targetCount !== config.targetPalette.length) {
        warnings.push('targetCount (' + config.targetCount + ') !== targetPalette.length (' + config.targetPalette.length + ')');
        config.targetCount = config.targetPalette.length;
    }

    // Ensure originToColumn length matches originalPalette
    while (config.originToColumn.length < config.originalPalette.length) {
        warnings.push('originToColumn too short, padding with column 0');
        config.originToColumn.push(0);
    }
    if (config.originToColumn.length > config.originalPalette.length) {
        warnings.push('originToColumn too long, truncating');
        config.originToColumn.length = config.originalPalette.length;
    }

    // Validate originToColumn values — clamp to valid target range
    for (let i = 0; i < config.originToColumn.length; i++) {
        const col = config.originToColumn[i];
        if (col === 'locked' || col === 'bank') continue;
        if (typeof col === 'number' && col >= config.targetPalette.length) {
            warnings.push('originToColumn[' + i + ']=' + col + ' exceeds targetPalette.length (' + config.targetPalette.length + '), clamping');
            config.originToColumn[i] = Math.min(col, config.targetPalette.length - 1);
        }
    }

    // Ensure colorPercentages length matches
    while (config.colorPercentages.length < config.originalPalette.length) {
        config.colorPercentages.push(0);
    }

    // Backfill missing optional fields
    if (!config.columnBypass) config.columnBypass = {};
    if (!config.originOpacity) config.originOpacity = {};
    if (!config.targetOpacity) config.targetOpacity = {};
    if (!config.algorithm) config.algorithm = 'simple';

    return warnings;
}

function loadConfig(configId) {
    const config = recolorHistory.find(c => c.id === configId) || savedConfigs.find(c => c.id === configId);
    if (!config) {
        setStatus('Configuration not found');
        return;
    }
    currentViewedConfigId = configId;

    // Validate and repair config data from potentially older versions
    const warnings = validateAndRepairConfig(config);
    if (warnings.length > 0) {
        console.warn('Config repair warnings:', warnings);
        debugLog(`[config-repair] ${warnings.join('; ')}`, 'warn');
    }

    const bypassCount = config.columnBypass ? Object.keys(config.columnBypass).filter(k => config.columnBypass[k]).length : 0;
    const origOpacityCount = config.originOpacity ? Object.keys(config.originOpacity).filter(k => config.originOpacity[k] !== 100).length : 0;
    const tgtOpacityCount = config.targetOpacity ? Object.keys(config.targetOpacity).filter(k => config.targetOpacity[k] !== 100).length : 0;
    debugLog(`[config-load] "${config.name || 'unnamed'}", origins=${config.originCount}, targets=${config.targetCount}, algo=${config.algorithm || 'simple'}, bypass=${bypassCount}, opacity=(orig=${origOpacityCount},tgt=${tgtOpacityCount}), lum=${config.luminosity || 0}, picked=${config.pickedColors ? config.pickedColors.length : 0}`);

    originCount = config.originCount;
    targetCount = config.targetCount;
    originalPalette = config.originalPalette.map(c => [...c]);
    targetPalette = config.targetPalette.map(c => c ? [...c] : [128, 128, 128]);
    colorPercentages = [...config.colorPercentages];
    originToColumn = [...config.originToColumn];
    columnBypass = config.columnBypass ? {...config.columnBypass} : {};
    originOpacity = config.originOpacity ? {...config.originOpacity} : {};
    targetOpacity = config.targetOpacity ? {...config.targetOpacity} : {};
    _opacityCache = null;   // invalidate so recolor picks up restored values
    selectedAlgorithm = config.algorithm || 'simple';
    document.getElementById('luminositySlider').value = config.luminosity || 0;
    document.getElementById('luminosityValue').textContent = config.luminosity || 0;

    // Restore picked color selections if saved (backfill the picker)
    if (config.pickedColors && config.pickedColors.length > 0) {
        pickedColors = config.pickedColors.map(c => [...c]);
        pickedPositions = config.pickedPositions
            ? config.pickedPositions.map(p => ({ x: p.x, y: p.y, color: [...p.color] }))
            : config.pickedColors.map(c => ({ x: 0, y: 0, color: [...c] }));
        pickedCategories = config.pickedCategories ? [...config.pickedCategories] : config.pickedColors.map(() => 0);

        // Remove any existing markers — data is preserved in arrays
        // Markers will reappear when the picker is re-engaged
        document.querySelectorAll('.picker-marker').forEach(m => m.remove());
        shouldKeepPickedMarkers = false;
    }

    document.getElementById('originCountDisplay').value = originCount;
    document.getElementById('targetCountDisplay').value = targetCount;
    document.getElementById('columnMappingContainer').setAttribute('data-target-count', targetCount);

    updateAlgorithmUI();

    // Reveal full UI BEFORE rendering swatches — ensures uiStage='complete' so
    // target swatches get correct selectability/opacity classes (not `.not-selectable`)
    revealFullUI();

    renderColumnMapping();
    autoRecolorImage();
    renderConfigList();

    // Enable live preview when loading a config (user expects to see the recolor)
    const liveToggle = document.getElementById('livePreviewToggle');
    if (liveToggle && !liveToggle.checked) {
        liveToggle.checked = true;
        toggleLivePreview(true);
    }
    // Expand origin section when loading a config so user can see the mapping
    const originCollapsible = document.getElementById('originCollapsible');
    if (originCollapsible) originCollapsible.setAttribute('open', '');
    // Re-center overlays in case wrapper size changed
    updateStickyOverlays();
    setStatus(`Loaded configuration: ${config.name}`);
}

function deleteConfig(configId) {
    recolorHistory = recolorHistory.filter(c => c.id !== configId);
    savedConfigs = savedConfigs.filter(c => c.id !== configId);
    renderConfigList();
    setStatus('Configuration deleted');
}

function renderConfigList() {
    const list = document.getElementById('configList');
    if (!list) return;

    list.innerHTML = '';

    [...recolorHistory].reverse().forEach((config, index) => {
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

        // Save for export button / saved indicator
        const saveBtn = document.createElement('button');
        saveBtn.className = 'config-item-save' + (config.savedForExport ? ' saved' : '');
        if (config.savedForExport) {
            saveBtn.innerHTML = '<span class="save-check">✓</span><span class="save-text">Saved!</span>';
        } else {
            saveBtn.innerHTML = 'Save';
            saveBtn.title = 'Save for export';
        }
        saveBtn.onclick = (e) => {
            e.stopPropagation();
            if (!config.savedForExport) {
                config.savedForExport = true;
                renderConfigList();
            }
        };

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
        item.appendChild(saveBtn);
        item.appendChild(deleteBtn);
        list.appendChild(item);
    });

    // Update export button appearance
    if (typeof updateExportButtonState === 'function') updateExportButtonState();
    updateSaveButtonState();
}

function updateSaveButtonState() {
    const btn = document.querySelector('.btn-save-config');
    if (!btn) return;
    const currentEntry = currentViewedConfigId
        ? recolorHistory.find(c => c.id === currentViewedConfigId)
        : null;
    const alreadySaved = currentEntry && currentEntry.savedForExport;
    btn.disabled = !!alreadySaved;
    btn.classList.toggle('btn-disabled', !!alreadySaved);
}

function exportAllConfigs() {
    const savedEntries = recolorHistory.filter(c => c.savedForExport);
    if (savedEntries.length === 0) {
        setStatus('No saved configurations to export. Click "Save" next to history items first.');
        return;
    }

    const exportData = {
        version: '18',
        exportDate: new Date().toISOString(),
        configs: savedEntries
    };

    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `palette-configs-${new Date().toISOString().split('T')[0]}.json`;
    link.click();

    URL.revokeObjectURL(url);
    setStatus(`Exported ${savedEntries.length} configurations`);
}

function importConfigFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);

            if (data.configs && Array.isArray(data.configs)) {
                const existingIds = new Set(recolorHistory.map(c => c.id));
                let importedCount = 0;

                data.configs.forEach(config => {
                    if (existingIds.has(config.id)) {
                        config.id = Date.now() + Math.random();
                    }
                    config.savedForExport = true; // imported configs are already saved
                    const repairWarnings = validateAndRepairConfig(config);
                    if (repairWarnings.length > 0) {
                        console.warn('Repaired imported config "' + (config.name || 'unnamed') + '":', repairWarnings);
                    }
                    recolorHistory.push(config);
                    importedCount++;
                });

                debugLog(`[config-import] ${importedCount} configs imported from file`);
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

function importConfigFileEarly(event) {
    // Early import from Color Analysis panel — imports configs, auto-loads first one, reveals full UI
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);

            if (data.configs && Array.isArray(data.configs)) {
                const existingIds = new Set(recolorHistory.map(c => c.id));
                let importedCount = 0;
                let firstNewIndex = recolorHistory.length;

                let repairedCount = 0;
                data.configs.forEach(config => {
                    if (existingIds.has(config.id)) {
                        config.id = Date.now() + Math.random();
                    }
                    config.savedForExport = true;
                    const repairWarnings = validateAndRepairConfig(config);
                    if (repairWarnings.length > 0) {
                        console.warn('Repaired imported config "' + (config.name || 'unnamed') + '":', repairWarnings);
                        repairedCount++;
                    }
                    recolorHistory.push(config);
                    importedCount++;
                });

                renderConfigList();

                // Auto-load the first imported config
                if (importedCount > 0) {
                    loadConfig(recolorHistory[firstNewIndex].id);
                }

                debugLog(`[config-import-early] ${importedCount} configs imported, ${repairedCount} repaired, auto-loading first`);
                const repairNote = repairedCount > 0 ? ` (${repairedCount} repaired from older format)` : '';
                setStatus(`Imported ${importedCount} configurations${repairNote}. First config auto-loaded.`);
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

// ============================================
// Tutorial Overlay System
// ============================================

let tutorialStep = 0; // 0 = Background, 1 = Locked, 2+ = Accent N
let tutorialCollapsed = false;
let tutorialHasBeenShown = false; // Track first-time activation
let tutorialCtrlHidden = false; // Separate from collapsed — Ctrl is temporary hide

const tutorialSteps = [
    {
        category: 0, // CATEGORY_BACKGROUND
        title: 'Select Background:',
        subtitle: 'Click your mouse on the Background of your image.',
        italic: null,
        highlightClass: 'highlight-bg',
        highlightWord: 'Background'
    },
    {
        category: -1, // CATEGORY_LOCKED
        title: 'Select Locked Colors:',
        subtitle: "Click your mouse on all the colors you DON'T want recolored.",
        italic: "The background should be separate from this, you can individually lock it later if you want.",
        highlightClass: 'highlight-locked',
        highlightWord: 'Locked'
    }
    // Accent steps are generated dynamically
];

function getOrdinal(n) {
    if (n === 1) return 'first';
    if (n === 2) return 'second';
    if (n === 3) return 'third';
    if (n === 4) return 'fourth';
    if (n === 5) return 'fifth';
    if (n === 6) return 'sixth';
    return n + 'th';
}

function getAccentStep(accentNum) {
    return {
        category: accentNum,
        title: `Select Accent Color ${accentNum}:`,
        subtitle: `Select all the colors you want consolidated as your ${getOrdinal(accentNum)} accent color.`,
        italic: 'If you select a blue spot and a red spot, and set it to recode to green, all blue and red spots will be green.',
        greenReminder: 'Click the green button when you have categorized all your colors.',
        highlightClass: 'highlight-accent',
        highlightWord: `Accent Color ${accentNum}`
    };
}

function getTutorialStepData(step) {
    if (step < tutorialSteps.length) return tutorialSteps[step];
    const accentNum = step - 1; // step 2 => Accent 1, step 3 => Accent 2, etc.
    return getAccentStep(accentNum);
}

function renderTutorialText() {
    const textEl = document.getElementById('tutorialText');
    if (!textEl) return;

    const stepData = getTutorialStepData(tutorialStep);
    const titleWord = stepData.highlightWord;
    const highlightClass = stepData.highlightClass;

    // Build title with highlighted word
    const titleHTML = stepData.title.replace(titleWord, `<span class="${highlightClass}">${titleWord}</span>`);

    let html = `<div class="tutorial-title">${titleHTML}</div>`;
    html += `<div class="tutorial-subtitle">${stepData.subtitle}</div>`;
    if (stepData.greenReminder) {
        html += `<div class="tutorial-green-reminder">${stepData.greenReminder}</div>`;
    }
    if (stepData.italic) {
        html += `<div class="tutorial-italic">${stepData.italic}</div>`;
    }
    textEl.innerHTML = html;

    // Show/hide finish button (visible from Accent 1 onward, i.e. step >= 2)
    const finishBtn = document.getElementById('tutorialFinishBtn');
    if (finishBtn) {
        finishBtn.classList.toggle('hidden', tutorialStep < 2);
    }

    // Next Color button: past Accent 4 (step 5), turn red and change text to prompt moving on
    const nextBtn = document.getElementById('tutorialNextBtn');
    if (nextBtn) {
        const accentNum = tutorialStep - 1; // step 2 = Accent 1, step 5 = Accent 4
        if (accentNum >= 4) {
            nextBtn.textContent = 'Make me another category';
            nextBtn.classList.add('tutorial-next-btn-warn');
        } else {
            nextBtn.textContent = 'Next Color';
            nextBtn.classList.remove('tutorial-next-btn-warn');
        }
    }
}

function renderTutorialRight() {
    const rightEl = document.getElementById('tutorialRight');
    if (!rightEl) return;

    rightEl.innerHTML = '';

    // Clone the "Picking for" category selector
    const catSelectorOriginal = document.getElementById('pickerCategorySelector');
    if (catSelectorOriginal) {
        const catClone = catSelectorOriginal.cloneNode(true);
        catClone.id = 'tutorialCategorySelector';
        catClone.classList.remove('hidden');
        // Make the cloned select work and act as navigation
        const clonedSelect = catClone.querySelector('select');
        if (clonedSelect) {
            clonedSelect.id = 'tutorialCategorySelect';
            clonedSelect.value = pickerTargetCategory;
            clonedSelect.onchange = function() {
                const newCategory = parseInt(this.value);
                updatePickerCategory(newCategory);
                // Sync the tutorial step to match the selected category
                const stepIdx = categoryToTutorialStep(newCategory);
                if (stepIdx !== null) {
                    tutorialStep = stepIdx;
                    renderTutorialText();
                    renderTutorialRight();
                }
            };
        }
        rightEl.appendChild(catClone);
    }

    // Clone the picked swatches list and re-attach click handlers
    const swatchListOriginal = document.getElementById('pickerSwatchesList');
    if (swatchListOriginal) {
        const swatchClone = swatchListOriginal.cloneNode(true);
        swatchClone.id = 'tutorialSwatchesList';
        swatchClone.classList.remove('hidden');
        // Re-attach category cycling and delete handlers (cloneNode doesn't copy JS handlers)
        // Use data-color-index to get the real data index (list is rendered newest-first)
        const clonedItems = swatchClone.querySelectorAll('.picker-swatch-item');
        clonedItems.forEach((item) => {
            const realIdx = parseInt(item.dataset.colorIndex);
            const catLabel = item.querySelector('.picker-swatch-category');
            if (catLabel) {
                catLabel.onclick = (e) => {
                    e.stopPropagation();
                    cyclePickedColorCategory(realIdx);
                    renderTutorialRight();
                };
            }
            const delBtn = item.querySelector('.picker-swatch-delete');
            if (delBtn) {
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    removePickedColor(realIdx);
                    renderTutorialRight();
                };
            }
        });
        rightEl.appendChild(swatchClone);
    }

    // Clone the clear button
    const clearBtnOriginal = document.getElementById('pickerClearBtn');
    if (clearBtnOriginal) {
        const clearClone = clearBtnOriginal.cloneNode(true);
        clearClone.id = 'tutorialClearBtn';
        clearClone.classList.remove('hidden');
        clearClone.onclick = function() {
            clearPickerSelections();
            tutorialStep = 0;
            syncTutorialCategory();
            renderTutorialText();
            renderTutorialRight();
        };
        rightEl.appendChild(clearClone);
    }

    // Orange-label hint below clear button
    const orangeHint = document.createElement('span');
    orangeHint.className = 'tutorial-hint-item tutorial-orange-hint';
    orangeHint.innerHTML = 'Change a color\'s category by clicking its <span style="color:#e08530;font-weight:600;">orange label</span>';
    rightEl.appendChild(orangeHint);
}

function syncTutorialCategory() {
    // Set the picker category to match the current tutorial step
    const stepData = getTutorialStepData(tutorialStep);
    pickerTargetCategory = stepData.category;

    // Update the real category selector
    const select = document.getElementById('pickerCategorySelect');
    if (select) select.value = stepData.category;

    // Also update the tutorial's cloned selector
    const tutSelect = document.getElementById('tutorialCategorySelect');
    if (tutSelect) tutSelect.value = stepData.category;
}

// Map a category value back to the corresponding tutorial step index
function categoryToTutorialStep(category) {
    if (category === CATEGORY_BACKGROUND) return 0;
    if (category === CATEGORY_LOCKED) return 1;
    if (category >= 1) return category + 1; // Accent N => step N+1
    return null;
}

function tutorialNextStep() {
    tutorialStep++;
    syncTutorialCategory();
    renderTutorialText();
    renderTutorialRight();
    updatePickerCategoryOptions();
}

// Single entry point for advancing the picker category (double-shift, etc.)
// Handles both tutorial mode and traditional picker mode without redundancy.
function advancePickerCategory() {
    const tutorialOverlay = document.getElementById('tutorialOverlay');
    const tutorialIsVisible = tutorialOverlay && !tutorialOverlay.classList.contains('hidden');

    if (tutorialIsVisible) {
        // Tutorial is open — advance the tutorial step (which syncs category + dropdown)
        tutorialNextStep();
    } else {
        // Traditional picker or collapsed tutorial — just cycle the dropdown category
        const nextCat = getNextCategory(pickerTargetCategory);
        pickerTargetCategory = nextCat;
        const select = document.getElementById('pickerCategorySelect');
        if (select) select.value = nextCat;
        updatePickerCategoryOptions();
    }

    // Visual feedback: briefly flash the category selector
    const catSelector = document.getElementById('pickerCategorySelector');
    const tutCatSelector = document.getElementById('tutorialCategorySelector');
    const flashEl = tutorialIsVisible ? tutCatSelector : catSelector;
    if (flashEl) {
        flashEl.classList.add('category-flash');
        setTimeout(() => flashEl.classList.remove('category-flash'), 300);
    }
}

function showTutorialOverlay() {
    const overlay = document.getElementById('tutorialOverlay');
    const tab = document.getElementById('tutorialTab');
    const pickerOverlay = document.getElementById('pickerOverlay');

    if (overlay) overlay.classList.remove('hidden');
    if (tab) tab.classList.add('hidden');

    // Hide the traditional picker overlay grouping when tutorial is visible
    if (pickerOverlay) {
        pickerOverlay.dataset.tutorialHidden = 'true';
        pickerOverlay.style.display = 'none';
    }

    tutorialCollapsed = false;
    tutorialCtrlHidden = false;
    syncTutorialCategory();
    renderTutorialText();
    renderTutorialRight();
}

function hideTutorialOverlay() {
    const overlay = document.getElementById('tutorialOverlay');
    const tab = document.getElementById('tutorialTab');

    if (overlay) overlay.classList.add('hidden');
    if (tab) tab.classList.add('hidden');

    // Restore the traditional picker overlay
    const pickerOverlay = document.getElementById('pickerOverlay');
    if (pickerOverlay) {
        delete pickerOverlay.dataset.tutorialHidden;
        pickerOverlay.style.display = '';
    }
}

function collapseTutorial() {
    // Collapse = "I don't need the tutorial" — show traditional picker
    const overlay = document.getElementById('tutorialOverlay');
    const tab = document.getElementById('tutorialTab');
    const pickerOverlay = document.getElementById('pickerOverlay');

    if (overlay) overlay.classList.add('hidden');
    if (tab) tab.classList.remove('hidden');

    // Show traditional picker overlay since tutorial is collapsed
    if (pickerOverlay) {
        delete pickerOverlay.dataset.tutorialHidden;
        pickerOverlay.style.display = '';
    }

    tutorialCollapsed = true;
}

function expandTutorial() {
    tutorialCollapsed = false;
    showTutorialOverlay();
}

// Hook into updatePickerOverlay to also update the tutorial right panel
const _originalUpdatePickerOverlay = updatePickerOverlay;
updatePickerOverlay = function() {
    _originalUpdatePickerOverlay();
    // If tutorial is visible, refresh the right panel
    const overlay = document.getElementById('tutorialOverlay');
    if (overlay && !overlay.classList.contains('hidden')) {
        renderTutorialRight();
    }
};

// Update the export button appearance based on saved configs
function updateExportButtonState() {
    const exportBtn = document.getElementById('exportSavedBtn');
    if (!exportBtn) return;
    const hasSaved = recolorHistory.some(c => c.savedForExport);
    if (hasSaved) {
        exportBtn.classList.add('btn-export-active');
    } else {
        exportBtn.classList.remove('btn-export-active');
    }
}
