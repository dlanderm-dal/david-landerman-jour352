/**
 * Learning Hub — Image Upload Server
 *
 * A minimal local server that:
 * 1. Serves the Learning Hub files (replaces python -m http.server)
 * 2. Accepts image uploads from the comment system
 * 3. Saves images to the images/ folder
 *
 * Security:
 * - Only accepts image MIME types (image/png, image/jpeg, image/gif, image/webp)
 * - Max file size: 10MB
 * - Only writes to the images/ subfolder (path traversal blocked)
 * - Only accessible from localhost
 *
 * Usage: node image-server.js
 * Then open http://localhost:3000
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = 3000;
const ROOT = __dirname;
const IMAGES_DIR = path.join(ROOT, 'images');
const MAX_SIZE = 10 * 1024 * 1024; // 10MB

// Ensure images directory exists
if (!fs.existsSync(IMAGES_DIR)) fs.mkdirSync(IMAGES_DIR, { recursive: true });

// MIME types for static file serving
const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain',
  '.md': 'text/markdown',
  '.ico': 'image/x-icon',
};

const ALLOWED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];

const server = http.createServer(function (req, res) {
  // CORS headers for localhost
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // ========== IMAGE UPLOAD ==========
  if (req.method === 'POST' && req.url === '/upload-image') {
    var contentType = (req.headers['content-type'] || '').split(';')[0].trim();

    // Accept multipart/form-data or raw image
    if (contentType === 'application/json') {
      // JSON body with base64 image
      var body = '';
      req.on('data', function (chunk) {
        body += chunk;
        if (body.length > MAX_SIZE) {
          res.writeHead(413, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'File too large (max 10MB)' }));
          req.destroy();
        }
      });

      req.on('end', function () {
        try {
          var data = JSON.parse(body);
          var base64 = data.image; // data:image/png;base64,xxxxx
          var filename = data.filename || ('img-' + Date.now() + '.png');

          // Validate it's actually an image
          if (!base64 || !base64.startsWith('data:image/')) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Not a valid image' }));
            return;
          }

          // Extract MIME type
          var mimeMatch = base64.match(/^data:(image\/\w+);base64,/);
          if (!mimeMatch || ALLOWED_IMAGE_TYPES.indexOf(mimeMatch[1]) === -1) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Unsupported image type. Allowed: PNG, JPEG, GIF, WebP' }));
            return;
          }

          // Sanitize filename — only allow alphanumeric, dash, underscore, dot
          filename = filename.replace(/[^a-zA-Z0-9._-]/g, '_');
          // Prevent path traversal
          filename = path.basename(filename);

          // Ensure correct extension
          var ext = mimeMatch[1] === 'image/jpeg' ? '.jpg' : '.' + mimeMatch[1].split('/')[1];
          if (!filename.endsWith(ext)) filename = filename.replace(/\.[^.]+$/, '') + ext;

          // Decode and save
          var imageData = base64.replace(/^data:image\/\w+;base64,/, '');
          var buffer = Buffer.from(imageData, 'base64');

          var filePath = path.join(IMAGES_DIR, filename);
          fs.writeFileSync(filePath, buffer);

          console.log('[upload] Saved: ' + filename + ' (' + (buffer.length / 1024).toFixed(1) + 'KB)');

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            success: true,
            filename: filename,
            path: 'images/' + filename,
            size: buffer.length
          }));
        } catch (e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Upload failed: ' + e.message }));
        }
      });
      return;
    }

    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Send JSON with { image: "data:image/...", filename: "name.png" }' }));
    return;
  }

  // ========== SCRATCH PAD: SAVE NOTES INTO HTML FILE ==========
  if (req.method === 'POST' && req.url === '/save-scratchpad') {
    var body = '';
    req.on('data', function (chunk) {
      body += chunk;
      if (body.length > MAX_SIZE) {
        res.writeHead(413, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Too large' }));
        req.destroy();
      }
    });

    req.on('end', function () {
      try {
        var data = JSON.parse(body);
        var targetFile = data.file; // e.g. "AI_study_guides/github-complete-guide.html"
        var notes = data.notes;    // array of note objects

        if (!targetFile || !Array.isArray(notes)) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Send { file: "path.html", notes: [...] }' }));
          return;
        }

        // Security: only allow files within ROOT
        var filePath = path.join(ROOT, targetFile);
        if (!filePath.startsWith(ROOT)) {
          res.writeHead(403, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Forbidden path' }));
          return;
        }

        if (!fs.existsSync(filePath)) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'File not found' }));
          return;
        }

        var html = fs.readFileSync(filePath, 'utf8');

        // Build the scratch pad HTML block
        var padHtml = '\n<!-- lh-scratchpad-start -->\n';
        padHtml += '<div id="lh-scratchpad-data" class="tab-content" style="display:none;">\n';
        notes.forEach(function (note) {
          padHtml += '  <div class="lh-scratchpad-note" data-id="' + (note.id || '') + '" data-rev="' + (note.rev || 1) + '" data-timestamp="' + (note.timestamp || '') + '">';
          // Escape HTML in note text
          var escaped = (note.text || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
          padHtml += escaped;
          padHtml += '</div>\n';
        });
        padHtml += '</div>\n<!-- lh-scratchpad-end -->\n';

        // Replace existing scratchpad block, or insert before </body>
        if (html.indexOf('<!-- lh-scratchpad-start -->') !== -1) {
          html = html.replace(/\n<!-- lh-scratchpad-start -->[\s\S]*?<!-- lh-scratchpad-end -->\n/, padHtml);
        } else {
          html = html.replace('</body>', padHtml + '</body>');
        }

        fs.writeFileSync(filePath, html, 'utf8');
        console.log('[scratchpad] Saved ' + notes.length + ' notes to ' + targetFile);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, count: notes.length }));
      } catch (e) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Save failed: ' + e.message }));
      }
    });
    return;
  }

  // ========== STATIC FILE SERVING ==========
  var parsedUrl = url.parse(req.url);
  var pathname = decodeURIComponent(parsedUrl.pathname);

  // Default to index.html
  if (pathname === '/') pathname = '/index.html';

  var filePath = path.join(ROOT, pathname);

  // Security: prevent path traversal
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  // Check if file exists
  fs.stat(filePath, function (err, stats) {
    if (err || !stats.isFile()) {
      res.writeHead(404);
      res.end('Not found: ' + pathname);
      return;
    }

    var ext = path.extname(filePath).toLowerCase();
    var contentType = MIME_TYPES[ext] || 'application/octet-stream';

    fs.readFile(filePath, function (err, data) {
      if (err) {
        res.writeHead(500);
        res.end('Server error');
        return;
      }
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(data);
    });
  });
});

server.listen(PORT, '127.0.0.1', function () {
  console.log('');
  console.log('  Learning Hub Server');
  console.log('  http://localhost:' + PORT);
  console.log('');
  console.log('  Static files:  ' + ROOT);
  console.log('  Image uploads: ' + IMAGES_DIR);
  console.log('  Max upload:    10MB');
  console.log('');
  console.log('  Press Ctrl+C to stop');
  console.log('');
});
