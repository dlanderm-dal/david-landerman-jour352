# Comment System Component

A standalone, reusable annotation/comment system for HTML websites. Designed for human↔AI feedback loops where users annotate pages and AI agents implement changes.

## Quick Start

Add two files to your project:

```html
<!-- Before </body> on every page that needs comments -->
<link rel="stylesheet" href="comment-system.css">
<script src="comment-system.js"></script>
```

That's it. The system auto-injects a navbar, sidebar, and comment tools.

## What It Does

Users can:
- **Highlight any text** on the page → a "Quick Fix" / "Comment" bar appears
- **Quick Fix** → opens a sidebar to write a one-off prompt, copies to clipboard for pasting into Claude Code
- **Comment** → opens a popup to write a note, optionally attach a screenshot
- **Thread replies** on existing comments
- **Resolve** implemented comments (hides the marker, keeps the data)
- **Export all comments** as a structured prompt for an AI coding agent
- **Import/export data** between browser origins (e.g., `file://` to `localhost`)

The AI agent:
- Receives the exported prompt (pasted by the user)
- Edits the target file to address each comment
- Adds `<!-- lh-implemented: id1, id2 -->` to the HTML
- On next page refresh, the comment system reads this marker and marks those comments as "implemented" (green checkmark)

## Features

### Comment Types
| Type | User Action | Agent Action |
|------|-------------|--------------|
| Text Edit | "Change this to..." | Replace the highlighted text |
| Question | "What does this mean?" | Add explanation near the text |
| Bug Report | "This overflows its container" | Use DOM path to find and fix CSS/HTML |
| Reorganize | "Move this to section 3" | Relocate content within the file |

### Comment Lifecycle
1. **Active** (yellow dot) — new comment, included in exports
2. **Implemented** (green checkmark) — agent addressed it, can be replied to
3. **Resolved** (hidden) — user confirmed the fix, viewable in archive

### Image Support
- Drag/drop or click-to-browse in comment popup
- Images resized to max 800px width
- If `imageUploadEndpoint` is configured: saved to server as a file
- If no server: stored as base64 in localStorage (limited by ~5MB quota)

### Tab Awareness
If the page has tab navigation (detected via `tabSelector` config), comment markers only show for the active tab. Comments from renamed/removed tabs show on all tabs (orphan recovery).

### Keyboard Shortcuts
- `Ctrl+Shift+A` — Toggle the Quick Fix sidebar
- `Escape` — Close popups and floating bars

## Configuration

```html
<script>
CommentSystem.init({
  // Required for proper prompt generation:
  projectRoot: 'my-project/',
  architectureFile: 'my-project/ARCHITECTURE.md',

  // Optional:
  storageKey: 'myapp_comments',
  instructionsKey: 'myapp_instructions',
  imageUploadEndpoint: '/api/upload-image',
  tabSelector: '.my-tab-btn',
  dynamicContainers: ['.todo-item', '.card'],
  navbarHeight: 64,
});
</script>
```

## Image Server (Optional)

For unlimited image storage, run a local server that accepts uploads:

```javascript
// Endpoint: POST /upload-image
// Body: { "image": "data:image/png;base64,...", "filename": "comment-123.png" }
// Response: { "success": true, "path": "images/comment-123.png" }
```

The comment system automatically falls back to base64 localStorage if the server isn't available.

## Theming

Override CSS custom properties:

```css
:root {
  --cs-accent: #7c3aed;      /* Purple instead of blue */
  --cs-green: #059669;
  --cs-bg: #fafafa;
  --cs-sidebar-width: 320px;  /* Narrower sidebar */
}
```

## How the Exported Prompt Works

When the user clicks "Export Comments," the system generates a markdown-formatted prompt containing:

1. A reference to the architecture file (for agent context)
2. The target file path
3. Guidelines for handling different comment types
4. Each comment with:
   - The highlighted text (for the agent to find in the file)
   - The user's note
   - Thread replies (with `[NEW]` / `[previously seen]` labels)
   - Location context (active tab, nearest heading, DOM path)
   - Attached image (if any)
5. Instructions to add `<!-- lh-implemented: ids -->` after making changes
6. Instructions to wrap new content in `<span class="lh-new-content">...</span>`

The user copies this prompt and pastes it into their AI coding agent.

## For AI Agents Processing Comments

When you receive an exported comment batch:

1. **Read the architecture file** referenced at the top of the prompt
2. **Find each comment's text** in the target file using the "Find this text" quote
3. **Use the Location context** (tab name, heading, DOM path) to pinpoint the location
4. **Apply the appropriate action** based on the comment type
5. **Wrap new/modified content** in `<span class="lh-new-content">...</span>`
6. **Remove old `lh-new-content` spans** from previous edits (unwrap, keep content)
7. **Add implementation marker** before `</body>`: `<!-- lh-implemented: id1, id2, id3 -->`
8. **Never write content that sounds like it's answering an invisible question** — the comment is not visible on the page. New content must read as if it was always part of the document.
9. **Integrate, don't isolate** — weave new information into existing text flow rather than always creating separate callout boxes.

## File Structure

```
component-comment_infrastructure/
  comment-system.js     ← Drop this into your project
  comment-system.css    ← Drop this into your project
  template.html         ← Demo/test page
  ARCHITECTURE.md       ← For AI agents (read-first context)
  README.md             ← This file (for humans)
```

## Browser Compatibility

- Chrome 80+, Firefox 78+, Safari 14+, Edge 80+
- Requires: localStorage, CSS custom properties, IntersectionObserver, TreeWalker
- Works on `file://` protocol (with localStorage limitations for images)
- Works on `localhost` (full feature set including image upload)

## Known Limitations

1. **localStorage size** — ~5MB limit. Image-heavy usage without a server will fill up.
2. **Comment markers on dynamic content** — markers use text search to find their anchor. If the text is rewritten, markers fall back to stored Y-coordinates.
3. **Cross-origin data** — localStorage is per-origin. Use Export/Import to move data between `file://` and `localhost`.
4. **Tab renames** — if tabs are renamed, comments become "orphans" and show on all tabs until resolved.
5. **No real-time collaboration** — single-user system, no syncing between browsers.
