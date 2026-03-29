# Comment System Component — Architecture

> **AGENT: START HERE.** This document describes a standalone, reusable comment/annotation system that can be added to any HTML website. Read this entire file before making changes.

## What This Is

A browser-based annotation system that lets users:
1. Highlight text on any page → leave comments (text edits, questions, bug reports)
2. Attach screenshots to comments via drag/drop
3. Thread replies on comments
4. Export all comments as a structured prompt for AI coding agents (Claude Code, Cursor, etc.)
5. Track which comments have been implemented by the agent
6. Resolve implemented comments to clean up the view

The system is designed for a human↔AI feedback loop: humans annotate, agents implement, humans verify.

## Files

```
component-comment_infrastructure/
  comment-system.js    ← Core JavaScript (IIFE, exposes CommentSystem global)
  comment-system.css   ← All styles (CSS custom properties for theming)
  template.html        ← Demo page showing the system in action
  ARCHITECTURE.md      ← This file
  README.md            ← Full documentation for developers/agents
```

## How to Add to Any Website

### Minimal integration (2 lines):

```html
<link rel="stylesheet" href="path/to/comment-system.css">
<script src="path/to/comment-system.js"></script>
```

The system auto-initializes with sensible defaults. For customization:

```html
<script>
  CommentSystem.init({
    storageKey: 'myproject_comments',
    projectRoot: 'my-project/',
    architectureFile: 'my-project/ARCHITECTURE.md'
  });
</script>
```

### Full config options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `storageKey` | string | `'comments'` | localStorage key for comments |
| `instructionsKey` | string | `'customInstructions'` | localStorage key for custom instructions |
| `pageFile` | string | auto-detect | Current page filename |
| `pageTitle` | string | auto-detect | Page title for exports |
| `projectRoot` | string | `''` | File path prefix in exported prompts |
| `imageUploadEndpoint` | string | `null` | Server endpoint for image uploads. Null = base64 in localStorage |
| `tabSelector` | string | `'.tab-btn'` | CSS selector for tab buttons. Null = no tab awareness |
| `dynamicContainers` | array | `['.queue-item']` | CSS selectors for dynamic content that needs special marker handling |
| `navbarHeight` | number | `56` | Pixels of top padding for the navbar |
| `promptFilePrefix` | string | `'prompts/'` | Path prefix for prompt changelog files |
| `architectureFile` | string | `null` | Path to architecture file, referenced in exported prompts |
| `onReady` | function | `null` | Callback when system is fully initialized |

## Data Storage

All data lives in the browser's localStorage. The system cannot write to the filesystem.

| Key | Content |
|-----|---------|
| `{storageKey}` | JSON array of comment objects |
| `{instructionsKey}` | Free-text custom instructions string |

### Comment object structure:

```json
{
  "id": "m1abc2def3",
  "timestamp": "2026-03-29T00:00:00.000Z",
  "pageName": "Page Title",
  "pageFile": "page.html",
  "highlightedText": "the text the user selected",
  "note": "the user's comment",
  "image": "images/comment-123.png or data:image/...",
  "x": 500,
  "y": 300,
  "status": "active|implemented|resolved",
  "resolved": false,
  "replies": [
    { "text": "reply text", "timestamp": "...", "implemented": false }
  ],
  "context": {
    "cssPath": "div.container > p.intro",
    "nearestHeading": "Section Title",
    "activeTab": "Tab Name",
    "elementTag": "p",
    "elementClasses": "intro"
  }
}
```

## Comment Lifecycle

```
Active (yellow dot)
  ↓ User clicks "Export Comments"
  ↓ User pastes prompt into AI agent
  ↓ Agent edits the file and adds <!-- lh-implemented: id1, id2 -->
  ↓ User refreshes the page
Implemented (green checkmark)
  ↓ User can reply (replies become new exportable items)
  ↓ User clicks "Resolve"
Resolved (hidden, viewable via "Show Resolved Comments")
  ↓ User can "Un-resolve" to revert to Implemented
```

## The localStorage Bridge

The AI agent edits files on disk. The browser stores comments in localStorage. These systems communicate via HTML comment markers:

### Implementation markers (permanent):
```html
<!-- lh-implemented: id1, id2, id3 -->
```
On page load, `scanForImplemented()` reads these and marks matching comments as "implemented."

### How the agent should process exported comments:

1. Read the architecture file referenced in the prompt
2. Edit the target file to address each comment
3. Add `<!-- lh-implemented: id1, id2 -->` before `</body>`
4. Wrap new/changed content in `<span class="lh-new-content">...</span>`
5. Remove old `lh-new-content` spans first (unwrap, keep content)
6. Append a changelog entry to the prompt file

## Exported Prompt Format

When the user clicks "Export Comments," the system generates a structured prompt:

```
FIRST (if you haven't already): Read `{architectureFile}`

THIS IS AN EXPORTED COMMENT BATCH from the annotation system.

Edit the file `{projectRoot}{pageFile}`.

Guidelines:
- TEXT EDIT: replace the highlighted text
- QUESTION: add explanation near the text
- BUG REPORT: use DOM path to find and fix
- Wrap new content in <span class="lh-new-content">
- Add <!-- lh-implemented: ids --> before </body>

---

### Comment 1 [ID: abc123]

Find this text:
> highlighted text

My comment: user's note

Location context:
- Active tab: "Tab Name"
- Under heading: "Section Title"
- DOM path: `div.container > p`
```

## UI Components

| Component | Trigger | Purpose |
|-----------|---------|---------|
| **Navbar** | Auto-injected | Comments dropdown + Export button |
| **Comment Mode** | Dropdown menu | Disables links for text selection |
| **Floating Bar** | Text selection | Quick Fix / Comment buttons |
| **Quick Fix Sidebar** | Click "Quick Fix" | Single-comment prompt builder |
| **Comment Popup** | Click "Comment" | Full comment form with image upload |
| **Markers** | Auto-rendered | Yellow/green dots in margin |
| **Detail Popup** | Click marker | View comment, reply, resolve |
| **Resolved Overlay** | Dropdown menu | View resolved comments |
| **Export/Import** | Dropdown menu | Data migration between origins |

## Theming

Override CSS custom properties:

```css
:root {
  --cs-accent: #1a73e8;      /* Primary blue */
  --cs-green: #1e8e3e;       /* Implemented state */
  --cs-red: #d93025;         /* Delete/danger */
  --cs-yellow: #fbbc04;      /* Active comment marker */
  --cs-bg: #ffffff;           /* Background */
  --cs-sidebar-width: 380px;  /* Sidebar width */
}
```

## Security Considerations

- Images uploaded via the server endpoint are validated (image MIME types only, max size)
- Comment text is escaped via `escapeHtml()` before rendering (prevents XSS)
- localStorage data is JSON-parsed with try/catch (prevents injection via corrupted data)
- The image server only accepts image files and sanitizes filenames
- No external network requests are made (everything is local)
