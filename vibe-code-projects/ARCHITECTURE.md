# Learning Hub — Architecture Guide

> **CLAUDE CODE: START HERE.** This is the single source of truth for the Learning Hub project. If you are a new session, read this entire file before doing anything. It tells you what every file does, how pages connect, how the comment/annotation system works, how to process user edits, and what conventions to follow. Do not guess — the answers are here.

## What Is This Project?

The Learning Hub is a local website (no server required) for self-directed learning. It contains interactive study guides (HTML pages), a comment/annotation system, a prompt builder for creating new guides, and an export system that generates ready-to-paste prompts for Claude Code. The user is a **beginner coder** — assume little technical knowledge in all content.

**The user's workflow:**
1. Opens `index.html` in a browser (the homepage)
2. Clicks into a study guide to read it
3. Highlights text they want changed/explained → leaves a comment or uses Quick Fix
4. Exports comments as a prompt → pastes into Claude Code
5. Claude Code reads this architecture file, understands the system, then edits the study guide

**Your job when you receive an exported prompt:** Read this file first. Then edit the target file following the conventions below. Save a changelog entry. Wrap new content in highlight spans.

## Folder Structure

```
vibe-code-projects/
  index.html              ← Homepage with card grid + queued topics
  new-topic.html           ← Prompt builder for new study guides
  r-study-guide.html       ← Socratic Q&A page for learning R
  change-file.html         ← View/export all annotations + custom instructions
  learning-hub.css         ← All shared styles (light theme, sidebar, navbar, comments)
  learning-hub.js          ← Navbar, sidebar, inline comments, search — injected on every page
  ARCHITECTURE.md          ← This file
  thumbnails/              ← Screenshot images for homepage cards
  AI_study_guides/         ← All study guide HTML files
    *.html                 ← Each guide is a self-contained HTML file
    prompts/               ← Generation prompts and changelogs for each guide
      *.prompt.txt         ← Named to match the guide (e.g., github-complete-guide.prompt.txt)
```

## How Pages Connect

Every HTML page (including study guides) loads two shared files at the bottom, before `</body>`:

```html
<link rel="stylesheet" href="../learning-hub.css">
<script src="../learning-hub.js"></script>
```

For pages in the root folder (not in AI_study_guides/), the paths have no `../` prefix:

```html
<link rel="stylesheet" href="learning-hub.css">
<script src="learning-hub.js"></script>
```

These two files inject the navbar and annotation sidebar onto every page automatically.

## Navbar

Injected by `learning-hub.js`. Contains:
- **Learning Hub** brand link (goes to index.html)
- **Home**, **+ New Topic**, **R Guide**, **Change File** navigation links
- **Search bar** — searches study guide titles/categories
- **Comment Mode** button — disables all page links so you can highlight text freely
- **Export Comments** button — copies all comments on the current page as a batch prompt

## Annotation System (Two Modes)

### Quick Fix (Sidebar)
- Collapsed by default on the right edge
- Toggle with the tab button or **Ctrl+Shift+A**
- Highlight text → click "Quick Fix" → describe the change → click "Copy Prompt for Claude Code"
- Generates a self-contained prompt with the file path, highlighted text, instruction, and guidelines

### Inline Comments (Google Docs style)
- Highlight text → click "Comment" → type note → Save
- A yellow dot appears in the right margin marking the comment
- Click any dot to view/delete the comment
- Click "Export Comments" in navbar to copy all comments as a batch prompt
- Comments stored in localStorage under key `learningHubComments`

## Custom Instructions

Stored in localStorage under key `learningHubCustomInstructions`. Editable from:
- The sidebar (bottom section on every page)
- The Change File page (change-file.html)

Custom instructions are automatically included in:
- Quick Fix prompts
- Batch Export prompts
- New Topic generated prompts

## Data Storage (all localStorage)

| Key | What it stores |
|-----|---------------|
| `learningHubComments` | Array of comment objects (highlighted text, note, page, position) |
| `learningHubCustomInstructions` | Free-text custom instructions string |
| `learningHubExportVersion` | Auto-incrementing version number for exports |
| `learningHubQueue` | Array of queued topic objects (title, link) |

## Study Guide File Format

Each guide in `AI_study_guides/` is a **self-contained HTML file** with:
- Inline `<style>` block (each guide has its own theme/colors)
- Inline `<script>` for any interactive features (tabs, demos)
- At the end, before `</body>`, the two shared file links (CSS + JS)

### Required structure for new guides:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>[Guide Title] — Learning Hub</title>
  <style>
    /* Guide-specific styles here */
  </style>
</head>
<body>
  <!-- Guide content here -->

  <link rel="stylesheet" href="../learning-hub.css">
  <script src="../learning-hub.js"></script>
</body>
</html>
```

### Style guidelines for new guides:
- Light, clean theme (white background, Google Docs adjacent)
- Georgia serif for body text, Arial for headings
- Include "Key Insight" callout boxes for important concepts
- Include interactive examples and visual demonstrations where possible
- Include a quick-reference table at the end
- Assume the reader knows very little about code
- If prerequisite knowledge is needed, lay it out in outline format:
  - To understand X, you need to understand Y (indented beneath it)

## Homepage Card System

Cards on `index.html` are defined in a JavaScript array and rendered dynamically. Each card has:
- `file` — filename in AI_study_guides/ (or path for tool pages)
- `title` — display title
- `desc` — short description
- `thumb` — thumbnail image path (in thumbnails/ folder)

To add a new guide to the homepage, add an entry to the `guides` array in `index.html`.

## Queued Topics

The "Queued Topics" section at the bottom of the homepage lets users save topic ideas with optional links. Stored in localStorage under `learningHubQueue`. Each queued item has a "Build This Topic" link to the New Topic prompt builder.

## Prompt Changelog System

Every study guide has a corresponding `.prompt.txt` file in `AI_study_guides/prompts/`. This file contains:
1. The original generation prompt (saved when the page is first created)
2. Dated addenda for every subsequent edit (appended by Claude Code when processing quick fix or batch prompts)

When creating a new guide, save the full prompt as `AI_study_guides/prompts/[guide-name].prompt.txt`.
When editing an existing guide, append the edit request with a timestamp header to the existing `.prompt.txt` file.

## Comment Lifecycle & Implementation Tracking

Comments go through four states:
1. **Active** (yellow dot) — new, unprocessed. Included in "Export Comments."
2. **Exported** (blue dot) — prompt was copied. No longer included in Export. Visible on page.
3. **Implemented** (green dot with checkmark) — Claude Code edited the page to address this comment. Can be replied to; replies become new exportable items.
4. **Resolved** (hidden) — user clicked Resolve. Viewable only via "Show Resolved Comments."

**How implementation tracking works:**
- Each comment has a unique ID (e.g., `m1abc2def3`).
- The exported prompt lists comment IDs and instructs Claude Code to add an HTML comment at the end of the file after making changes: `<!-- lh-implemented: id1, id2, id3 -->`
- On page load, the Learning Hub JS scans the HTML for this marker, reads the IDs, and auto-marks those comments as "implemented" in localStorage.
- **You MUST add this HTML comment when you finish implementing comments.** Without it, comments won't transition to "implemented" state.

**Replying to implemented comments:**
- Users can reply to implemented comments. New replies are flagged as exportable.
- The next "Export Comments" will include those replies as follow-ups, with context about the original comment.
- The exported prompt will note: "This is a follow-up on a previously implemented comment."

## The localStorage Bridge Pattern

Claude Code edits files on disk. The browser stores user data (comments, queue items, settings) in localStorage. These are separate systems that cannot directly communicate. The "localStorage bridge" uses HTML comment markers as a one-way message channel from Claude Code to the browser:

1. Claude Code writes an HTML comment marker into a file (e.g., `<!-- lh-implemented: id1 -->`)
2. The browser loads the file and the Learning Hub JS scans for markers
3. The JS applies the marker's instructions to localStorage
4. The UI updates to reflect the change

### Two types of markers

| Marker | Purpose | Persistent? | Remove after use? |
|--------|---------|-------------|-------------------|
| `<!-- lh-implemented: id1, id2 -->` | Marks comments as implemented | YES — leave forever | No, never remove |
| `<!-- lh-queue-update: {...} -->` | Updates queue item data | NO — one-shot | **YES — Claude Code must remove in the SAME edit session** |

**Why the difference:** Implementation markers are idempotent (setting `status = 'implemented'` twice has no effect). Queue update markers with `appendNotes` are NOT idempotent (they append text, so re-applying duplicates content). The JS has a guard (`indexOf` check to prevent duplicate appends), but the proper fix is to remove the marker after one refresh cycle.

### Queue update marker workflow (step by step)

1. User leaves a comment on a queue item asking to change its notes/title
2. User exports comments → Claude Code receives the prompt
3. Claude Code writes `<!-- lh-queue-update: {"title":"Item Name","appendNotes":"new text"} -->` before `</body>`
4. Claude Code also writes `<!-- lh-implemented: commentId -->` for the comment
5. **Claude Code then immediately removes the `lh-queue-update` marker from the file** (in the same edit session)
6. User is told to refresh the browser
7. On refresh: JS finds the marker (from the server's cached response), applies the update to localStorage, triggers `renderQueue()`
8. Next refresh: marker is gone from the file, no re-application

**If Claude Code forgets to remove the marker:** The `appendNotes` guard (`indexOf` check) prevents duplicate appends, but `notes` (full replacement) would overwrite user edits. Always remove.

### Queue update syntax

```html
<!-- Append to notes (use when adding items to a list): -->
<!-- lh-queue-update: {"title":"Item Name","appendNotes":"text to add"} -->

<!-- Replace notes entirely: -->
<!-- lh-queue-update: {"id":"abc123","notes":"full replacement"} -->

<!-- Update title: -->
<!-- lh-queue-update: {"id":"abc123","title":"New Title"} -->
```

Match by `id` (from `data-id` on `.queue-item` in DOM) or by `title`. Must have at least one.

### Known issues and fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Queue notes not updating | `if (!update.id) return;` skipped title-only matches | Changed to `if (!update.id && !update.title) return;` |
| Queue UI not refreshing | `renderQueue()` ran before scan applied updates | Added `window.renderQueue()` call after scan |
| Duplicate appends on refresh | `lh-queue-update` marker left in file | Added `indexOf` guard in JS + documented that Claude Code must remove markers |
| Scan not finding comments | `innerHTML` strips comments in some browsers | Added DOM TreeWalker fallback |
| Markers outside queue items | Container detection not matching `.queue-item` | Added `data-id`, `position: relative`, fixed `closest()` |
| Markers stacking | All got `top: 50%` | Offset tracker, 20px spacing |
| Comments lost on edit | Edit deleted item, created new ID | In-place update preserving ID |
| Long code blocks breaking page layout | `<pre>` with `overflow-x:auto` stretches parent containers | Use `white-space:pre-wrap;word-break:break-all;` on `<pre>` blocks. Add `overflow:hidden` to parent containers. Always add a Copy button for long code blocks so users don't need to select/copy manually. |

## Comment Processing Methodology

When processing user comments on study guides, follow these rules:

1. **Cross-reference between tabs/sections**: If a comment asks a question that belongs in a different section/tab, respond at the comment location with "This is covered in the [Tab Name] tab" and add the actual content to the appropriate tab/section.
2. **Don't duplicate — redirect**: If content already exists elsewhere on the page that answers the question, point to it rather than repeating it.
3. **Harvest methodology**: If a user's comment contains instructions about HOW they want future comments handled (not just what to change), extract that methodology and add it to this architecture file.
4. **Comment types**: Read each comment carefully to determine if it's a text edit, a question, a reorganization request, or a bug report. Handle accordingly.
5. **Change highlighting**: Wrap new/modified content in `<span class="lh-new-content">` tags. Remove old highlight tags from previous edits first. The highlighting should be SUBTLE — just a light blue text background, not a separate box or disruptive formatting.
6. **Thread context**: If a comment has thread replies, read the full thread to understand the evolving request before making changes.

### CRITICAL: How to integrate new content

7. **NEVER write content that sounds like it's answering an invisible question.** The user's comment is NOT visible on the page — only the study guide text is. New content must read as if it was always part of the document. Bad: "Yes, exactly! GitHub is a GUI for Git." Good: "GitHub provides a visual, web-based interface for Git, so you can click buttons instead of typing terminal commands."
8. **Integrate, don't isolate.** When a comment asks for more detail on a topic, WEAVE the new information into the existing text flow. Don't always create a separate callout/info box. Expand the existing paragraph, add a sentence, extend a bullet point. Use callout boxes only for genuinely separate "key insight" or "warning" content — not for every addition.
9. **The `lh-new-content` span is just a text-level highlight.** It should wrap inline text, not entire div blocks. Think of it like a yellow highlighter on a printed page — it marks what's new without changing the layout.

## Key Design Decisions

1. **Server optional** — basic features work from file:// protocol, but image uploads require `node image-server.js` (localhost:3000). The image server also serves static files, replacing `python -m http.server`.
2. **No build tools** — pure HTML/CSS/JS, no npm/webpack (node_modules in the project root is for Puppeteer thumbnails only)
3. **localStorage for persistence** — no database, no accounts
4. **lh- CSS prefix** — all injected styles use `lh-` prefix with `!important` to avoid conflicts with guide styles
5. **Comment Mode** — disables page links so text can be highlighted on clickable elements
6. **Sticky tab bars** — on pages with tab navigation, `learning-hub.js` auto-detects the tab container and adds `lh-sticky-tabs` class, making it stick below the navbar on scroll. New study guides with tabs get this for free.
7. **Reserve hover effects for functional elements** — only apply hover highlights/color changes to elements that are clickable (links, buttons, interactive controls). Static list items, table rows, or text blocks should NOT have hover effects unless clicking them does something.
8. **Tab-aware comment markers** — on pages with tabs, comment dots only appear for the currently active tab. All comments still count in badges/exports regardless of tab.
9. **Section sub-navigation bar** — when a single tab contains multiple sections (like a merged chapter), add a horizontal sub-navigation bar that sticks below the chapter tabs. Requirements:
   - Blue background (`#1a73e8`), white text
   - Same font size as the tab bar (`0.82rem`)
   - Sticky, positioned directly below the tab bar with no vertical gap
   - Highlights the current section as the user scrolls (use Intersection Observer)
   - Hides when the user switches to a different tab
   - Links use anchor IDs to scroll to sections
   - CSS class: `.ch1-subnav` (reuse this class for all chapter sub-navs)
   - Each chapter gets its own `<nav>` with a unique ID (e.g., `ch1Subnav`, `ch2Subnav`)
   - The tab-switching JS shows/hides the correct sub-nav when chapters are switched
   - The scroll-spy JS must be set up for each sub-nav independently (use a shared `setupScrollSpy` function)
   - When adding a new chapter: add a `<nav class="ch1-subnav" id="chNSubnav">`, hide it by default, and register it in the tab-switch and scroll-spy code

## Image System

Images in the Learning Hub are handled in two ways:

### With image server running (`node image-server.js`)
- User drops an image into a comment → it's resized, uploaded to `/upload-image` endpoint → saved to `images/` folder → comment stores the file path (`images/comment-12345.png`)
- When Claude Code implements the comment, it references the image at its file path
- Images persist as real files on disk — no localStorage limits

### Without image server (file:// protocol)
- User drops an image → it's resized and stored as base64 in localStorage with the comment
- Works but limited by localStorage size (~5MB total)
- The exported prompt includes the base64 inline

### When Claude Code processes an image comment
- If the image is a file path (e.g., `images/comment-12345.png`), reference it directly in the HTML
- If the image is base64, decode it, save to `images/`, and reference the saved file
- Use agent-browser (`agent-browser navigate [url] && agent-browser screenshot [path]`) for fetching screenshots from websites efficiently

### Agent-Browser
Installed globally. Fast headless browser for AI agents. Use for:
- Taking screenshots of websites: `agent-browser navigate [url] && agent-browser screenshot [path]`
- Extracting content from web pages: `agent-browser navigate [url] && agent-browser text`
- Getting accessibility snapshots: `agent-browser navigate [url] && agent-browser snapshot`

## Scratch Pad

The scratch pad is a per-page note-taking system. Unlike comments (which drive changes to the page content), scratch pad notes are personal user notes that live IN the page source.

**How it works:**
1. User clicks "Add Scratch Pad" in the Comments dropdown → a new "Scratch Pad" tab appears
2. User types a note and clicks "Add Note"
3. The note is saved to localStorage immediately (instant UI) AND written to the HTML file via the image server's `/save-scratchpad` endpoint
4. Notes are stored in the HTML file inside a `<!-- lh-scratchpad-start -->` / `<!-- lh-scratchpad-end -->` block
5. On page load, notes are read from both the HTML source and localStorage, merged by ID (highest revision wins)
6. Notes are editable (contenteditable) — editing increments the rev number
7. Notes can be deleted

**Server requirement:** Notes only save permanently when the image server (`node image-server.js`) is running. Without the server, notes save to localStorage only (temporary, per-origin).

**Note object structure:**
```json
{
  "id": "sp-m1abc2def3",
  "rev": 1,
  "text": "the note content",
  "timestamp": "2026-03-29T00:00:00.000Z"
}
```

**HTML storage format:**
```html
<!-- lh-scratchpad-start -->
<div id="lh-scratchpad-data" class="tab-content" style="display:none;">
  <div class="lh-scratchpad-note" data-id="sp-abc" data-rev="1" data-timestamp="...">Note text here</div>
</div>
<!-- lh-scratchpad-end -->
```

## Restructuring Safety Rules

When reorganizing study guide pages (renaming tabs, merging content, splitting pages), follow these rules to avoid breaking the comment system:

1. **Comments survive text changes.** Comments are anchored to `highlightedText` (a string search). If the exact text still exists somewhere on the page, the marker will find it. If the text was rewritten, the marker falls back to its stored Y-coordinate — less accurate but still visible.

2. **Comments survive tab renames.** If a comment references a tab name that no longer exists (e.g., `activeTab: "Big Picture"` but the tab was renamed to "Chapter 1"), the comment becomes an "orphan" and shows on ALL tabs rather than disappearing. This is intentional — orphaned comments are visible everywhere until the user resolves them.

3. **Comments survive tab merges.** Merging multiple tabs into one works because orphan recovery kicks in. The comments will show on the new merged tab since their original tab name no longer exists.

4. **Comments do NOT survive page file renames.** Comments are keyed to `pageFile` (the filename). If you rename `github-complete-guide.html` to `github-guide.html`, all comments for the old filename become invisible. Avoid renaming files. If you must rename, update localStorage manually.

5. **The `<!-- lh-implemented: -->` markers survive restructuring** because they're just HTML comments with IDs. They don't reference tab names or text positions.

6. **When merging tabs:** keep the same `data-tab` IDs if possible. If you create a new tab called "Chapter 1" with `data-tab="chapter-1"`, the tab-switching JS will work fine — it only cares about `data-tab` attributes, not the visible text on the button.

7. **Always test after restructuring:** open the page, check the Comments dropdown for the status counters, verify markers appear on the correct tabs.
