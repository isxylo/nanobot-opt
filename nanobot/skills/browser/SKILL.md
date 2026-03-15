---
name: browser
description: Use native browser tools (Camoufox) to open pages, extract rendered content, click elements, and capture screenshots.
metadata: {"nanobot":{"emoji":"🦊"}}
---

# Browser

Use this skill when `web_fetch` cannot access content due to JS rendering, Cloudflare challenge, or anti-bot protection.

## Quick Flow

1. Open page
```text
browser_open(url)
```

2. Read rendered text
```text
page_get_text()
```

3. If needed, inspect full markup
```text
page_get_html()
```

4. Interact then re-read
```text
page_click(selector)
page_get_text()
```

5. Capture proof/debug image
```text
page_screenshot()
```

## Selection Rules

- Prefer `web_fetch` for simple static pages.
- Switch to browser tools for login, dynamic content, or anti-bot pages.
- After any click that triggers navigation or lazy loading, call `page_get_text()` again.
