import type { CallbackRenderState } from "./types.js"

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;")
}

export function renderOAuthResultHtml(state: CallbackRenderState): string {
  const title = state.ok ? "Authentication complete" : "Authentication failed"
  const body = state.ok ? "You can return to the agent controller." : state.error
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>${escapeHtml(title)}</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 3rem; line-height: 1.5; }
    main { max-width: 42rem; }
  </style>
</head>
<body>
  <main>
    <h1>${escapeHtml(title)}</h1>
    <p>${escapeHtml(body)}</p>
  </main>
</body>
</html>`
}
