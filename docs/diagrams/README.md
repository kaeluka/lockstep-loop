# Diagram sources and previews

[Visual guide](../concepts.md) · [Documentation](../README.md)

| Concept | SVG preview | PNG preview | Editable source |
| --- | --- | --- | --- |
| Product checks: agent, conversation, assets | [SVG](agent-trie.svg) | [PNG](agent-trie.png) | [Excalidraw](agent-trie.excalidraw) |
| Dataset releases: replay | [SVG](replay.svg) | [PNG](replay.png) | [Excalidraw](replay.excalidraw) |
| Supplier evidence: feedback and inheritance | [SVG](feedback.svg) | [PNG](feedback.png) | [Excalidraw](feedback.excalidraw) |
| Research screening: child divergence containment | [SVG](subagents.svg) | [PNG](subagents.png) | [Excalidraw](subagents.excalidraw) |

The four domains illustrate the same generic state machine; none is a built-in
integration or privileged environment type. The guide supplies descriptive alt
text. Labels are intentionally abbreviated;
the library records full evidence, not these diagram summaries. Blue denotes
shared reasoning, green completed values, orange new work/feedback, and purple
child investigations. Text labels convey the meaning without relying on color.

## Edit and render locally

Open a source in a local Excalidraw canvas. Do not use the public share/upload
command. These diagrams were built with `mcp-excalidraw-server` 2.1.2, using its
headless Excalidraw renderer. SVG exports embed fonts; PNGs are fallback previews.

From the repository root, offline rendering needs no live canvas:

```sh
npx -y mcp-excalidraw-server render docs/diagrams/replay.excalidraw \
  --format svg --out docs/diagrams/replay.svg
npx -y mcp-excalidraw-server render docs/diagrams/replay.excalidraw \
  --out docs/diagrams/replay.png
```

For live edits, use an isolated canvas, e.g. set
`EXPRESS_SERVER_URL=http://127.0.0.1:3107` before CLI commands. If a canvas is already
shared, snapshot it first and restore it afterwards. Import the selected source,
edit, then **describe → screenshot → inspect → fix → screenshot again**. Check
label clipping, arrow crossings, spacing, and readability at documentation width.
Export the edited source, then render both previews from that saved source so
font/layout normalization matches a fresh import:

```sh
npx -y mcp-excalidraw-server import docs/diagrams/replay.excalidraw --replace
npx -y mcp-excalidraw-server describe
npx -y mcp-excalidraw-server screenshot --out replay-check.png
npx -y mcp-excalidraw-server export --out docs/diagrams/replay.excalidraw
npx -y mcp-excalidraw-server render docs/diagrams/replay.excalidraw \
  --out docs/diagrams/replay.png
npx -y mcp-excalidraw-server render docs/diagrams/replay.excalidraw \
  --format svg --out docs/diagrams/replay.svg
```

The original layout was drawn on an isolated loopback canvas. The domain-neutral
documentation rewrite updated the saved labels, rendered both formats offline,
and visually checked all four PNGs for clipping, overlaps, and readability.
Contract removal relabels the agent root as “Initial prompt + tool configuration”;
the source and both previews were regenerated locally and the PNG inspected.
No diagram was uploaded to a public service.
