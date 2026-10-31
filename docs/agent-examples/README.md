# Agent-file experiment: Rust + OpenRouter

These definitions illustrate a proposed format, not a stable core loader API:

- `source-code-analyst.json` analyzes source code and delegates focused security work.
- `security-reviewer.json` implements that delegated review with the same read-only tools.
- [`../../rust/examples/agent_file.rs`](../../rust/examples/agent_file.rs) is a runnable host using the real Rust `Agent` request/completion loop and native `SubagentTool` integration. The core owns refinement and child invocation history; this example owns loading, validation, templates, filesystem execution, and OpenRouter adaptation.

## Run

Put the OpenRouter API key (one line) in `~/secret/lockstep-loop.openrouter.key`.
Keep that file **outside the repository root you expose to the tools**. The example
reads it only at runtime and does not log it. Do not use your home directory as the
repository root. Inspected source and prompts are sent to OpenRouter and its model
provider; choose a repository you are authorized to send there.

From the workspace root:

```sh
cargo run -p lockstep-loop --example agent_file -- \
  --root /path/to/repository \
  --inputs '{"path":"src","question":"Explain the entry points and find correctness or security risks."}'
```

The default definition is located relative to the Rust package's compile-time
manifest directory. To select a different definition:

```sh
cargo run -p lockstep-loop --example agent_file -- \
  --agent docs/agent-examples/security-reviewer.json \
  --root /path/to/repository \
  --inputs '{"path":"src","focus":"Trace untrusted paths to filesystem access."}' \
  --max-turns 20 --timeout-secs 180
```

`--inputs-file PATH` avoids putting potentially sensitive input bindings in shell
history. `--model ID` overrides the model for **all** agents, including descendants;
otherwise each agent uses its own `manifest.model`. No failure-triggered model fallback is performed. Use
`--help` for all options. Stdout contains the final Markdown answer; stderr contains
brief progress. Failures exit nonzero.

### Automatic history cache

The runner automatically loads and saves a sibling JSON history bundle:

- `source-code-analyst.json` → `source-code-analyst.state.json`
- Running `security-reviewer.json` directly → `security-reviewer.state.json`

The location follows the resolved agent definition, not the working directory.
No extra flag is needed. A parent's bundle includes its child invocation histories;
it does not separately publish those children into their standalone cache files.

The bundle retains multiple instantiated conversations. Selection compares rendered
initial messages, ordered tools and nested contracts, plus the declarative
templates/schemas of linked children. Names and unrelated manifest tags/versions do
not invalidate reasoning. As an external host policy, this example also separates
histories by canonical repository root and **effective model routing for the whole
linked agent tree**. Changing a child model isolates the parent's cached history
as well. This is not a core structural-compatibility rule. Within a matching
configuration, tools run again: unchanged observations reuse reasoning, and changed
observations branch. A fully cached run needs no API key and sends no model requests.

Only successful, output-validated investigations save history. A sibling temporary
file is flushed and synced before atomic replacement; on Unix the published file
has owner-only permissions and the containing directory is synced. Publication
includes every reachable record, including child invocation bases and results,
with fresh UUID writer namespaces. It rewrites the complete bundle rather than
performing incremental disk updates. Other inputs/configurations are retained.
Concurrent invocations are **last publication wins without merging**.

These caches contain prompts, source evidence, model responses and child history.
Keep them private; generated `*.state.json` and staging files are gitignored.
The active cache and its staging files are excluded from filesystem tool reads,
listings and searches so saving history does not alter the source observations.
This exclusion is not a general protection for other agents' cache files.

Missing cache means a fresh start. Invalid JSON, unsupported format, duplicate or
missing references, and reference cycles fail explicitly; move the cache aside
to start fresh. Load only trusted caches: reference checks are not exhaustive
semantic validation of arbitrary history. The format is Rust-example-specific,
with no cross-version or cross-language compatibility guarantee. Loading is eager,
and cache size/entry retention are currently unbounded.

### Model and thinking

Set a model separately in each agent JSON:

```json
"manifest": {
  "model": "deepseek/deepseek-v4.1-flash-1017",
  "tags": ["source-code"]
}
```

The complete manifest is attached as [opaque runtime metadata](../runtime-metadata.md).
Each child receives its own manifest, not the parent's. The OpenRouter handler
reads the originating request's metadata; it never injects the manifest into
model-visible messages. Model precedence is:

1. Explicit `--model ID` (overrides parent and every descendant).
2. That agent's `manifest.model`.
3. The built-in default below.

A present `model` must be a nonempty string. Metadata is reattached from current
definitions when loading history, not recovered from core snapshots. Both bundled
definitions currently select the same model; edit either manifest independently.

The default is exactly **`deepseek/deepseek-v4.1-flash-1017`**, verified against the
[OpenRouter model catalog](https://openrouter.ai/api/v1/models). Standard/default
DeepSeek thinking is sent as `"reasoning": {"effort": "high"}`. Despite its name,
`high` is this model's default, not its maximum (`max`). At implementation time the
catalog advertised `max`, `high`, `low`, and `none`, with default `high`; `medium`
was not an advertised effort. See also
[DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode) and
[OpenRouter reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
A model override keeps the same `high` effort setting; choose a compatible model.

Requests use HTTPS chat completions, a 16,384-token output limit including thinking,
a 15-second connection timeout, and a configurable whole-request timeout. HTTP
redirects and automatic retries are disabled. Tool-turn assistant messages retain
provider reasoning fields verbatim for subsequent tool turns. Incomplete responses
(including `finish_reason: length`) fail rather than becoming final answers.

## Implemented format profile

- `format_version: 1`; manifest must be a JSON object. It stays opaque to the core;
  this host interprets only `manifest.model`.
- Input bindings are JSON objects validated against the declared JSON Schema.
- JSON Schema Draft 2020-12 is used for inputs, tool arguments, and outputs.
  Only document-local `$ref`/`$dynamicRef` references are accepted; there is no
  external schema retrieval. `format` is annotation-only by default, not an
  additional format-assertion vocabulary.
- `{{input_name}}` substitution is single-pass. Strings interpolate verbatim; other
  JSON values use compact JSON encoding. Dotted lookups, expressions, filters,
  recursive substitution, and delimiter escaping are not supported. Unknown,
  missing, or malformed references fail. Initial roles are system/user/assistant;
  tool messages and initial tool calls are not supported by this profile.
- Subagent links resolve relative to the containing definition. The child input
  schema becomes the parent-facing tool schema; call arguments instantiate the
  child's messages. Children inherit the repository handle, not parent inputs.
  Cycles and definition trees deeper than eight agents are rejected.
- Output source is `last_answer`; the current text adapter returns a JSON string
  and validates it against the output schema. It does not parse Markdown as JSON.
  Structured final-tool outputs are deliberately not implemented yet. Invalid
  model arguments or output stop the run; there is no repair/retry loop.
- The global model-turn limit (default 1,000 fresh requests) covers the parent and all descendants. Exhaustion
  fails explicitly without fabricating an answer. Child failures abort the run;
  child failure-to-tool-result recovery is not implemented by this example.
- The runner uses one stable repository environment per invocation and automatically
  restores/publishes matching history beside the agent definition (see above).
  It does not persist pending work or provide interrupted-run crash recovery.

## Filesystem tools

All paths are relative to the supplied root. Parent traversal, absolute paths, and
symlink traversal are rejected. A listed symlink is not followed. Tools are
read-only. The caller must provide a stable repository tree: canonicalization and
symlink checks are **not an OS sandbox** against concurrent filesystem mutation or
hard links. Any regular file within the root is eligible for inspection; `.git`
is omitted from recursive search but is not a general access-control boundary.

- `read_file(path, from_line, num_of_lines)`: 1-based lines, numbered JSON line
  records, total line count, and explicit truncation. UTF-8 regular files up to
  1 MiB; at most 1,000 lines and 64 KiB of line text per response. A line too large
  for the response is omitted and truncation is reported.
- `list_dir(path)`: sorted immediate entries with file/directory/symlink/other kind.
  At most 1,000 entries and 64 KiB of names; directories exceeding 20,000 entries
  fail rather than returning nondeterministic filesystem-order subsets.
- `string_search(path_wildcard, text_wildcard)`: case-sensitive whole-path and
  whole-line matching. `*` and `?` match characters; in paths they do not cross `/`.
  A whole path segment `**` matches zero or more segments. Brackets, braces, and
  backslashes are literal, not glob/regex syntax. Wrap text in `*` for substring
  search. Search is sorted, skips `.git` and symlinks, stops at 20,000 visited
  entries / 1,000 matches / 64 KiB of path+line text, and reports truncation and
  skipped/unreadable/oversized/binary file counts. Patterns are at most 512 bytes.

Tool I/O failures become JSON error results so the model can choose another
inspection. Validation and provider errors fail the investigation. Size limits
refer to source text/name bytes, not JSON escaping overhead. Source content is
untrusted data; the example definitions explicitly tell the model not to treat
repository text as instructions.

## Offline verification

```sh
cargo test --example agent_file
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The example's tests exercise real parent/child coordination through a local mock
HTTP server, JSON schema/template failures, provider adaptation and reasoning
preservation, per-agent model routing without metadata leakage, model-policy cache
isolation, budgets, filesystem tools, and symlink/traversal rejection. They do
not load the real API key or send repository content to OpenRouter. Cache tests
round-trip parent/child histories through disk, verify zero-model replay and changed
source branching, preserve multiple configurations, check failed-run nonpublication,
and reject corrupt references. Live inference has not been used to verify this example.
