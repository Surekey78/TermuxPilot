# TermuxPilot (`tp`)

A **CLI-first autonomous AI agent that runs natively in Termux on Android**.
It chats with *any* OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter,
Together, DeepSeek, Ollama, llama.cpp server, LM Studio over LAN, …), streams
answers to your terminal, and — as versions land — controls the device, runs
shell commands through a permission-gated tool layer, manages files, and automates
workflows.

No hardcoded provider: all model access goes through one
**OpenAI-compatible provider layer** (`base_url` + `api_key` + `model`) with a
**fallback chain** (cloud → LAN → local).

---

## Roadmap

| Version | Scope | Status |
|---|---|---|
| **v0.1** | Project scaffold, config system + profiles, provider layer (SSE streaming, fallback chain), chat REPL (`tp -i`), one-shot + pipe support | ✅ |
| **v0.2** | Tool layer: shell executor + file ops behind a permission-gated tool router (risk classifier, allow/blocklist, dry-run, secret redaction, audit log); agent loop with native function calling + graceful JSON-mode degradation | ✅ **this release** |
| v0.3 | Termux:API bridge (clipboard, notifications, battery, …) | next |
| v0.4 | Memory (SQLite + sqlite-vec) + local RAG over `~/notes` | planned |
| v0.5 | Voice loop, vision input, background agent daemon | planned |

---

## Install

On-device (Termux, Python 3.12):

```bash
pkg install python
pipx install termuxpilot     # once published to PyPI
# or from source:
git clone https://github.com/Surekey78/TermuxPilot && cd TermuxPilot
pipx install .
```

```bash
tp config init    # writes a commented sample to ~/.termuxpilot/config.yaml
tp -i             # start chatting
```

---

## Configuration

File: `~/.termuxpilot/config.yaml`
Override: `--config-path PATH` (any subcommand) or `$TERMUXPILOT_CONFIG`.

If no config file exists, TermuxPilot falls back to the conventional
`OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_MODEL` environment variables;
if those are absent too, it tells you to run `tp config init`.

### Format

```yaml
default_profile: cloud          # profile used when no --profile is given

# Optional global system prompt (TermuxPilot ships a Termux-aware default).
# system_prompt: |
#   You are TermuxPilot, an assistant on my Android phone, running in Termux.

# Top-level blocks = profile "default" (used without --profile)
# AND the inheritance base for named profiles below.
provider:
  base_url: "https://api.groq.com/openai/v1"
  api_key: "${GROQ_API_KEY}"
  model: "llama-3.3-70b-versatile"
  temperature: 0.3
  max_tokens: 4096
  timeout: 60                   # seconds (default 60)
  # headers:                    # extra headers, e.g. for OpenRouter
  #   HTTP-Referer: "https://myapp"

# Fallback chain — tried in order when the primary fails or times out.
# `fallback` may be a SINGLE provider mapping or a LIST of mappings
# (a list of bare base_url strings is also accepted).
fallback:
  - name: lan-ollama
    base_url: "http://192.168.1.5:11434/v1"
    model: "qwen2.5-coder:14b"
  - name: local-llamacpp
    base_url: "http://localhost:8080/v1"
    api_key: "none"
    model: "qwen2.5-coder-7b-instruct-q4_K_M"

# Named profiles ---------------------------------------------------------
profiles:
  # Inherits everything from the top-level blocks; just given a name.
  cloud: {}

  local:
    provider:
      base_url: "http://localhost:8080/v1"
      api_key: "none"
      model: "qwen2.5-coder-7b-instruct-q4_K_M"
    fallback: []                # explicit [] = no fallback for this profile

  ollama-lan:
    provider:
      base_url: "http://192.168.1.5:11434/v1"
      model: "qwen2.5-coder:14b"
    fallback:
      - base_url: "http://localhost:8080/v1"
        api_key: "none"
        model: "qwen2.5-coder-7b-instruct-q4_K_M"
```

### Rules

- **`fallback` shapes** — a single mapping, a list of mappings, or a list of
  bare `base_url` strings are all valid. Chain order is list order.
- **Profile inheritance** — a named profile merges over the top-level
  `provider` block: set only what differs. If a profile does *not* mention
  `fallback`, it inherits the top-level chain; `fallback: []` disables it.
- **`default_profile`** — required only when there is no top-level
  `provider`/`fallback` block (profiles-only config). Otherwise defaults to
  `default`.
- **Env expansion** — any string supports `${VAR}` and `${VAR:-default}`.
  Unset variable without a default expands to `""`.
- **`api_key`** — omit it or set `"none"` for keyless local servers (no
  `Authorization` header is sent).
- **Precedence** (highest wins):
  CLI flags (`--model`, `--base-url`, `--api-key`, `--temperature`,
  `--max-tokens`, `--timeout`) → selected profile → top-level defaults.

### Profile switching

```bash
tp --profile local -i                 # whole session on the local endpoint
tp --profile ollama-lan "quick task"  # one-shot
```

Inside the REPL the switch is live: `/profile local` rebuilds the provider
chain in place (conversation history is kept), and `/profiles` shows the
table of configured profiles.

---

## Usage

```bash
tp "compress all photos in ~/storage/dcim older than 30 days"
tp -i                                    # interactive agent (rich markdown, streaming)
cat error.log | tp "why is this failing?"   # stdin attached as context
tp --json "ping"                         # machine-readable one-shot (incl. tool calls)
tp --mode safe "check disk usage"        # force a permission mode for this run
tp --dry-run "reorganize ~/notes"        # preview tool calls; nothing is executed
tp --list-profiles                       # show profiles (add --plain for scriptable output)
tp config init [--force]                 # write the sample config
tp config show [--config-path P]         # show the resolved config (keys masked)
tp --model gpt-4o --base-url https://api.openai.com/v1 "..."   # per-run overrides
```

One-shot exit codes: `0` final answer returned · `1` provider failure ·
`2` config/usage error · **`3` incomplete (limit reached, empty or interrupted
provider response)** · `130` user interruption. A final answer is not an
independent verification that every requested operation succeeded; inspect
individual tool results too.

The agent answers through a **tool loop**: it calls tools, sees their
(redacted) output, and continues until it has a final answer
(`agent.max_tool_rounds` caps the loop). Total tool attempts and message size
also have independent limits. `--json` reports `status`, `truncated`, and
`stop_reason`; reaching a limit is **never** reported as `ok: true`.
Completed tool results are kept in the current REPL session if a later provider
call fails, so the next turn can inspect what already happened rather than
blindly repeat it. This is in-memory progress, not restart recovery.

### REPL commands

| Command | Effect |
|---|---|
| `/help` | list commands |
| `/exit`, `/quit` (or Ctrl+D) | leave |
| `/reset` | clear the conversation |
| `/clear` | clear the screen |
| `/profiles` | table of configured profiles |
| `/profile <name>` | switch profile live |
| `/config` | resolved settings for the active profile (key masked) |
| `/mode [safe\|standard\|yolo]` | show or set the permission mode |
| `/dry-run [on\|off]` | preview-only: tool calls are shown, never executed |
| `/tools` | list the available tools |
| `/audit [n]` | last `n` audited tool calls (default 10) |
| `/model <name>`, `/base-url <url>` | session-only overrides |
| `/about` | version + active provider info |

---

## How failover works

The provider chain tries endpoints **in order**: primary first, then each
fallback. A switch happens only for *retryable* failures:

| Failure | Fallback? | Why |
|---|---|---|
| connection refused / DNS / TLS | ✅ | endpoint unreachable (maybe phone-side) |
| timeout (connect or read) | ✅ | |
| HTTP 401 / 403 | ✅ | dead cloud key — a keyless local server may still work |
| HTTP 404 | ✅ | wrong `base_url` path — next endpoint may be right |
| HTTP 408 / 429 / 5xx | ✅ | outage / rate limit |
| HTTP 400 (bad request) | ❌ | the request itself is wrong; switching won't help |
| stream dies **after** visible output | ❌ | failing over would duplicate text you already saw (the partial content is kept) |

When the chain fails completely, `tp --json` reports every attempt:

```json
{"ok": false, "error": "'local-llamacpp' returned HTTP 503",
 "attempts": [{"provider": "cloud", "error": "..."},
              {"provider": "local-llamacpp", "error": "..."}]}
```

Other endpoint quirks handled transparently: servers that reject
`stream_options` (retried once without it) and servers that answer a
streaming request with plain JSON.

### Tool layer & safety (v0.2)

Every capability is a discrete, named **tool** — there is no free-form shell
execution behind the agent's back. In v0.2:

| Tool | Category | What it does |
|---|---|---|
| `run_shell` | execute | bounded foreground command, process-group cleanup, exit code + redacted output (configured cwd, otherwise CLI cwd) |
| `read_file` | read | bounded numbered line windows, including files larger than 1 MB |
| `write_file` | write | create/overwrite up to 1 MB UTF-8, diff preview + atomic replacement |
| `edit_file` | write | exact unique search/replace, diff preview + atomic replacement |
| `move_file` | write | move/rename, refuses overwrite without `overwrite: true` |
| `diff_files` | read | unified diff between two files |

**Permission model** (enforced in the tool router, `tools.mode`):

| Mode | read tools | write/execute tools |
|---|---|---|
| `safe` | allowed | only a conservative subset of inspection commands; other execution and all writes **denied** |
| `standard` | allowed | vetted inspection commands auto-run; unknown or mutating commands require **preview + y/N confirmation** |
| `yolo` | allowed | executed automatically (blocklist still applies) |

**Permission policy, not an OS sandbox:** binaries on `PATH` and installed tools
must be trusted. Automatic shell execution is limited to simple commands such
as `ls`, `cat`, `head`, `grep`, `df`, and pipelines/`&&` combinations of those
commands. Interpreters, `git` (which can invoke configured helpers), `find`,
`awk`, `sed`, unknown commands, substitutions, redirects, and glob expansion
require approval in standard mode and are denied in safe mode. Conservative
false positives, including quoted metacharacters, also require approval.
`yolo` is an explicit opt-out from confirmation, not isolation. Regex
allowlists do not override safe-mode restrictions and are not shell parsers.

Additional safety:

* **risk classifier** flags destructive patterns (`rm -rf /`, `dd of=/dev/…`,
  `mkfs`, fork bombs, `curl | sh`, `sudo`, redirects into `/etc`, …) with
  levels low→critical; high/critical get a red warning panel;
* **config blocklist/allowlist** (regexes): the blocklist denies in *any*
  mode; a non-empty allowlist restricts what may run at all;
* **protected paths** (`/etc`, `/dev`, `/boot`, `/system`, `/vendor`, …)
  escalate file-tool risk to high after resolving relative paths, `..`, and symlinks;
* **secret redaction** masks API keys (OpenAI/GitHub/AWS/Groq/Google/JWT/
  Bearer/`key=value`), private-key blocks, etc. in tool outputs, file diffs,
  previews, errors, and recorded arguments. Shell streams are sanitized before
  bounded retention; private-key state is tracked across lines. This is
  best-effort recognition, not a guarantee of detecting every secret. Avoid
  exposing sensitive files unnecessarily. Audit arguments are always redacted,
  even when model/UI redaction is explicitly disabled;
* **audit log**: every tool decision/execution is appended to
  `~/.termuxpilot/audit.jsonl` (JSON lines) — view with `/audit`;
* **dry-run** (`tools.dry_run`, `--dry-run`, `/dry-run on`): tool calls are
  previewed without execution approval and never applied (blocklists still
  win); read-only file inspections may still run;
* **argument validation**: required fields, built-in parameter types, bounds,
  and unknown fields are checked before previews; booleans/NaN/infinity cannot
  masquerade as valid numeric timeouts;
* **file writes** use a same-directory temporary file and atomic replacement,
  preserve existing permission bits where supported, and reject detected
  changes since the preview. Failed writes leave the original intact. This is
  not a multi-file transaction or a lock against arbitrary concurrent writers.

**Function calling:** `agent.function_calling: auto` (default) sends native
`tools` payloads and, if the endpoint rejects them with HTTP 400, degrades
to JSON-mode prompting (strict `{"thought","tool","args","response"}`
contract + argument validation; plain-text final answers are tolerated for
endpoints that ignore `response_format`) for the rest of the session. Use `native` to
require it or `json` to always use JSON mode.

---

## Long-running and heavy tasks

The execution layer now keeps noisy commands from accumulating all stdout and
stderr in RAM. Both pipes are drained concurrently into bounded, UTF-8-decoded
head/tail buffers. Lines longer than 64 Ki characters are omitted as whole lines
rather than exposing partial secrets. No full command log is written to disk.
Tool results expose `truncated` and `timed_out` when applicable; JSON tool-call
summaries retain at most 500 characters, also with a truncation indicator.

Commands run in their own POSIX process group with stdin closed. Use
non-interactive flags for package/build tools. Timeouts and Ctrl+C send TERM,
then KILL after a configurable grace period, retaining captured partial output.
Unmanaged background descendants are cleaned up on normal completion too.
**Do not use `run_shell` to launch a daemon.** Deliberately detached descendants
and an Android/OS kill of TermuxPilot are outside this cleanup guarantee.

### Runtime limits

Defaults (merge these blocks into your existing provider configuration):

```yaml
tools:
  mode: standard
  redact_secrets: true
  max_output_chars: 30000   # per result, including truncation notes; 128..1000000
  shell:
    timeout: 60            # default per command, seconds
    max_timeout: 3600      # model-supplied timeout cannot exceed this ceiling
    kill_grace: 1          # TERM grace period, 0..30 seconds
    # workdir: "~/project"

agent:
  max_tool_rounds: 8       # model/tool rounds per user turn
  max_tool_calls: 32       # attempts across ALL rounds, including batches/denials
  max_context_chars: 200000 # messages + reserved tool schemas, NOT a token count
  function_calling: auto
```

For a trusted long build, raise `tools.shell.timeout` (for example, to `900`)
and explicitly configure a sufficient `max_timeout`. If `max_timeout` is
omitted, it defaults to the larger of 3600 and the configured default timeout,
so existing long-timeout configurations remain usable. For multi-step work,
raise the round/call budgets deliberately rather than disabling limits.

A batch that would exceed the remaining call budget is rejected **before any
of its tools execute**. Context overflow stops cleanly rather than silently
dropping messages or sending an oversized request. Use smaller reads or
`/reset` as appropriate. This character guard is not a tokenizer, does not
compact history, and cannot guarantee that a particular model's context window
will fit. Usage totals sum counters actually reported across model rounds;
missing provider usage and unreported failed attempts are not estimated.

`tp config show` and REPL `/config` display these limits. Example incomplete
JSON result (additional tool/model fields omitted here):

```json
{"ok": false, "status": "incomplete", "truncated": true,
 "stop_reason": "max_tool_rounds", "rounds": 8}
```

### Large files

- Files over 1 MB require `start_line` and/or `end_line`; they are no longer
  rejected just because the **file** is large when a window was requested.
- `start_line` without `end_line` reads up to 200 lines. Each window selects at
  most 1 MB and 10,000 lines. Continuation hints identify byte-limit stops.
- `read_file` rejects lines over 64 KiB and non-regular files (including
  FIFOs/devices). Edit/diff previews also reject non-regular files. Reads scan
  to the requested line with bounded memory, not
  constant-time random access. They do not scan the suffix just to count lines.
- Edit/diff/write previews remain bounded to 1 MB. For large transformations,
  use an appropriate streaming command with approval instead of loading the
  entire file into a model prompt.
- Piped stdin is capped at 1,000,000 characters; pass a file path for larger
  input. Non-TTY input cannot supply interactive execution approval.

### What this does not provide yet

These limits bound output retention and request size, **not a child process's
RAM/CPU use**. Computation still runs on the phone; provider fallback changes
where inference runs, not where shell commands execute. Real-device thermal,
battery, and Android lifecycle testing is still necessary.

The next reliability milestone is durable SQLite jobs/checkpoints, resumable
workflows with reconciliation of uncertain side effects, resource-aware
scheduling, bounded log artifacts/retention, and optional remote workers.
There is no `tp task` command, restart recovery, background job service,
automatic context summarization, or remote executor in this release.

---

## Architecture

```
CLI / REPL (argparse + rich + prompt_toolkit)
  └─ Agent loop (agent/core.py)      ReAct rounds: model -> tool call -> result -> ...
      ├─ ToolRouter (tools/router.py)  permission mode, allow/blocklist, dry-run,
      │                               confirmation, audit — the single gate
      ├─ tools: run_shell, read/write/edit/move/diff   (tools/)
      ├─ safety.py                   risk classifier, redaction, path guards
      └─ Provider layer (provider/)
          ├─ OpenAICompatibleClient  httpx, SSE streaming, tool-call accumulation
          ├─ ProviderChain           ordered fallback with failover hook
          └─ errors                  retryable / non-retryable policy
      └─ config.py                   YAML: profiles + env expansion + tools/agent blocks
```

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest                                   # unit, integration, and bounded-resource regression tests
python tests/mockserver.py --port 8100   # local OpenAI-compatible server
                                          #   --fail-next N  fail N requests (503)
                                          #   --drop-after N kill the stream mid-way
                                          #   --key K        require Bearer auth
```

Tests spin up real (threaded) mock servers on random ports, so the full
stack — YAML → profiles → chain → SSE → failover → REPL — is covered end to
end without any real provider. Additional regressions cover multi-megabyte
stdout/stderr with bounded Python allocation, 1 GB sparse-file windows,
process-group cancellation, atomic-write failures, permission bypass attempts,
and a 100-step mocked workflow. These are regression checks, not a claim of
full on-device performance certification. GitHub Actions runs the suite on
Python 3.12 and 3.13.

## Notes for Termux

- Pure-Python dependencies only (httpx, rich, prompt_toolkit, PyYAML) — no
  native builds, so `pipx install` works on-device.
- The REPL degrades gracefully: on a TTY it renders a live markdown panel;
  on piped stdin it reads lines (scriptable, and used by the test suite).
