# TermuxPilot (`tp`)

A **CLI-first autonomous AI agent that runs natively in Termux on Android**.
It chats with *any* OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter,
Together, DeepSeek, Ollama, llama.cpp server, LM Studio over LAN, …), streams
answers to your terminal, and — as versions land — controls the device, runs
shell commands through a sandboxed tool layer, manages files, and automates
workflows.

No hardcoded provider: all model access goes through one
**OpenAI-compatible provider layer** (`base_url` + `api_key` + `model`) with a
**fallback chain** (cloud → LAN → local).

---

## Roadmap

| Version | Scope | Status |
|---|---|---|
| **v0.1** | Project scaffold, config system + profiles, provider layer (SSE streaming, fallback chain), chat REPL (`tp -i`), one-shot + pipe support | ✅ **this release** |
| v0.2 | Shell tool with `safe`/`standard`/`yolo` permission gates, risk classifier, file ops with diff previews, session audit log | next |
| v0.3 | Termux:API bridge (clipboard, notifications, battery, …) | planned |
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
tp -i                                    # interactive chat (rich markdown, streaming)
cat error.log | tp "why is this failing?"   # stdin attached as context
tp --json "ping"                         # machine-readable one-shot
tp --list-profiles                       # show profiles (add --plain for scriptable output)
tp config init [--force]                 # write the sample config
tp config show [--config-path P]         # show the resolved config (keys masked)
tp --model gpt-4o --base-url https://api.openai.com/v1 "..."   # per-run overrides
```

Exit codes: `0` success · `1` provider failure (all chain links exhausted) ·
`2` config/usage error.

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
streaming request with plain JSON. Native `tools` (function calling) payloads
are passed through when you supply them — the v0.2 tool layer builds on this,
degrading to strict-JSON prompting on endpoints without tool support.

---

## Architecture

```
CLI / REPL (argparse + rich + prompt_toolkit)
  └─ Agent turn engine (repl.py: Conversation + streaming turn view)
      └─ Provider layer (provider/)
          ├─ OpenAICompatibleClient  httpx, SSE streaming, tool-call accumulation
          ├─ ProviderChain           ordered fallback with failover hook
          └─ errors                  retryable / non-retryable policy
      └─ config.py                   YAML + profiles + env expansion (v0.2+: allowlists)
```

v0.2 adds the **tool layer** (shell executor with `safe`/`standard`/`yolo`
gates, risk classifier, audit log, secret redaction) behind the same
`chain.chat(messages, tools=...)` interface — the REPL already renders
tool-call deltas.

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest                                   # 56 tests: config, SSE, client, chain, CLI e2e
python tests/mockserver.py --port 8100   # local OpenAI-compatible server
                                          #   --fail-next N  fail N requests (503)
                                          #   --drop-after N kill the stream mid-way
                                          #   --key K        require Bearer auth
```

Tests spin up real (threaded) mock servers on random ports, so the full
stack — YAML → profiles → chain → SSE → failover → REPL — is covered end to
end without any real provider.

## Notes for Termux

- Pure-Python dependencies only (httpx, rich, prompt_toolkit, PyYAML) — no
  native builds, so `pipx install` works on-device.
- The REPL degrades gracefully: on a TTY it renders a live markdown panel;
  on piped stdin it reads lines (scriptable, and used by the test suite).
