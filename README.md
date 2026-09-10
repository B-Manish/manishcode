# mcp-ollama-client

A local MCP client: connect to any number of [MCP](https://modelcontextprotocol.io)
servers and let a **local Ollama model** (default `qwen3:8b`) call their tools.
No cloud LLM involved.

## What it does

1. Launches one or more MCP servers as subprocesses (from a config file or `--server-cmd`).
2. Merges every server's tools into a single list and converts the schemas to
   Ollama's tool-calling format.
3. Runs an interactive chat loop: your message + tools go to Ollama, any tool
   calls it requests are routed to the owning server, results are fed back, and
   it repeats until the model gives a final text answer.

## Setup

### Requirements

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (used here as the package manager)
- **[Ollama](https://ollama.com)** running locally with a tool-capable model pulled:
  ```
  ollama pull qwen3:8b
  ```
- **Node.js 18+** — only needed for `npx`-based MCP servers (the filesystem,
  memory, everything, … servers are distributed as npm packages). Check with
  `node --version`.

### Install

```
uv sync
```

That installs the `mcp` and `ollama` Python packages into `.venv`.

## Usage

```
uv run mcp-ollama --config config.json
```

or without installing the entry point:

```
uv run python -m mcp_client --config config.json
```

### Common flags

| Flag | Meaning |
|------|---------|
| `--config FILE` | Path to the JSON config (default: `./config.json`). |
| `--server NAME` | Connect only to this server from the config. Repeatable. Default: all. |
| `--server-cmd "CMD"` | Launch a server by raw command, ignoring the config. Repeatable. |
| `--model NAME` | Ollama model to use (default `qwen3:8b`). |
| `--think` | Enable the model's thinking/reasoning mode. |
| `--debug` | Print the full JSON of every Ollama request/response and tool call/result. |
| `--confirm-tools MODE` | Ask for `y/N` before a tool call runs. `risky` (default) prompts only for side-effecting tools (send mail, trash, write/move files, …); `all` prompts for every call; `none` never prompts. Piped/non-interactive input counts as "no". |
| `--context TOKENS` | Effective Ollama context window, used only to print a "context used" percentage after each turn (default: `$OLLAMA_CONTEXT_LENGTH` or 4096). Set it to whatever you started `ollama serve` with. |
| `--max-tool-result CHARS` | Trim any single tool result to this many chars before sending it to the model, so one big file (a 33 KB README, a directory tree, an API dump) can't overflow a small local context window. Default 8000; `0` = unlimited. `--debug` still logs the full result. |
| `--history FILE` | Load conversation from `FILE` at startup and save back after every turn. |
| `--list-servers` | Print the servers defined in the config and exit. |

### In-session commands

- `/quit` — exit
- `/reset` — clear conversation history
- `/tools` — list connected tools and their server
- `/context` — show how many prompt tokens the conversation + tools currently use
- `/save FILE` — write the current conversation to `FILE`

After every turn the client prints a line like
`[context] 4,210 prompt tokens in use  ~34% of 12,288`. When it passes ~75% it
warns you to `/reset`; past ~90% Ollama silently drops the oldest messages (see
[Small context windows](#small-context-windows)).

### Examples

Two servers from the config, thinking mode on:

```
uv run mcp-ollama --config config.json --server filesystem --server memory --think
```

One-off server without a config:

```
uv run mcp-ollama --server-cmd "npx -y @modelcontextprotocol/server-filesystem C:\Users\me\Documents"
```

Resume a saved conversation with debug logging:

```
uv run mcp-ollama --config config.json --history session.json --debug
```

## Configuring servers

Create `config.json` (start from `config.example.json`). The `mcpServers` key
matches the convention used by Claude Desktop, so an existing config often works
as-is.

```json
{
  "mcpServers": {
    "<name>": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\path"],
      "env": { "OPTIONAL_VAR": "value" }
    }
  }
}
```

- `command` — executable to run (`npx`, `uvx`, `python`, an absolute path, …).
- `args` — list of arguments.
- `env` — optional extra environment variables. The parent environment (PATH etc.)
  is always inherited; these are layered on top. A value may reference a variable
  from the parent environment as `${VAR}` (e.g. `"BRAVE_API_KEY": "${BRAVE_API_KEY}"`).
  If any `${VAR}` a server needs is **not set**, that server is skipped at startup
  with a `[warn]` line instead of crashing the session — so a config can carry an
  API-key server that only activates when you've exported the key.

If two servers expose a tool with the same name, the second one's tools are
exposed to the model as `<servername>__<toolname>`.

### Example server configs

Copy any of these into the `mcpServers` object. All are official servers from
`@modelcontextprotocol`.

**Filesystem** — read/write files under the given directories:

```json
"filesystem": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem",
           "C:\\Users\\me\\Documents", "C:\\Users\\me\\projects"]
}
```

**Memory** — a persistent knowledge graph the model can write to and query:

```json
"memory": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-memory"],
  "env": { "MEMORY_FILE_PATH": "C:\\Users\\me\\.mcp-memory.json" }
}
```

**Everything** — reference server exercising every MCP feature; handy for testing
tool-calling:

```json
"everything": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-everything"]
}
```

**Playwright** — drive a real browser (navigate, click, type, screenshot, read
page content). First run downloads Chromium (~150 MB), so the server may take a
minute to report ready. Exposes ~20 tools, which adds ~3–4k prompt tokens per
call — best connected on its own rather than alongside other servers:

```json
"playwright": {
  "command": "npx",
  "args": ["-y", "@playwright/mcp@latest", "--headless"]
}
```

**Git** — inspect and operate on a local git repo (needs `uvx` from uv):

```json
"git": {
  "command": "uvx",
  "args": ["mcp-server-git", "--repository", "C:\\Users\\me\\projects\\myrepo"]
}
```

**Fetch** — fetch a URL and convert it to trimmed markdown for the model (needs
`uvx`). Its `max_length` / `start_index` args let the model page through a long
page instead of swallowing it whole — pair it with a GitHub file's `download_url`
to read a big README in bites:

```json
"fetch": {
  "command": "uvx",
  "args": ["mcp-server-fetch"]
}
```

### Web search

Local models have **no built-in web access** — they can't answer "what's the
weather in Hyderabad today" or "latest news about X" on their own. Add one of
these search servers and the model will call it for anything time-sensitive.

**DuckDuckGo** — no API key, works out of the box (needs `uvx` from uv). Exposes
`search` (DuckDuckGo results) and `fetch_content` (pull and clean a result page):

```json
"duckduckgo": {
  "command": "uvx",
  "args": ["duckduckgo-mcp-server"]
}
```

**Brave Search** — higher-quality results but needs a free API key from
<https://brave.com/search/api/>. Export the key as `BRAVE_API_KEY`; the config
reads it via `${BRAVE_API_KEY}`, and if the variable isn't set the server is
skipped with a warning (the rest of the session runs normally):

```json
"brave-search": {
  "command": "npx",
  "args": ["-y", "@brave/brave-search-mcp-server", "--transport", "stdio"],
  "env": { "BRAVE_API_KEY": "${BRAVE_API_KEY}" }
}
```

```powershell
# PowerShell — set for the current session, then run
$env:BRAVE_API_KEY = "your-key-here"
uv run mcp-ollama --config config.json --server brave-search
```

Quick check (DuckDuckGo, no key needed):

```
uv run mcp-ollama --config config.json --server duckduckgo
>>> what's the weather in Hyderabad today
```

You should see a `[tool] search(...)` line and an answer grounded in real,
current results.

### Gmail

Read, search, summarise, draft, and send mail from your Gmail account. Uses
[`@gongrzhe/server-gmail-autoauth-mcp`](https://www.npmjs.com/package/@gongrzhe/server-gmail-autoauth-mcp).

```json
"gmail": {
  "command": "npx",
  "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"]
}
```

**One-time OAuth setup** (Gmail has no API-key mode):

1. In the [Google Cloud console](https://console.cloud.google.com/): create a
   project, enable the **Gmail API**, and configure the OAuth consent screen
   (External; add your own address as a test user).
2. Create an **OAuth client ID** of type **Desktop app**. Download the JSON and
   save it as `gcp-oauth.keys.json` in `~/.gmail-mcp/` (i.e.
   `C:\Users\<you>\.gmail-mcp\gcp-oauth.keys.json`).
3. Run the auth flow once — a browser window opens, you approve, and a token is
   cached next to the keys file:
   ```powershell
   npx -y @gongrzhe/server-gmail-autoauth-mcp auth
   ```

After that the server starts with no further prompts. Then:

```powershell
uv run mcp-ollama --config config.json --server gmail --model qwen2.5:7b-instruct-q4_0
```
```
>>> summarise my 5 most recent unread emails
>>> draft a reply to the one from Alice saying I'll review it Monday
```

**Sending / deleting is gated.** This client auto-runs whatever tool the model
picks, so by default `--confirm-tools risky` makes it stop and ask before any
`send_*`, `trash_*`, `delete_*`, or `modify_*` call:

```
  [tool] send_email({'to': 'alice@example.com', 'subject': 'Re: proposal', ...})
  [confirm] run send_email({...}) ? [y/N]
```

Answer `n` and the model is told you declined and moves on. Use
`--confirm-tools all` to confirm every call, or `none` to disable (not
recommended with a Gmail server connected).

> **Context window:** full message bodies and long thread lists are large. Ask
> for a few messages at a time; `--max-tool-result` caps each result but the
> model can still lose track on "analyse my whole inbox".

## Error handling

- **Ollama not running / model missing** — checked at startup with a clear message.
- **A server fails to launch** — a warning is printed and the session continues
  with the servers that did start.
- **A tool call fails** (bad arguments, server error, unknown tool) — the error
  text is fed back to the model as the tool result, so it can recover or explain.
- **A server crashes mid-session** — its tools are marked `[DOWN]`; a warning
  prints after each turn. Other servers keep working.

## Small context windows

A local model's context is small (qwen3:8b defaults to ~4k tokens in Ollama). One
large tool result — a big README, a recursive directory tree, a GitHub API dump —
can exceed it, and Ollama then **silently drops tokens from the front**, so the
model loses the tool definitions and your question and answers poorly or not at
all. To stay under the limit:

- `--max-tool-result` caps each tool result (default 8000 chars ≈ 2k tokens).
- Ask for one small thing at a time — a specific file, a sub-path, fewer items —
  not "read the README and three source files".
- For a big document, use the **fetch** server on its raw URL with `max_length`,
  and let the model page through it.
- For a repo overview, `list_commits` + a directory listing + one short file is
  far lighter than a 33 KB README.
- Or raise Ollama's window: set `OLLAMA_CONTEXT_LENGTH=16384` (plus
  `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` to keep the extra
  KV-cache memory small) and restart Ollama.

## Testing

### Automated smoke test

```
uv run python smoke_test.py
```

Spins up real filesystem + memory MCP servers and your local Ollama, then checks
config parsing, multi-server routing, tool-name collisions, server isolation,
bad-server handling, a real tool-calling round trip, history save/load, graceful
tool-failure, and missing-model detection. Prints `[PASS]`/`[FAIL]` per check and
exits non-zero on any failure.

- `--model llama3.1:8b` to test a different model.
- `--skip-ollama` to run only the checks that don't need Ollama (fast).

### Manual check

```
uv run mcp-ollama --server-cmd "npx -y @modelcontextprotocol/server-filesystem ." --debug
```

Then try: `list the files in the current directory` — you should see the
`[tool] list_directory(...)` line and the model's summary. `--debug` shows the
full request/response JSON.

## Project layout

```
mcp_client/
  cli.py       argparse + REPL
  config.py    load config.json -> ServerSpec objects
  servers.py   ServerManager: one task per server, tool merging + routing
  chat.py      ChatSession: the Ollama <-> tools loop
  history.py   save/load conversation JSON
  debug.py     --debug logging
```
