# manishcode

**Give a local LLM real tools, on your own machine.**

Local models like Llama or Qwen can reason but can't *do* anything on their own —
they can't read your files, search the web, or check your email. `manishcode`
bridges that gap: it connects a local **Ollama** model to any number of
[MCP](https://modelcontextprotocol.io) servers and lets the model call their
tools in a chat loop. Nothing leaves your machine — no cloud LLM, no API keys for
the model.

Point it at a folder and ask questions about your code. Connect the web-search
server and ask about today's news. Connect Gmail and have it triage your inbox.
All driven by a model running locally.

On a real terminal it opens a full-screen TUI (scrollable transcript, `/`-command
autocomplete, a status bar with model / connected servers / context usage, a y-N
prompt before anything that writes or sends). Piped or with `--plain` it's a
plain REPL.

## Setup

### Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — installs `manishcode` and fetches Python for you
  (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh` ·
  Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)
- **[Ollama](https://ollama.com)** installed and running, with a tool-capable model pulled:
  `ollama pull qwen2.5:7b-instruct-q4_0`
- **Node.js** installed — the MCP servers run via `npx`

That's it.

### Install

One command:

```
uv tool install "git+https://github.com/B-Manish/manishcode"
```

That puts a `manishcode` command on your PATH. Update later with
`uv tool upgrade manishcode`; remove with `uv tool uninstall manishcode`.

> Prefer not to install globally? Run it straight from the repo:
> ```
> uvx --from "git+https://github.com/B-Manish/manishcode" manishcode
> ```

### First run

```
mkdir my-agent && cd my-agent
manishcode               # start chatting
```

On the first run in a folder with no `config.json`, `manishcode` writes a starter
one (`filesystem` scoped to that folder + `duckduckgo` web search) so `/mcp
connect` has something to offer. Edit it to add servers (see
[Configuring servers](#configuring-servers)); `manishcode init --force`
regenerates it.

**Bare `manishcode` starts with no MCP servers connected** — a local model
drowns in 80+ tool schemas. Add only what a task needs:

```
manishcode                       # no tools; then  /mcp connect duckduckgo
manishcode --server duckduckgo --server gmail
manishcode --all                 # everything in the config (needs a big context)
```

## Usage

```
manishcode --config config.json --server filesystem
```

(`--config` defaults to `./config.json`, so inside a folder set up by
`manishcode init` you can just run `manishcode`.)

### Developing on this repo

```
git clone https://github.com/B-Manish/manishcode && cd manishcode
uv sync
uv run manishcode --config config.json      # or: uv run python -m mcp_client ...
uv run python smoke_test.py
```

### Common flags

| Flag | Meaning |
|------|---------|
| `--config FILE` | Path to the JSON config (default: `./config.json`). |
| `--server NAME` | Connect to this server from the config at startup. Repeatable. **Default: none** — bare `manishcode` starts with no tools; add them with `/mcp connect` or `--all`. |
| `-a`, `--all` | Connect to every server in the config at startup. |
| `--server-cmd "CMD"` | Launch a server by raw command, ignoring the config. Repeatable. |
| `--model NAME` | Ollama model to use (default `qwen3:8b`). |
| `--think` | Enable the model's thinking/reasoning mode. |
| `--debug` | Print the full JSON of every Ollama request/response and tool call/result. Implies `--plain`. |
| `--plain` | Skip the full-screen TUI and use the plain REPL. Auto-on when output isn't a terminal (piped, CI) or with `--debug`. |
| `--confirm-tools MODE` | Ask for `y/N` before a tool call runs. `risky` (default) prompts only for side-effecting tools (send mail, trash, write/move files, …); `all` prompts for every call; `none` never prompts. Piped/non-interactive input counts as "no". |
| `--context TOKENS` | Effective Ollama context window, used only to print a "context used" percentage after each turn (default: `$OLLAMA_CONTEXT_LENGTH` or 4096). Set it to whatever you started `ollama serve` with. |
| `--max-tool-result CHARS` | Trim any single tool result to this many chars before sending it to the model, so one big file (a 33 KB README, a directory tree, an API dump) can't overflow a small local context window. Default 8000; `0` = unlimited. `--debug` still logs the full result. |
| `--history FILE` | Load conversation from `FILE` at startup and save back after every turn. |
| `--list-servers` | Print the servers defined in the config and exit. |
| `--no-tools` | Pure chat: no servers **and** no tool-oriented system prompt. (Bare `manishcode` also starts no servers, but keeps the prompt so `/mcp connect` works well.) |

### In-session commands

Type `/` (or `/help`) at the prompt to print the list. Unknown `/commands` are
rejected rather than sent to the model.

- `/help`, `/`, `/?` — show the command list with descriptions
- `/new`, `/reset`, `/clear` — start a fresh conversation (keeps the system prompt)
- `/tools` — list connected tools and their server
- `/mcp` — list the servers in the config with their status (connected /
  available / needs-env); `/mcp connect NAME` starts one mid-session and merges
  its tools in, `/mcp disconnect NAME` stops it and drops its tools. Handy for
  keeping the tool count low: start bare, add a server when you need it, drop it
  when you're done.
- `/context` — show how many prompt tokens the conversation + tools currently use
- `/save FILE` — write the current conversation to `FILE`
- `/quit`, `/exit` — exit

After every turn the client prints a line like
`[context] 4,210 prompt tokens in use  ~34% of 12,288`. When it passes ~75% it
warns you to `/reset`; past ~90% Ollama silently drops the oldest messages (see
[Small context windows](#small-context-windows)).

### Examples

Two servers from the config, thinking mode on:

```
manishcode --config config.json --server filesystem --server memory --think
```

One-off server without a config:

```
manishcode --server-cmd "npx -y @modelcontextprotocol/server-filesystem C:\Users\me\Documents"
```

Start bare, add a server when you need it:

```
manishcode --model qwen3:8b
>>> ...just chat...
>>> /mcp connect duckduckgo     # now the model can search the web
>>> /mcp disconnect duckduckgo  # drop it again, free the context
```

Resume a saved conversation with debug logging:

```
manishcode --config config.json --history session.json --debug
```

## Configuring servers

`config.json` is a **menu**, not an autostart list — bare `manishcode` connects
nothing. `manishcode init` (or the first run) writes one pre-filled with the
servers below; `/mcp` lists them, and you connect what a task needs:

```
manishcode --server duckduckgo --server filesystem   # at startup
manishcode --all                                     # everything
>>> /mcp connect gmail                                # mid-session
```

| Server | Tools for | Setup |
|--------|-----------|-------|
| `filesystem` | read/write files under a directory | scoped to `.` — edit the path in `config.json` |
| `duckduckgo` | web search + fetch a result page | none |
| `fetch` | pull any URL as trimmed markdown | none |
| `git` | inspect/operate a local git repo | runs against the current directory |
| `brave-search` | higher-quality web search | free key from <https://brave.com/search/api/>, `setx BRAVE_API_KEY ...` |
| `github` | search repos/issues/code, read PRs | `setx GITHUB_TOKEN ghp_...` |
| `gmail` | read/search/draft/send mail | one-time OAuth, see below |
| `playwright` | drive a real browser | first connect downloads Chromium (~150 MB); ~24 tools — connect it alone |

Servers that need a key (`brave-search`, `github`) are **skipped with a `[warn]`**
if the variable isn't set, so they can sit in the config harmlessly until you add
one. Restart your terminal after `setx` so the variable is visible.

### The config format

```json
{
  "mcpServers": {
    "<name>": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\path"],
      "env": { "SOME_KEY": "${SOME_KEY}" }
    }
  }
}
```

- `command` / `args` — how to launch the server (`npx`, `uvx`, `python`, a path).
- `env` — extra environment variables, layered on the inherited environment.
  `${VAR}` is filled from your environment at launch; a server with an unset
  `${VAR}` is skipped with a `[warn]` instead of crashing the session.
- Same key name as Claude Desktop's config, so an existing `mcpServers` object
  usually works as-is.
- If two servers expose a tool with the same name, the second is renamed
  `<servername>__<toolname>`.

### Gmail — one-time OAuth

Gmail has no API-key mode, so a first-time setup is needed:

1. [Google Cloud console](https://console.cloud.google.com/) → new project →
   enable the **Gmail API** → configure the OAuth consent screen (External; add
   your own address as a test user).
2. Create an **OAuth client ID**, type **Desktop app**. Download the JSON and
   save it as `~/.gmail-mcp/gcp-oauth.keys.json`
   (`C:\Users\<you>\.gmail-mcp\gcp-oauth.keys.json`).
3. Run the auth flow once (opens a browser, caches a token next to the keys):
   ```
   npx -y @gongrzhe/server-gmail-autoauth-mcp auth
   ```

Then `/mcp connect gmail` (or `--server gmail`) works with no further prompts.
Sending, deleting and label changes go through the `--confirm-tools` y-N prompt.

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
manishcode --server-cmd "npx -y @modelcontextprotocol/server-filesystem ." --debug
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
  ui.py        PlainUI / RichUI / TuiUI output adapters
  tui.py       full-screen Textual app (default on a real terminal)
  debug.py     --debug logging
```
