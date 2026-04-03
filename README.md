# PyClaudeCode

Local agentic coding assistant — 
Runs fully offline using llama-cpp with any GGUF model (Qwen3 recommended).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env → set MODEL_PATH to your Qwen3 .gguf file
python main.py
```

## Usage

```
python main.py                    # Interactive REPL
python main.py -p "fix the bug"   # Headless one-shot
python main.py --auto             # Auto-approve all tools
python main.py --http             # Use llama-server instead of direct
python main.py -v                 # Verbose / debug
```

## Slash commands

| Command | Effect |
|---------|--------|
| `/clear` | Clear session history |
| `/compact` | Force context compression |
| `/status` | Show token usage |
| `/mode auto\|ask\|deny` | Change permission mode |
| `/save [file]` | Save session to JSON |
| `/memory` | Show persistent memory |
| `/help` | Show all commands |

## Tools (15)

`bash` `read_file` `write_file` `edit_file` `glob` `grep` `list_dir`
`web_search` `web_fetch` `agent` `todo_write` `todo_read`
`memory_write` `memory_read` `skill`

## Skills (on-demand, 8)

`deep_reason` `cot_reason` `lateral_thinking` `constraint_solve`
`bayes_reason` `causal_reason` `analogical_reason` `recursive_decompose`

Skills are loaded only when called — they never clog the context window.

## Architecture

```
main.py                   ← CLI, streaming REPL, server lifecycle
config/settings.py        ← All env-var config
agent/
  query_engine.py         ← ReAct loop (tool parse → execute → loop)
  session.py              ← Sliding/unlimited context window
  dispatcher.py           ← Intent routing (recall/skill/cot/normal)
  prompt_fmt.py           ← Qwen3/Llama3/DeepSeek prompt templates
tools/                    ← 15 tool implementations
utils/
  model_client.py         ← llama-cpp-python direct + HTTP modes
  permissions.py          ← ask/auto/deny permission gating
memory/manager.py         ← Persistent cross-session memory
skills/                   ← On-demand reasoning modules
heartbeat.py              ← Background autosave
```

## Context window

Set `UNLIMITED_CONTEXT=1` in `.env` for sliding-window summarisation.
Old messages are compressed by the model; recent messages stay verbatim.
Effective context is unlimited — only limited by disk.
