EVE — Persistent Agent Harness
Local-first, fully autonomous AI agent with hybrid memory, 51 reasoning skills,
50+ built‑in tools, and WhatsApp integration. Runs on Windows, macOS, and Linux
– no cloud required.

✨ Features
51 reasoning skills – deep reason, causal, Bayesian, game theory, lateral
thinking, multi‑objective, timeline, recursive decomposition, and many more.

50+ built‑in tools – file I/O, bash/PowerShell, web search/fetch, Git,
LSP code intelligence, cron scheduling, task lifecycle, hybrid memory, wiki,
sandbox, structured output, MCP servers, team management, and a WhatsApp bridge.

Hybrid memory – full‑text search + vector embeddings + automatic
dreaming, curation, and idle consolidation.

WhatsApp via Baileys – no Twilio, no paid accounts. Scan a QR code once
and chat with EVE from your phone.

Plan mode – propose steps before execution, then run with approval gating.

Multi‑agent swarms – spawn sub‑agents in‑process or via tmux.

Prose workflows – .prose scripting engine for complex pipelines.

Hook system – register pre/post hooks on any event.

Plugin marketplace – install community plugins.

Session persistence & recovery – survives power cuts and resumes
incomplete tasks.

Vision support – pass [IMG:data:...] with a compatible mmproj file.

Streaming, reasoning display, token usage tracking, telemetry.

🚀 Quick Start
Clone the repository

text
git clone https://github.com/yourusername/eve.git
cd eve
Python 3.10+ and Node.js 18+ (for WhatsApp) required.

Install Python dependencies

text
pip install -r requirements.txt
(If requirements.txt is missing, run python autoinstall.py or simply
python main.py – it will auto‑install what it needs.)

Configure the model

Copy .env.example to .env.

Set MODEL_PATH to your GGUF file (e.g., C:/models/qwen2.5-7b-q4_k_m.gguf).

Optionally set MMPROJ_PATH for vision.

Run

text
python main.py
Type a request and press Enter. See /help for commands.

🔧 Configuration
All settings live in .env (or environment variables). Key ones:

Variable	Default	Description
MODEL_PATH	(empty)	Path to your GGUF model. Required.
DIRECT_MODE	1	0 = cloud, 1 = local llama‑cpp, 2 = hybrid.
LLAMA_SERVER_URL	http://127.0.0.1:8080	Server URL when not direct.
CONTEXT_SIZE	16384	Context window size in tokens.
MAX_TOKENS	4096	Max tokens per generation.
TEMPERATURE	1.0	Sampling temperature.
PERMISSION_MODE	auto	auto, ask, or deny.
N_THREADS	physical cores	CPU threads for generation.
N_GPU_LAYERS	-1 (auto)	GPU layers (0 = CPU only).
KV_CACHE_TYPE	8	KV cache quantization (8=q8_0).
USE_MLOCK	1	Lock memory to prevent swapping.
MMPROJ_PATH	(empty)	Path to mmproj for vision.
DRAFT_MODEL_PATH	(empty)	Path to a draft model for speculative decoding (optional).
See config/settings.py for the full list and advanced tuning options.

💬 Usage
Interactive chat – just type your request.

Slash commands – type /help to see all built‑in commands (/clear,
/compact, /status, /save, /load, /mode, /skills, /tasks,
/whatsapp, /voice, etc.).

WhatsApp – type /whatsapp to start the bridge; scan the QR code with
WhatsApp → Linked Devices. Then send messages to EVE directly from your phone.

Voice – type /voice to toggle microphone input (speech‑to‑text).

🧠 Reasoning Skills
EVE comes with 51 reasoning skills that can be invoked via the skill tool
or automatically chosen by the engine. They include:

Skill	Description
skill_router	Not sure which skill to use? Start here.
mode_switch	Auto‑detect reasoning mode, switch mid‑execution.
reason_chain	Chain multiple skills with residual context.
surgical_debug	Diagnose bugs/crashes/unexpected behaviour.
compound_fix	Full fix methodology: measure, test, verify, ship.
deep_reason	Deep multi‑step analysis for complex questions.
cot_reason	Step‑by‑step chain‑of‑thought for math and logic.
causal_reason	Root cause analysis, 5‑why, counterfactuals.
causal_forward_reason	Trace cascading consequences forward.
abduct	Best explanation / diagnosis by inference.
lateral_thinking	Creative, unexpected, non‑obvious solutions.
lateral_forward_thinking	Non‑obvious future paths and wild cards.
multi_objective	Trade‑offs, Pareto analysis, conflicting criteria.
multi_objective_future_optimization	Robust strategies across multiple futures.
epistemic_reason	Evaluate evidence quality, knowledge vs belief.
epistemic_future_reasoning	Predict how knowledge and beliefs will evolve.
bayes_reason	Bayesian inference, base rates, conditional probability.
probabilistic_forecasting	Calibrated probability estimates for future events.
constraint_solve	Logic puzzles, CSP, zebra riddles, knight/knave.
game_solve	Minimax, Nash equilibrium, optimal game strategy.
game_theoretic_forward_simulation	Predict moves and counter‑moves.
analogical_reason	Structural mapping between domains, analogies.
timeline_reason	Order events, detect conflicts, schedule dependencies.
timeline_projection_reason	Project milestones and future sequences.
recursive_decompose	Break big problems into sub‑problems recursively.
recursive_future_decomposition	Break complex forecasts into sub‑forecasts.
inductive_reason	Find patterns and rules from sequences or examples.
scenario_whatif_simulation	What‑if branches, best/worst case, stress test.
deep_multi_layer_prediction	Long‑arc societal and emergent predictions.
adaptive_reason	Adapt reasoning strategy mid‑problem.
web_search	Deep BFS/IDS web research (use for heavy searches).
debug	Structured debugging with hypothesis testing.
pm	Project management reasoning and planning.
self_healing	Automatically detect and correct agent errors.
code_remediation	Fix code issues using patterns and tests.
goal_anchor	Periodically re‑focus on the active goal (automatic).
wakefulness	Monitor agent output for loop/degradation (automatic).
temporal_cognition	Track time patterns and episode memory (automatic).
thinking_controller	Choose optimal thinking style for the task (automatic).
conversational_intent	Classify user intent (chat, command, task, stop).
Additional skills cover sentiment, emotional tone, priority, summary,
self‑critique, reflection, and context‑aware adaptation.

🛠️ Tools
EVE ships with over 50 built‑in tools, organised by category.

Shell & Execution
bash – Run Bash/shell commands (use command parameter).

powershell – Run PowerShell scripts/commands on Windows.

sandbox – Execute code in an isolated sandbox (bubblewrap on Linux).

File & Code
read_file – Read a file by path.

write_file – Create or overwrite a file (use path and content).

edit_file – Apply find‑and‑replace edits to a file.

glob – Find files matching a pattern.

grep – Search file contents with regex.

list_dir – List directory contents.

git – Run Git commands (status, diff, commit, etc.).

LSP – Language Server Protocol (go‑to‑definition, diagnostics, symbols).

notebook – Edit Jupyter notebooks programmatically.

test_runner – Run test suites and report results.

Web
web_search – Search the web (requires query).

web_fetch – Fetch and extract text from a URL.

Memory & Knowledge
memory_write – Store a memory with a key.

memory_read – Retrieve memories by query.

memory_search – Full‑text + vector search across session and past conversations.

journal – Append a dated journal entry.

scratchpad – Read/write a temporary scratchpad for working notes.

reflect – Trigger reflection/self‑critique.

wiki_write – Save a note to the knowledge base (inbox/notes/reference).

wiki_read – Retrieve a wiki note by title.

wiki_search – Full‑text search the wiki.

wiki_list – List notes, optionally filtered by folder or tag.

wiki_promote – Promote a note from inbox → notes → reference.

Task & Project Management
pm – Project management reasoning tool.

todo_write – Create/update a structured task list.

todo_read – Read current task list.

task_create – Create a persistent task (SQLite‑backed).

task_get – Get task details.

task_list – List tasks, optionally by status.

task_update – Update task status or fields.

task_stop – Cancel/stop a task.

cron_create – Schedule a recurring job.

cron_list – List scheduled jobs.

cron_delete – Remove a scheduled job.

Agent & Swarm
agent – Spawn a sub‑agent to handle a sub‑task.

team_create – Create a team of agents.

team_delete – Delete a team.

team_status – Get team status.

remote_trigger – Trigger an action via HTTP/webhook.

Interaction
ask_user – Ask the user a question (pause and wait for response).

send_message – Send a message to a connected endpoint.

receive_message – Receive pending messages.

brief – Generate a structured briefing of recent activity.

sleep – Wait for a specified number of seconds.

structured_output – Return output in a specific JSON schema.

task_output – Mark a task as complete with a structured result.

Plan & Worktree
enter_plan_mode – Switch to plan‑only mode (no execution).

exit_plan_mode – Exit plan‑only mode.

enter_worktree – Create/switch to a Git worktree.

exit_worktree – Leave a Git worktree.

worktree_list – List active worktrees.

Integrations & Meta
skill – Invoke any reasoning skill by name.

export – Export session to a file.

tool_search – Search for available tools by keyword.

mcp_invoke – Call a tool on a connected MCP server.

mcp_list_servers – List configured MCP servers.

whatsapp_send – Send a WhatsApp message (mirrors replies when active).

🔌 Connectors & Integrations
WhatsApp (Baileys)
No Twilio, no accounts. Uses the Baileys library to connect directly to WhatsApp Web.

On first activation (/whatsapp), a QR code appears. Scan it with WhatsApp → Linked Devices.

Session is saved – no need to scan again.

Inbound messages are queued and answered autonomously.

Replies are mirrored to the terminal when "WhatsApp mode" is active.

MCP (Model Context Protocol)
Register external MCP servers with /mcp add NAME URL.

EVE can invoke their tools via mcp_invoke and list them with /mcp.

Supports any MCP‑compatible tool provider.

Voice
Type /voice to toggle microphone input.

Uses speech‑to‑text (backend configurable: whisper, vosk, or system STT).

Responses can be spoken via text‑to‑speech.

Bridge Server
--bridge or /bridge start launches an HTTP/SSE server (default port 7777).

Enables remote access to the agent, multi‑client sessions, and integration with custom UIs.

Git Worktrees
Parallel branches can be checked out into separate worktrees.

EVE can work on multiple branches simultaneously with full tool access.

Cron & Task Lifecycle
Schedule recurring tasks with cron_create.

Persistent task tracking (SQLite) with status lifecycle: pending → in_progress → done/failed.

Plugin Marketplace
/plugin install NAME pulls plugins from the marketplace.

Plugins are self‑contained folders placed in ~/.mastermind/plugins/.

📁 Project Structure
text
eve/
├── main.py                 # Entry point, REPL, server launcher
├── agent/                  # Core agent loop, session, goal tracker
├── tools/                  # All tools (bash, file, web, memory, LSP, cron…)
├── skills/                 # Reasoning skills
├── services/               # Memory, consolidation, team sync
├── config/settings.py      # All configuration knobs
├── utils/model_client.py   # LLM backend (llama‑cpp or cloud)
├── wa_bridge.js            # WhatsApp Baileys bridge
├── node_modules/           # (auto‑installed WhatsApp deps)
└── .env                    # Your personal configuration
⚡ Performance
EVE ships with sensible defaults that keep inference fast without altering model quality:

KV cache q8_0 (double effective context)

Flash attention when available

Memory lock (mlock)

Greedy decoding by default (TEMPERATURE=0, TOP_K=1) – can be changed

Perf counters and internal logging disabled

Adjustable thread counts and batch sizes

All tweaks are in config/settings.py – nothing changes logits or sampling
unless you explicitly configure it.

🧠 Memory & Persistence
Hybrid memory – keyword + vector search across past conversations.

Auto‑dreaming – consolidates memories when idle.

Session resume – if EVE crashes, next launch detects the interruption and
injects the previous goal back into context, so it can pick up exactly where
it left off.

🔌 Extending EVE
Plugins – drop folders into ~/.mastermind/plugins/ or install from
marketplace with /plugin install.

MCP servers – register with /mcp add NAME URL.

Hooks – add Python functions in ~/.mastermind/hooks/.

Skills – write new reasoning modules in skills/.

Tools – subclass BaseTool and add to _build_tools() in main.py.

📦 Requirements
Python ≥ 3.10

llama‑cpp‑python (auto‑installed)

Node.js ≥ 18 (only for WhatsApp bridge)

sentence-transformers (optional, for vector memory)

A GGUF model file (e.g., from Hugging Face)

🤝 Contributing
Pull requests are welcome. Please open an issue first to discuss what you'd like to change.

📄 License
MIT License – see LICENSE file.
