"""
agent/cfe_integration_patch.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run this file (python agent/cfe_integration_patch.py) next to main.py.
It applies surgical patches to agent/query_engine.py.

Files it reads/writes:
  agent/query_engine.py   ← patched in place

New files it requires (already written):
  agent/cfe_compressor.py
  agent/mcts_selector.py
  memory/three_tier.py
  memory/error_rag.py
  memory/trajectory_store.py

WHAT EACH PATCH DOES
━━━━━━━━━━━━━━━━━━━━
  P1  Import the 5 new modules
  P2  Instantiate CFE, ThreeTierMemory, ErrorRAG, MCTSSelector,
      TrajectoryStore in QueryEngine.__init__
  P3  After session.add_user() — ingest user text into CFE + 3-tier memory
  P4  After tool result — ingest result into CFE + 3-tier memory,
      store trajectory (success or failure)
  P5  Error path — store to ErrorRAG + augment error hint with past errors
  P6  _get_system_prompt() — append CFE features + memory snippets
  P7  Inject LLM callable into CFE + ThreeTierMemory after client ready
"""

import re
from pathlib import Path

SRC = Path(__file__).parent / "query_engine.py"
if not SRC.exists():
    raise FileNotFoundError(f"Not found: {SRC}")

code = SRC.read_text(encoding="utf-8")
original = code
applied, skipped = [], []


def patch(label, old, new, required=True):
    global code
    if old in code:
        code = code.replace(old, new, 1)
        applied.append(label)
        return True
    skipped.append(label)
    marker = "✗" if required else "~"
    print(f"  {marker} SKIP [{label}]")
    return False


# ════════════════════════════════════════════════════════════════
# P1 — Imports
# ════════════════════════════════════════════════════════════════

patch(
    "P1: import CFE bundle",
    "from agent.error_learner import ErrorLearner",
    "from agent.error_learner import ErrorLearner\n"
    "from agent.cfe_compressor import ContextFeatureEngineer\n"
    "from agent.mcts_selector  import MCTSSelector\n"
    "from memory.three_tier    import ThreeTierMemory\n"
    "from memory.error_rag     import ErrorRAG\n"
    "from memory.trajectory_store import TrajectoryStore",
)


# ════════════════════════════════════════════════════════════════
# P2 — __init__ instantiation
# ════════════════════════════════════════════════════════════════

patch(
    "P2: instantiate CFE bundle in __init__",
    "        self._error_learner = ErrorLearner(working_dir=working_dir)\n"
    "        self._goal_tracker = GoalTracker(working_dir=working_dir)",
    "        self._error_learner = ErrorLearner(working_dir=working_dir)\n"
    "        self._goal_tracker  = GoalTracker(working_dir=working_dir)\n"
    "\n"
    "        # ── CFE bundle ────────────────────────────────────────────────\n"
    "        _memdir = str(Path(working_dir) / 'memdir') if working_dir else 'memdir'\n"
    "        self._cfe          = ContextFeatureEngineer(token_budget=700)\n"
    "        self._three_tier   = ThreeTierMemory(\n"
    "            db_path=str(Path(_memdir) / 'three_tier'),\n"
    "        )\n"
    "        self._error_rag    = ErrorRAG(\n"
    "            db_path=str(Path(_memdir) / 'error_rag'),\n"
    "        )\n"
    "        self._trajectories = TrajectoryStore(\n"
    "            db_path=str(Path(_memdir) / 'trajectories'),\n"
    "        )\n"
    "        self._mcts = MCTSSelector(\n"
    "            error_rag=self._error_rag,\n"
    "            trajectory_store=self._trajectories,\n"
    "        )\n"
    "        self._turn_count = 0\n"
    "        # ─────────────────────────────────────────────────────────────",
)


# ════════════════════════════════════════════════════════════════
# P3 — Ingest user text into CFE + 3-tier on every user turn
# ════════════════════════════════════════════════════════════════

patch(
    "P3: CFE + 3-tier ingest on user turn",
    "        self.session.add_user(user_text)",
    "        self.session.add_user(user_text)\n"
    "        self._turn_count += 1\n"
    "        self._cfe.ingest('user', user_text)\n"
    "        self._three_tier.store(user_text, role='user', turn=self._turn_count)\n"
    "        self._three_tier.maybe_consolidate(self._turn_count)",
)


# ════════════════════════════════════════════════════════════════
# P4 — After tool result: ingest + trajectory store
#      Targets the session.add_tool_result(result_xml) at end of loop
# ════════════════════════════════════════════════════════════════

patch(
    "P4: ingest tool result into CFE + trajectory",
    "            self.session.add_tool_result(result_xml)",
    "            self.session.add_tool_result(result_xml)\n"
    "            # ── CFE: ingest tool result ───────────────────────────────\n"
    "            _r_text = result.output[:600] if hasattr(result, 'output') else ''\n"
    "            self._cfe.ingest('tool', _r_text)\n"
    "            self._three_tier.store(_r_text, role='tool', turn=self._turn_count)\n"
    "            # ── Trajectory: record success / failure ──────────────────\n"
    "            _traj_outcome = 'failure' if result.is_error else 'success'\n"
    "            self._trajectories.store(\n"
    "                task=self._goal_text or user_text[:120],\n"
    "                actions=[{'tool': t_name, 'args': t_inp, 'result': _r_text[:200]}],\n"
    "                outcome=_traj_outcome,\n"
    "                summary=f\"{t_name} {'succeeded' if not result.is_error else 'failed'}"
    ": {self._goal_text[:60] if self._goal_text else ''}\",\n"
    "            )\n"
    "            if result.is_error:\n"
    "                self._error_rag.store(\n"
    "                    task=self._goal_text or user_text[:120],\n"
    "                    action=f\"{t_name}: {str(t_inp)[:200]}\",\n"
    "                    error=result.output[:300],\n"
    "                )",
)


# ════════════════════════════════════════════════════════════════
# P5 — Error path: augment ErrorLearner hint with ErrorRAG recall
# ════════════════════════════════════════════════════════════════

patch(
    "P5: augment error hint with ErrorRAG past errors",
    "                        hint = self._error_learner.get_hint(\n"
    "                            t_name, t_inp, result.output, attempt=self._inner\n"
    "                        )\n"
    "                        # Also warn if this pattern failed in past sessions\n"
    "                        cross_warn = self._error_learner.cross_session_warnings(t_name, {})",
    "                        hint = self._error_learner.get_hint(\n"
    "                            t_name, t_inp, result.output, attempt=self._inner\n"
    "                        )\n"
    "                        # ChromaDB semantic recall of past errors\n"
    "                        _rag_past = self._error_rag.recall(\n"
    "                            task=self._goal_text or '',\n"
    "                            action=f'{t_name}: {str(t_inp)[:100]}',\n"
    "                        )\n"
    "                        _rag_block = self._error_rag.format_hints(_rag_past)\n"
    "                        if _rag_block:\n"
    "                            hint = _rag_block + '\\n' + hint\n"
    "                        # Also warn if this pattern failed in past sessions\n"
    "                        cross_warn = self._error_learner.cross_session_warnings(t_name, {})",
)


# ════════════════════════════════════════════════════════════════
# P6 — Inject CFE features + memory snippets into system prompt
# ════════════════════════════════════════════════════════════════

patch(
    "P6: inject CFE + memory into _get_system_prompt",
    "    def _get_system_prompt(self) -> str:",
    "    def _get_system_prompt(self) -> str:\n"
    "        # ── CFE injection ─────────────────────────────────────────────\n"
    "        _cfe_block = self._cfe.render(query=self._goal_text or '')\n"
    "        _mem_snips = self._three_tier.retrieve(\n"
    "            self._goal_text or '', k=4)\n"
    "        _traj_snips = self._trajectories.retrieve_similar(\n"
    "            self._goal_text or '', k=2)\n"
    "        _traj_block = self._trajectories.format_hint(_traj_snips)\n"
    "        _cfe_inject = ''\n"
    "        if _cfe_block:\n"
    "            _cfe_inject += '\\n\\n' + _cfe_block\n"
    "        if _mem_snips:\n"
    "            _cfe_inject += '\\n\\n## Retrieved Memory\\n'\n"
    "            _cfe_inject += '\\n'.join(f'- {s}' for s in _mem_snips)\n"
    "        if _traj_block:\n"
    "            _cfe_inject += '\\n\\n' + _traj_block\n"
    "        # Store for append below\n"
    "        self._cfe_inject_block = _cfe_inject\n"
    "        # ─────────────────────────────────────────────────────────────",
)

# Append CFE block to the assembled prompt inside _get_system_prompt
patch(
    "P6b: append _cfe_inject_block to final prompt",
    "        return self._base_sys_prompt",
    "        _final = self._base_sys_prompt\n"
    "        if hasattr(self, '_cfe_inject_block') and self._cfe_inject_block:\n"
    "            _final = _final + self._cfe_inject_block\n"
    "        return _final",
)


# ════════════════════════════════════════════════════════════════
# P7 — Inject LLM callable into CFE + ThreeTierMemory
#      (needed for LLM-powered feature extraction + consolidation)
#      Hook: right after the model client is ready
# ════════════════════════════════════════════════════════════════

patch(
    "P7: inject LLM callable after client ready",
    "        self._base_sys_prompt = _build_system_prompt(",
    "        # Wire up LLM callbacks now that client is initialised\n"
    "        def _cfe_llm(prompt, system='', temperature=0.05, max_tokens=256):\n"
    "            try:\n"
    "                return self.client.complete(\n"
    "                    messages=[{\"role\": \"user\", \"content\": prompt}],\n"
    "                    system=system, temperature=temperature,\n"
    "                    max_tokens=max_tokens,\n"
    "                )\n"
    "            except Exception:\n"
    "                return ''\n"
    "        self._cfe.set_llm(_cfe_llm)\n"
    "        self._three_tier.set_llm(_cfe_llm)\n"
    "        self._base_sys_prompt = _build_system_prompt(",
    required=False,
)


# ════════════════════════════════════════════════════════════════
# Write
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    if code != original:
        SRC.write_text(code, encoding="utf-8")
        print(f"✅  query_engine.py patched. Applied {len(applied)} patch(es):")
        for a in applied:
            print(f"    ✓ {a}")
        if skipped:
            print(f"\n  Skipped {len(skipped)}:")
            for s in skipped:
                print(f"    ~ {s}")
    else:
        print("⚠️  No changes written — all patterns already applied or not found.")

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RUNTIME BEHAVIOUR AFTER PATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  System prompt now ends with:

  ## Compressed Context (CFE)
  ## Active Goals
  - Build Tetris with 10 levels using pygame
  ## Key Entities
  - tetris.py, tetris_bot.py, agent_workspace
  ## Established Facts
  - pygame is available; no external APIs
  ## Code / Exact Values
  - CoT: Write game loop with 60ms interval at level 10

  ## Retrieved Memory
  - [episodic] Wrote stress_test.py fibonacci; ran ok; 1 1 2 3 5...
  - [semantic] EVE uses gemma-4; user prefers local-only tools

  PAST SOLUTIONS (adapt — do NOT copy blindly):
    [✓ SUCCESS] score=0.82  task: write python file with pygame
      action: tool=bash  args=python tetris.py

  On error:
  ⚠ PAST ERRORS on similar tasks (do NOT repeat these):
    • Error: FileNotFoundError: agent_workspace/tetris.py not found
      Fix:  create directory first with os.makedirs(..., exist_ok=True)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
