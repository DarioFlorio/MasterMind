"""
MasterMind Harness — extended command modules.
Registers MasterMind Harness slash commands:
  - /prose           → workflow language engine
  - /memory          → memory-core search + dreaming
  - /approve         → exec approval management
  - /compact         → manual session compaction
  - /discover        → mDNS local network discovery
  - /flow            → task flow management
  - /canvas          → A2UI canvas render demo
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING


def register_harness_commands(commands_registry: dict, engine=None) -> None:
    """Register MasterMind Harness slash commands."""

    # ── /prose ────────────────────────────────────────────────────────────────

    async def cmd_prose(args: str, session=None, **kwargs) -> str:
        """Run a .prose workflow file or show prose help."""
        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0] if parts else "help"
        rest = parts[1] if len(parts) > 1 else ""

        if subcmd == "help" or not subcmd:
            return (
                "**Open-Prose Workflow Language**\n"
                "  /prose run <file.prose>   — execute a workflow file\n"
                "  /prose run <source>       — run inline prose source\n"
                "  /prose examples           — show example programs\n"
                "  /prose compile <file>     — validate syntax only\n\n"
                "Example:\n"
                "```prose\n"
                "agent researcher:\n"
                "  model: qwen3\n"
                "  prompt: \"You are a research assistant\"\n\n"
                "session: researcher\n"
                "  prompt: \"Research quantum computing trends\"\n"
                "```\n"
                "Save as `research.prose` then: `/prose run research.prose`"
            )

        if subcmd == "run":
            if not rest:
                return "Usage: /prose run <file.prose>"
            path = Path(rest)
            if path.exists():
                source = path.read_text()
            else:
                source = rest  # inline source

            from prose.parser import parse_prose
            from prose.vm import ProseVM

            try:
                program = parse_prose(source)
                agents_list = list(program.agents.keys())
                stmts_count = len(program.stmts)
                return (
                    f"✓ Parsed prose program:\n"
                    f"  • Agents: {agents_list or 'none'}\n"
                    f"  • Statements: {stmts_count}\n"
                    f"  • Blocks: {list(program.blocks.keys()) or 'none'}\n\n"
                    f"To execute, the VM needs an async agent_runner. "
                    f"Use `prose.vm.run_prose_file()` from Python code, or "
                    f"integrate with the MasterMind swarm engine."
                )
            except Exception as e:
                return f"Prose parse error: {e}"

        if subcmd == "compile":
            if not rest:
                return "Usage: /prose compile <file.prose>"
            path = Path(rest)
            if not path.exists():
                return f"File not found: {rest}"
            from prose.parser import parse_prose
            try:
                program = parse_prose(path.read_text())
                return (
                    f"✓ Syntax OK — {path.name}\n"
                    f"  {len(program.stmts)} statements, "
                    f"{len(program.agents)} agents, "
                    f"{len(program.blocks)} blocks"
                )
            except Exception as e:
                return f"✗ Syntax error: {e}"

        if subcmd == "examples":
            examples_dir = Path(__file__).parent / "prose" / "examples"
            if examples_dir.exists():
                files = sorted(examples_dir.glob("*.prose"))
                return "Available examples:\n" + "\n".join(f"  • {f.name}" for f in files[:20])
            return (
                "Example prose programs:\n\n"
                "**Hello World:**\n```prose\nsession \"Introduce yourself briefly\"\n```\n\n"
                "**Parallel research:**\n```prose\nparallel:\n"
                "  a = session \"Research topic A\"\n"
                "  b = session \"Research topic B\"\nsession \"Synthesize findings\"\n"
                "  context: { a, b }\n```"
            )

        return f"Unknown prose subcommand: {subcmd}. Try /prose help"

    commands_registry["prose"] = cmd_prose

    # ── /memory ───────────────────────────────────────────────────────────────

    async def cmd_memory(args: str, session=None, **kwargs) -> str:
        """Memory-core: search, ingest, status, dream."""
        from memory_core.manager import get_memory_manager
        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0] if parts else "help"
        rest = parts[1] if len(parts) > 1 else ""

        mgr = get_memory_manager()

        if subcmd == "help" or not subcmd:
            return (
                "**Memory Core** (hybrid FTS + vector search)\n"
                "  /memory search <query>       — search stored memories\n"
                "  /memory ingest <file|text>   — index a file or text\n"
                "  /memory status               — show index statistics\n"
                "  /memory dream                — consolidate frequent memories\n"
                "  /memory clear                — remove all memories"
            )

        if subcmd == "search":
            if not rest:
                return "Usage: /memory search <query>"
            results = mgr.search_hybrid(rest, limit=5)
            if not results:
                return f"No memories found for: {rest}"
            lines = [f"**Memory search:** `{rest}`\n"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. [{r.match_type}] score={r.score:.2f} "
                    f"— `{r.chunk.path}`\n   {r.snippet[:200]}"
                )
            return "\n".join(lines)

        if subcmd == "ingest":
            if not rest:
                return "Usage: /memory ingest <file-path or text>"
            path = Path(rest)
            if path.exists():
                ids = mgr.ingest_file(str(path))
                return f"✓ Indexed {len(ids)} chunks from `{path}`"
            else:
                cid = mgr.ingest_text(rest, label="note")
                return f"✓ Indexed note (id: {cid})"

        if subcmd == "status":
            st = mgr.status()
            return (
                f"**Memory Index Status**\n"
                f"  Total chunks: {st['total_chunks']}\n"
                f"  Embedded: {st['embedded_chunks']}\n"
                f"  Sources: {st['sources']}\n"
                f"  Vector search: {'enabled' if st['vector_enabled'] else 'disabled'}\n"
                f"  DB: `{st['db_path']}`"
            )

        if subcmd == "dream":
            if engine is None:
                return "Dream requires an active engine. Pass engine= when registering commands."
            def simple_llm(prompt: str) -> str:
                # Use the main engine's completion — synchronous wrapper
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(engine.complete_simple(prompt))
                    return result
                finally:
                    loop.close()
            new_ids = mgr.dream(simple_llm, min_recall_count=2, limit=5)
            return (
                f"✓ Dreaming complete — consolidated {len(new_ids)} memory fragments"
                if new_ids else "No memories met the dreaming threshold yet."
            )

        if subcmd == "clear":
            confirm = rest.strip().lower()
            if confirm not in ("yes", "confirm"):
                return "Type `/memory clear yes` to confirm deleting all memories."
            # Simple clear via dropping and recreating
            db_path = mgr._db_path
            mgr.close()
            db_path.unlink(missing_ok=True)
            from memory_core.manager import MemoryManager, _GLOBAL_MANAGER
            import memory_core.manager as mc
            mc._GLOBAL_MANAGER = None
            return "✓ Memory index cleared."

        return f"Unknown memory subcommand: {subcmd}. Try /memory help"

    commands_registry["memory"] = cmd_memory

    # ── /approve ──────────────────────────────────────────────────────────────

    async def cmd_approve(args: str, session=None, **kwargs) -> str:
        """Exec approval management."""
        from exec_approvals.manager import get_approval_manager, ApprovalDecision
        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0] if parts else "help"
        rest = parts[1] if len(parts) > 1 else ""

        mgr = get_approval_manager()

        if subcmd == "help" or not subcmd:
            return (
                "**Exec Approvals**\n"
                "  /approve list              — show allowlist patterns\n"
                "  /approve add <pattern>     — add a pattern to allowlist\n"
                "  /approve log               — show recent approval decisions\n"
                "  /approve status            — show current policy\n"
                "  /approve check <cmd>       — check if command is pre-approved"
            )

        if subcmd == "list":
            policy = mgr.get_policy("main")
            default = mgr._default_policy
            all_allows = list(set(policy.allowlist + default.allowlist))
            return (
                f"**Allowlist ({len(all_allows)} patterns):**\n"
                + "\n".join(f"  • `{p}`" for p in sorted(all_allows)[:30])
            )

        if subcmd == "add":
            if not rest:
                return "Usage: /approve add <glob-pattern>"
            mgr.add_to_allowlist(rest.strip())
            return f"✓ Added `{rest.strip()}` to exec allowlist."

        if subcmd == "log":
            entries = mgr.get_audit_log(limit=10)
            if not entries:
                return "No exec approval events logged yet."
            lines = ["**Recent exec approvals:**"]
            for e in entries:
                icon = "✓" if e["decision"] == "allow" else "✗"
                lines.append(f"  {icon} [{e['agent']}] {e['command'][:60]} — {e['reason']}")
            return "\n".join(lines)

        if subcmd == "check":
            if not rest:
                return "Usage: /approve check <command>"
            approved = mgr.is_approved(rest)
            return (
                f"✓ Pre-approved: `{rest}`" if approved
                else f"⚠ Not pre-approved: `{rest}` (will prompt user)"
            )

        if subcmd == "status":
            p = mgr._default_policy
            return (
                f"**Exec Approval Policy**\n"
                f"  Ask mode: {p.ask.value}\n"
                f"  Fallback: {p.ask_fallback.value}\n"
                f"  Auto-allow skills: {p.auto_allow_skills}\n"
                f"  Allowlist patterns: {len(p.allowlist)}\n"
                f"  Blocklist patterns: {len(p.blocklist)}"
            )

        return f"Unknown approve subcommand: {subcmd}"

    commands_registry["approve"] = cmd_approve

    # ── /compact ──────────────────────────────────────────────────────────────

    async def cmd_compact(args: str, session=None, **kwargs) -> str:
        """Manually trigger session compaction."""
        if session is None:
            return "No active session to compact."

        from compaction.compactor import SessionCompactor, CompactionConfig

        instruction = args.strip()
        messages = getattr(session, "_messages", [])
        if not messages:
            return "No messages in session to compact."

        if engine is None:
            return "Compaction requires an active engine."

        def llm_fn(prompt: str, model=None):
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(engine.complete_simple(prompt))
            finally:
                loop.close()

        config = CompactionConfig(notify_user=True, min_tail_turns=4)
        compactor = SessionCompactor(llm_fn=llm_fn, config=config)
        result = compactor.compact(messages, context_limit=32000, instruction=instruction)

        # Update session messages
        if hasattr(session, "_messages"):
            session._messages = result.compacted_messages

        return (
            f"✓ Session compacted\n"
            f"  Before: ~{result.tokens_before:,} tokens\n"
            f"  After:  ~{result.tokens_after:,} tokens\n"
            f"  Reduction: {result.reduction_pct:.0f}%\n\n"
            f"**Summary:**\n{result.summary[:500]}"
        )

    commands_registry["compact"] = cmd_compact

    # ── /discover ─────────────────────────────────────────────────────────────

    async def cmd_discover(args: str, **kwargs) -> str:
        """Discover MasterMind instances on the local network."""
        from discovery.mdns import scan_local_network, _ZEROCONF_AVAILABLE
        if not _ZEROCONF_AVAILABLE:
            return (
                "mDNS discovery requires the `zeroconf` library.\n"
                "Install with: `pip install zeroconf`"
            )
        parts = args.strip().split()
        timeout = float(parts[0]) if parts and parts[0].replace('.', '').isdigit() else 5.0
        print(f"[discover] Scanning local network for {timeout}s...")
        gateways = await asyncio.get_event_loop().run_in_executor(
            None, lambda: scan_local_network(timeout=timeout)
        )
        if not gateways:
            return f"No MasterMind gateways found on local network (scanned {timeout}s)."
        lines = [f"**Found {len(gateways)} gateway(s):**"]
        for gw in gateways:
            lines.append(f"  • **{gw.name}** — {gw.url}")
            for k, v in gw.properties.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    commands_registry["discover"] = cmd_discover

    # ── /flow ─────────────────────────────────────────────────────────────────

    async def cmd_flow(args: str, **kwargs) -> str:
        """Task flow management."""
        from taskflow.flow import FlowRegistry
        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0] if parts else "help"
        rest = parts[1] if len(parts) > 1 else ""

        reg = FlowRegistry()

        if subcmd == "help" or not subcmd:
            return (
                "**Task Flows**\n"
                "  /flow list              — list all flows\n"
                "  /flow show <id>         — show flow details\n"
                "  /flow cancel <id>       — cancel a running flow\n"
                "  /flow delete <id>       — delete a flow record"
            )

        if subcmd == "list":
            flows = reg.list_flows()
            if not flows:
                return "No task flows recorded."
            lines = [f"**Task Flows ({len(flows)}):**"]
            for f in flows[:20]:
                status_icon = {"succeeded": "✓", "failed": "✗",
                               "running": "⟳", "cancelled": "⊘",
                               "pending": "○"}.get(f["status"], "?")
                lines.append(
                    f"  {status_icon} **{f['name']}** [{f['status']}] "
                    f"— {f['steps']} steps — `{f['flow_id']}`"
                )
            return "\n".join(lines)

        if subcmd == "show":
            if not rest:
                return "Usage: /flow show <flow-id>"
            flow = reg.get_flow(rest)
            if not flow:
                return f"Flow not found: {rest}"
            lines = [f"**Flow: {flow['name']}** ({flow['status']})\n"]
            for step in flow.get("steps", []):
                icon = {"succeeded": "✓", "failed": "✗", "running": "⟳",
                        "pending": "○", "skipped": "-"}.get(step["status"], "?")
                lines.append(f"  {icon} {step['name']}: {step['status']}")
                if step.get("error"):
                    lines.append(f"    Error: {step['error'][:100]}")
            return "\n".join(lines)

        if subcmd == "cancel":
            if not rest:
                return "Usage: /flow cancel <flow-id>"
            ok = reg.cancel_flow(rest)
            return f"✓ Cancelled flow: {rest}" if ok else f"Flow not found: {rest}"

        if subcmd == "delete":
            if not rest:
                return "Usage: /flow delete <flow-id>"
            ok = reg.delete_flow(rest)
            return f"✓ Deleted flow: {rest}" if ok else f"Flow not found: {rest}"

        return f"Unknown flow subcommand: {subcmd}"

    commands_registry["flow"] = cmd_flow

    # ── /canvas ───────────────────────────────────────────────────────────────

    async def cmd_canvas(args: str, **kwargs) -> str:
        """A2UI canvas demo — renders JSON widget protocol."""
        from canvas.renderer import Canvas

        canvas = Canvas(buffer=True)
        canvas.heading("Canvas Demo (A2UI Protocol)")
        canvas.text("This renders structured widgets via JSON protocol.")
        canvas.divider()
        canvas.text("**Supported widgets:**", style="subheading")
        canvas.list_widget(["text", "button", "card", "image", "slider",
                            "checkbox", "multiple-choice", "tabs", "modal"])
        canvas.divider()
        canvas.button("Example Button", on_click="handle_click", variant="primary")
        canvas.text("^^ buttons emit click events back to the agent ^^", style="caption")

        lines = canvas.flush()
        return (
            f"**A2UI JSONL output** ({len(lines)} actions):\n"
            "```json\n" + "\n".join(lines[:10]) + "\n```\n\n"
            "In a full UI integration, these actions are streamed to a "
            "web/mobile renderer that displays the widgets."
        )

    commands_registry["canvas"] = cmd_canvas
