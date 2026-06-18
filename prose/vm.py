"""
Open-Prose VM executor.
MasterMind workflow VM executor.

Executes .prose ASTs by spawning MasterMind subagents, managing state in
.prose/runs/{run_id}/ directories, and tracking bindings across the run.
"""
from __future__ import annotations
import asyncio
import datetime
import json
import os
import random
import string
import textwrap
from pathlib import Path
from typing import Any, Callable, Optional

from .parser import (
    ProseProgram, SessionStmt, ResumeStmt, LetBinding, ParallelBlock,
    RepeatBlock, ForEachBlock, LoopBlock, PipelineOp, BlockDef, DoBlock,
    ChoiceBlock, IfBlock, TryCatch, ThrowStmt, InlineSequence,
    AgentDef, UseDecl, parse_prose, _interpolate, Stmt
)


def _run_id() -> str:
    now = datetime.datetime.now()
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return now.strftime("%Y%m%d-%H%M%S") + f"-{suffix}"


class ProseError(Exception):
    """Raised by throw statements or fatal prose VM errors."""
    pass


class ProseVM:
    """
    Execute a ProseProgram by spawning MasterMind subagent sessions.

    Usage:
        vm = ProseVM(agent_runner=my_runner)
        result = await vm.run(program)
    """

    def __init__(
        self,
        agent_runner: Callable,  # async fn(prompt: str, model: str | None, context: str | None) -> str
        work_dir: str = ".",
        model: str | None = None,
        verbose: bool = True,
    ):
        self._runner = agent_runner
        self._work_dir = Path(work_dir)
        self._model = model
        self._verbose = verbose
        self._run_id = _run_id()
        self._run_dir = self._work_dir / ".prose" / "runs" / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Execution state
        self._bindings: dict[str, Any] = {}
        self._agents: dict[str, AgentDef] = {}
        self._blocks: dict[str, BlockDef] = {}
        self._outputs: dict[str, Any] = {}

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[prose] {msg}")

    def _save_state(self) -> None:
        state_path = self._run_dir / "state.md"
        lines = [f"# Prose Run {self._run_id}\n", f"## Bindings\n"]
        for k, v in self._bindings.items():
            lines.append(f"- **{k}**: {str(v)[:200]}\n")
        state_path.write_text("".join(lines))

    def _save_binding(self, name: str, value: Any) -> None:
        bindings_dir = self._run_dir / "bindings"
        bindings_dir.mkdir(exist_ok=True)
        (bindings_dir / f"{name}.md").write_text(str(value))
        self._bindings[name] = value

    def _load_agent_memory(self, agent_name: str) -> str:
        """Load persistent agent memory from .prose/agents/{name}/memory.md"""
        paths = [
            self._run_dir / "agents" / agent_name / "memory.md",
            self._work_dir / ".prose" / "agents" / agent_name / "memory.md",
            Path.home() / ".prose" / "agents" / agent_name / "memory.md",
        ]
        for p in paths:
            if p.exists():
                return p.read_text()
        return ""

    def _save_agent_memory(self, agent_name: str, content: str) -> None:
        agent_dir = self._run_dir / "agents" / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        mem_path = agent_dir / "memory.md"
        # Append new segment
        existing = mem_path.read_text() if mem_path.exists() else ""
        timestamp = datetime.datetime.now().isoformat()
        new_content = existing + f"\n\n---\n*{timestamp}*\n\n{content}"
        mem_path.write_text(new_content.strip())

    def _resolve_context(self, ctx: Any) -> str:
        """Resolve context prop (var name, list of var names, or dict) to a string."""
        if ctx is None:
            return ""
        if isinstance(ctx, str):
            return str(self._bindings.get(ctx, ""))
        if isinstance(ctx, list):
            parts = []
            for item in ctx:
                val = self._bindings.get(item, "")
                if val:
                    parts.append(f"[{item}]:\n{val}")
            return "\n\n".join(parts)
        if isinstance(ctx, dict):
            parts = []
            for k, v in ctx.items():
                val = self._bindings.get(v if isinstance(v, str) else k, "")
                if val:
                    parts.append(f"[{k}]:\n{val}")
            return "\n\n".join(parts)
        return str(ctx)

    async def _spawn_session(
        self,
        prompt: str,
        model: str | None = None,
        context: str | None = None,
        system: str | None = None,
        retry: int = 0,
    ) -> str:
        """Spawn a subagent session and return its output."""
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\n---\n\n{prompt}"
        if system:
            full_prompt = f"[System: {system}]\n\n{full_prompt}"

        m = model or self._model
        for attempt in range(max(1, retry + 1)):
            try:
                self._log(f"  spawning session: {prompt[:80]}...")
                result = await self._runner(prompt=full_prompt, model=m)
                return result
            except Exception as e:
                if attempt >= retry:
                    raise
                self._log(f"  retry {attempt + 1}/{retry}: {e}")
                await asyncio.sleep(2 ** attempt)

        return ""

    async def run(self, program: ProseProgram) -> dict[str, Any]:
        """Execute a ProseProgram. Returns dict of output bindings."""
        # Register agents and blocks
        self._agents.update(program.agents)
        self._blocks.update(program.blocks)

        # Process input declarations (already bound externally)
        for inp in program.inputs:
            if inp.name not in self._bindings:
                self._log(f"  input '{inp.name}' not bound ({inp.description})")

        # Execute statements
        for stmt in program.stmts:
            await self._exec_stmt(stmt)

        # Collect outputs
        for out in program.outputs:
            val = self._bindings.get(out.name, "")
            self._outputs[out.name] = val

        self._save_state()
        return dict(self._outputs) or dict(self._bindings)

    async def _exec_stmt(self, stmt: Stmt) -> Any:
        if isinstance(stmt, SessionStmt):
            return await self._exec_session(stmt)
        elif isinstance(stmt, ResumeStmt):
            return await self._exec_resume(stmt)
        elif isinstance(stmt, LetBinding):
            val = await self._exec_stmt(stmt.expr)
            self._save_binding(stmt.name, val)
            return val
        elif isinstance(stmt, ParallelBlock):
            return await self._exec_parallel(stmt)
        elif isinstance(stmt, RepeatBlock):
            return await self._exec_repeat(stmt)
        elif isinstance(stmt, ForEachBlock):
            return await self._exec_foreach(stmt)
        elif isinstance(stmt, LoopBlock):
            return await self._exec_loop(stmt)
        elif isinstance(stmt, PipelineOp):
            return await self._exec_pipeline(stmt)
        elif isinstance(stmt, BlockDef):
            self._blocks[stmt.name] = stmt
        elif isinstance(stmt, DoBlock):
            return await self._exec_do(stmt)
        elif isinstance(stmt, ChoiceBlock):
            return await self._exec_choice(stmt)
        elif isinstance(stmt, IfBlock):
            return await self._exec_if(stmt)
        elif isinstance(stmt, TryCatch):
            return await self._exec_try(stmt)
        elif isinstance(stmt, ThrowStmt):
            raise ProseError(stmt.message or "prose throw")
        elif isinstance(stmt, InlineSequence):
            result = None
            for s in stmt.steps:
                result = await self._exec_stmt(s)
            return result
        return None

    async def _exec_session(self, stmt: SessionStmt) -> str:
        prompt = ""
        model = stmt.props.get('model')
        context = self._resolve_context(stmt.props.get('context'))
        retry = stmt.props.get('retry', 0)
        system = None

        if stmt.agent:
            agent = self._agents.get(stmt.agent)
            if agent:
                model = model or agent.model
                system = agent.prompt
                prompt = stmt.props.get('prompt', '') or agent.prompt or ""
            else:
                prompt = f"Act as {stmt.agent}. " + stmt.props.get('prompt', '')
        elif stmt.prompt:
            prompt = _interpolate(stmt.prompt, self._bindings)

        result = await self._spawn_session(
            prompt=prompt,
            model=model,
            context=context or None,
            system=system,
            retry=retry,
        )

        if stmt.bind_name:
            self._save_binding(stmt.bind_name, result)

        return result

    async def _exec_resume(self, stmt: ResumeStmt) -> str:
        memory = self._load_agent_memory(stmt.agent)
        agent = self._agents.get(stmt.agent)
        system = agent.prompt if agent else None
        model = stmt.props.get('model') or (agent.model if agent else None)
        prompt = stmt.props.get('prompt', '')
        context = self._resolve_context(stmt.props.get('context'))
        if memory:
            context = f"Agent memory:\n{memory}\n\n{context}"
        result = await self._spawn_session(
            prompt=prompt, model=model, context=context, system=system
        )
        self._save_agent_memory(stmt.agent, result)
        return result

    async def _exec_parallel(self, stmt: ParallelBlock) -> list[Any]:
        tasks = [asyncio.create_task(self._exec_stmt(s)) for s in stmt.branches]
        if stmt.strategy == "first":
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            return [list(done)[0].result()]
        elif stmt.strategy == "any":
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]
        else:
            if stmt.on_fail == "continue":
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = await asyncio.gather(*tasks)
            return list(results)

    async def _exec_repeat(self, stmt: RepeatBlock) -> list[Any]:
        results = []
        for i in range(stmt.count):
            if stmt.index_var:
                self._bindings[stmt.index_var] = i
            for s in stmt.body:
                r = await self._exec_stmt(s)
                results.append(r)
        return results

    async def _exec_foreach(self, stmt: ForEachBlock) -> list[Any]:
        source = self._bindings.get(stmt.source, [])
        if isinstance(source, str):
            source = source.splitlines()

        async def run_one(item, idx):
            self._bindings[stmt.item_var] = item
            if stmt.index_var:
                self._bindings[stmt.index_var] = idx
            results = []
            for s in stmt.body:
                results.append(await self._exec_stmt(s))
            return results

        if stmt.parallel:
            tasks = [run_one(item, i) for i, item in enumerate(source)]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for i, item in enumerate(source):
                results.extend(await run_one(item, i))
            return results

    async def _exec_loop(self, stmt: LoopBlock) -> list[Any]:
        results = []
        iteration = 0
        max_iter = stmt.max_iterations or 100  # safety cap

        while iteration < max_iter:
            if stmt.condition:
                # Ask the model to evaluate the condition
                cond_prompt = (
                    f"Evaluate this condition and answer with exactly 'true' or 'false':\n"
                    f"{stmt.condition}\n\n"
                    f"Current iteration: {iteration}\n"
                    f"Bindings: {json.dumps({k: str(v)[:200] for k, v in self._bindings.items()})}"
                )
                verdict = (await self._spawn_session(cond_prompt)).strip().lower()
                cond_true = verdict.startswith('true')
                if stmt.condition_kind == "while" and not cond_true:
                    break
                if stmt.condition_kind == "until" and cond_true:
                    break

            if stmt.index_var:
                self._bindings[stmt.index_var] = iteration

            for s in stmt.body:
                results.append(await self._exec_stmt(s))

            iteration += 1

        return results

    async def _exec_pipeline(self, stmt: PipelineOp) -> Any:
        source = self._bindings.get(stmt.source, [])
        if isinstance(source, str):
            source = source.splitlines()

        result = source
        for op in stmt.ops:
            kind = op["kind"]
            body = op["body"]
            parallel = op.get("parallel", False)

            if kind in ("map", "pmap"):
                async def transform(item):
                    self._bindings["item"] = item
                    out = None
                    for s in body:
                        out = await self._exec_stmt(s)
                    return out
                if parallel:
                    result = await asyncio.gather(*[transform(item) for item in result])
                else:
                    result = [await transform(item) for item in result]

            elif kind == "filter":
                async def keep(item):
                    self._bindings["item"] = item
                    for s in body:
                        r = await self._exec_stmt(s)
                    cond = str(r).strip().lower() if r else ""
                    return cond.startswith("true") or cond.startswith("keep") or cond.startswith("yes")
                filtered = []
                for item in result:
                    if await keep(item):
                        filtered.append(item)
                result = filtered

            elif kind == "reduce":
                extra = op.get("extra", "acc, item")
                parts = [p.strip() for p in extra.split(',')]
                acc_var = parts[0] if len(parts) > 0 else "acc"
                item_var = parts[1] if len(parts) > 1 else "item"
                acc = None
                for item in result:
                    self._bindings[acc_var] = acc
                    self._bindings[item_var] = item
                    for s in body:
                        acc = await self._exec_stmt(s)
                result = acc

        return result

    async def _exec_do(self, stmt: DoBlock) -> Any:
        block = self._blocks.get(stmt.name)
        if not block:
            self._log(f"  block '{stmt.name}' not found")
            return None
        # Bind params
        for i, param in enumerate(block.params):
            if i < len(stmt.args):
                self._bindings[param] = self._bindings.get(stmt.args[i], stmt.args[i])
        result = None
        for s in block.body:
            result = await self._exec_stmt(s)
        return result

    async def _exec_choice(self, stmt: ChoiceBlock) -> Any:
        # Ask model which option to pick
        option_labels = [opt["label"] for opt in stmt.options]
        prompt = (
            f"You are making a choice based on: {stmt.criteria}\n\n"
            f"Available options:\n"
            + "\n".join(f"- {label}" for label in option_labels)
            + "\n\nRespond with ONLY the exact label of the best option."
        )
        context = json.dumps({k: str(v)[:300] for k, v in self._bindings.items()})
        chosen = (await self._spawn_session(prompt, context=context)).strip()

        # Find best matching option
        selected = stmt.options[0]  # default
        for opt in stmt.options:
            if opt["label"].lower() in chosen.lower() or chosen.lower() in opt["label"].lower():
                selected = opt
                break

        self._log(f"  choice selected: {selected['label']}")
        result = None
        for s in selected["body"]:
            result = await self._exec_stmt(s)
        return result

    async def _exec_if(self, stmt: IfBlock) -> Any:
        async def eval_cond(cond: str) -> bool:
            prompt = (
                f"Evaluate this condition and respond with exactly 'true' or 'false':\n{cond}\n\n"
                f"Context: {json.dumps({k: str(v)[:200] for k, v in self._bindings.items()})}"
            )
            verdict = (await self._spawn_session(prompt)).strip().lower()
            return verdict.startswith("true")

        if await eval_cond(stmt.condition):
            result = None
            for s in stmt.then_body:
                result = await self._exec_stmt(s)
            return result

        for branch in stmt.elif_branches:
            if await eval_cond(branch["condition"]):
                result = None
                for s in branch["body"]:
                    result = await self._exec_stmt(s)
                return result

        if stmt.else_body:
            result = None
            for s in stmt.else_body:
                result = await self._exec_stmt(s)
            return result

        return None

    async def _exec_try(self, stmt: TryCatch) -> Any:
        try:
            result = None
            for s in stmt.try_body:
                result = await self._exec_stmt(s)
            return result
        except Exception as e:
            if stmt.catch_var:
                self._bindings[stmt.catch_var] = str(e)
            result = None
            for s in stmt.catch_body:
                result = await self._exec_stmt(s)
            return result
        finally:
            if stmt.finally_body:
                for s in stmt.finally_body:
                    await self._exec_stmt(s)


async def run_prose_file(
    path: str,
    agent_runner: Callable,
    inputs: dict | None = None,
    model: str | None = None,
    work_dir: str = ".",
    verbose: bool = True,
) -> dict[str, Any]:
    """High-level helper: parse and execute a .prose file."""
    source = Path(path).read_text()
    program = parse_prose(source)
    vm = ProseVM(agent_runner=agent_runner, work_dir=work_dir, model=model, verbose=verbose)
    if inputs:
        vm._bindings.update(inputs)
    return await vm.run(program)


async def run_prose_source(
    source: str,
    agent_runner: Callable,
    inputs: dict | None = None,
    model: str | None = None,
    work_dir: str = ".",
    verbose: bool = True,
) -> dict[str, Any]:
    """High-level helper: parse and execute .prose source string."""
    program = parse_prose(source)
    vm = ProseVM(agent_runner=agent_runner, work_dir=work_dir, model=model, verbose=verbose)
    if inputs:
        vm._bindings.update(inputs)
    return await vm.run(program)
