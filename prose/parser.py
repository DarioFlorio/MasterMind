"""
Open-Prose workflow language parser.
MasterMind workflow language engine.

Supports the full .prose grammar:
  session, agent, parallel, repeat, loop, for, block, choice, if/elif/else,
  let/const bindings, context, try/catch/finally, pipeline (map/filter/reduce),
  use (imports), input/output declarations, string interpolation.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union


# ── AST node types ────────────────────────────────────────────────────────────

@dataclass
class Comment:
    text: str

@dataclass
class InputDecl:
    name: str
    description: str

@dataclass
class OutputDecl:
    name: str
    expr: "Expr"

@dataclass
class UseDecl:
    path: str
    alias: str

@dataclass
class AgentDef:
    name: str
    model: Optional[str] = None
    prompt: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    permissions: dict = field(default_factory=dict)
    persist: bool = False

@dataclass
class SessionStmt:
    prompt: Optional[str] = None       # inline prompt string
    agent: Optional[str] = None        # agent reference
    bind_name: Optional[str] = None    # let/const binding name
    props: dict[str, Any] = field(default_factory=dict)  # model, context, retry, etc.

@dataclass
class ResumeStmt:
    agent: str
    props: dict[str, Any] = field(default_factory=dict)

@dataclass
class LetBinding:
    name: str
    expr: "Stmt"
    const: bool = False

@dataclass
class ParallelBlock:
    branches: list["Stmt"]
    strategy: str = "all"   # "all", "first", "any"
    on_fail: str = "fail"   # "fail", "continue"

@dataclass
class RepeatBlock:
    count: int
    index_var: Optional[str]
    body: list["Stmt"]

@dataclass
class ForEachBlock:
    item_var: str
    index_var: Optional[str]
    source: str
    body: list["Stmt"]
    parallel: bool = False

@dataclass
class LoopBlock:
    body: list["Stmt"]
    condition: Optional[str] = None   # AI-evaluated condition string
    condition_kind: str = "while"     # "while" or "until"
    max_iterations: Optional[int] = None
    index_var: Optional[str] = None

@dataclass
class PipelineOp:
    source: str
    ops: list[dict]  # [{"kind": "map"|"filter"|"reduce", "body": [...], "parallel": bool}]

@dataclass
class BlockDef:
    name: str
    params: list[str]
    body: list["Stmt"]

@dataclass
class DoBlock:
    name: str
    args: list[str]

@dataclass
class ChoiceBlock:
    criteria: str
    options: list[dict]   # [{"label": str, "body": [...]}]

@dataclass
class IfBlock:
    condition: str
    then_body: list["Stmt"]
    elif_branches: list[dict]   # [{"condition": str, "body": [...]}]
    else_body: Optional[list["Stmt"]] = None

@dataclass
class TryCatch:
    try_body: list["Stmt"]
    catch_body: list["Stmt"]
    catch_var: Optional[str] = None
    finally_body: Optional[list["Stmt"]] = None

@dataclass
class ThrowStmt:
    message: Optional[str] = None

@dataclass
class InlineSequence:
    steps: list["Stmt"]

Stmt = Union[
    SessionStmt, ResumeStmt, LetBinding, ParallelBlock, RepeatBlock,
    ForEachBlock, LoopBlock, PipelineOp, BlockDef, DoBlock, ChoiceBlock,
    IfBlock, TryCatch, ThrowStmt, InlineSequence, AgentDef, UseDecl,
    InputDecl, OutputDecl, Comment
]

Expr = Union[SessionStmt, str]

@dataclass
class ProseProgram:
    uses: list[UseDecl] = field(default_factory=list)
    inputs: list[InputDecl] = field(default_factory=list)
    outputs: list[OutputDecl] = field(default_factory=list)
    agents: dict[str, AgentDef] = field(default_factory=dict)
    blocks: dict[str, BlockDef] = field(default_factory=dict)
    stmts: list[Stmt] = field(default_factory=list)


# ── Tokeniser / line iterator ──────────────────────────────────────────────────

def _strip_comment(line: str) -> str:
    """Strip inline # comment, respecting strings."""
    in_str = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"' and not in_str:
            in_str = True
        elif c == '"' and in_str:
            in_str = False
        elif c == '#' and not in_str:
            return line[:i]
        i += 1
    return line

def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())

def _parse_string(s: str) -> str:
    """Unescape a quoted string value."""
    s = s.strip()
    if s.startswith('"""') and s.endswith('"""'):
        return s[3:-3]
    if s.startswith('"') and s.endswith('"'):
        inner = s[1:-1]
        return (inner
                .replace('\\"', '"')
                .replace('\\n', '\n')
                .replace('\\t', '\t')
                .replace('\\\\', '\\'))
    return s

def _interpolate(s: str, bindings: dict) -> str:
    """Replace {var} with binding values."""
    def repl(m):
        name = m.group(1)
        return str(bindings.get(name, m.group(0)))
    return re.sub(r'\{(\w+)\}', repl, s)


# ── Parser ─────────────────────────────────────────────────────────────────────

class ProseParser:
    """Parse a .prose source string into a ProseProgram AST."""

    def __init__(self, source: str):
        raw_lines = source.splitlines()
        # Strip comments, keep blank lines for indentation tracking
        self._lines: list[tuple[int, str]] = []  # (original_lineno, stripped_content)
        multi = False
        multi_buf: list[str] = []
        multi_start = 0
        for i, raw in enumerate(raw_lines):
            stripped = _strip_comment(raw).rstrip()
            self._lines.append((i + 1, stripped))
        self._pos = 0

    @property
    def _at_end(self) -> bool:
        return self._pos >= len(self._lines)

    def _peek(self) -> Optional[tuple[int, str]]:
        while self._pos < len(self._lines):
            lineno, content = self._lines[self._pos]
            if content.strip() == '':
                self._pos += 1
                continue
            return lineno, content
        return None

    def _next(self) -> Optional[tuple[int, str]]:
        item = self._peek()
        if item:
            self._pos += 1
        return item

    def _collect_block(self, parent_indent: int) -> list[tuple[int, str]]:
        """Collect lines that are indented deeper than parent_indent."""
        result = []
        while not self._at_end:
            lineno, content = self._lines[self._pos]
            if content.strip() == '':
                self._pos += 1
                continue
            if _indent(content) <= parent_indent:
                break
            result.append((lineno, content))
            self._pos += 1
        return result

    def _parse_props(self, lines: list[tuple[int, str]], base_indent: int) -> dict:
        """Parse key: value property lines into a dict."""
        props = {}
        for _, line in lines:
            if _indent(line) <= base_indent:
                continue
            m = re.match(r'\s*(\w+):\s*(.*)', line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if val.startswith('[') and val.endswith(']'):
                    props[key] = [v.strip().strip('"') for v in val[1:-1].split(',') if v.strip()]
                elif val.startswith('"'):
                    props[key] = _parse_string(val)
                elif val.isdigit():
                    props[key] = int(val)
                elif val.lower() in ('true', 'false'):
                    props[key] = val.lower() == 'true'
                else:
                    props[key] = val
        return props

    def parse(self) -> ProseProgram:
        prog = ProseProgram()
        while not self._at_end:
            item = self._peek()
            if not item:
                break
            lineno, line = item
            stripped = line.strip()
            if not stripped:
                self._pos += 1
                continue

            stmt = self._parse_stmt(prog)
            if stmt is not None:
                if isinstance(stmt, AgentDef):
                    prog.agents[stmt.name] = stmt
                elif isinstance(stmt, BlockDef):
                    prog.blocks[stmt.name] = stmt
                elif isinstance(stmt, UseDecl):
                    prog.uses.append(stmt)
                elif isinstance(stmt, InputDecl):
                    prog.inputs.append(stmt)
                elif isinstance(stmt, OutputDecl):
                    prog.outputs.append(stmt)
                elif not isinstance(stmt, Comment):
                    prog.stmts.append(stmt)

        return prog

    def _parse_stmt(self, prog: ProseProgram) -> Optional[Stmt]:
        item = self._peek()
        if not item:
            return None
        lineno, line = item
        stripped = line.strip()
        cur_indent = _indent(line)

        # Comment
        if stripped.startswith('#'):
            self._pos += 1
            return Comment(stripped[1:].strip())

        # use "path" as alias
        m = re.match(r'use\s+"([^"]+)"\s+as\s+(\w+)', stripped)
        if m:
            self._pos += 1
            return UseDecl(path=m.group(1), alias=m.group(2))

        # input name: "description"
        m = re.match(r'input\s+(\w+):\s*"([^"]*)"', stripped)
        if m:
            self._pos += 1
            return InputDecl(name=m.group(1), description=m.group(2))

        # output name = expr
        m = re.match(r'output\s+(\w+)\s*=\s*(.*)', stripped)
        if m:
            self._pos += 1
            return OutputDecl(name=m.group(1), expr=m.group(2).strip())

        # agent name:
        m = re.match(r'agent\s+(\w+)\s*:', stripped)
        if m:
            self._pos += 1
            name = m.group(1)
            prop_lines = self._collect_block(cur_indent)
            props = self._parse_props(prop_lines, cur_indent)
            return AgentDef(
                name=name,
                model=props.get('model'),
                prompt=props.get('prompt'),
                skills=props.get('skills', []),
                persist=props.get('persist', False),
            )

        # block name(params):
        m = re.match(r'block\s+(\w+)(\([^)]*\))?\s*:', stripped)
        if m:
            self._pos += 1
            name = m.group(1)
            params_str = m.group(2) or '()'
            params = [p.strip() for p in params_str[1:-1].split(',') if p.strip()]
            body_lines = self._collect_block(cur_indent)
            body = self._parse_stmts_from_lines(body_lines, prog)
            return BlockDef(name=name, params=params, body=body)

        # do name(args)
        m = re.match(r'do\s+(\w+)(\([^)]*\))?$', stripped)
        if m:
            self._pos += 1
            name = m.group(1)
            args_str = m.group(2) or '()'
            args = [a.strip() for a in args_str[1:-1].split(',') if a.strip()]
            return DoBlock(name=name, args=args)

        # let/const binding
        m = re.match(r'(let|const)\s+(\w+)\s*=\s*(.*)', stripped)
        if m:
            self._pos += 1
            const = m.group(1) == 'const'
            name = m.group(2)
            rest = m.group(3).strip()
            # rest should be a session statement
            expr = self._parse_inline_expr(rest, cur_indent, prog)
            return LetBinding(name=name, expr=expr, const=const)

        # parallel:
        if stripped.startswith('parallel'):
            return self._parse_parallel(stripped, cur_indent, prog)

        # repeat N:
        m = re.match(r'repeat\s+(\d+)(?:\s+as\s+(\w+))?\s*:', stripped)
        if m:
            self._pos += 1
            count = int(m.group(1))
            idx_var = m.group(2)
            body_lines = self._collect_block(cur_indent)
            body = self._parse_stmts_from_lines(body_lines, prog)
            return RepeatBlock(count=count, index_var=idx_var, body=body)

        # for item[, idx] in source:
        m = re.match(r'(?:parallel\s+)?for\s+(\w+)(?:,\s*(\w+))?\s+in\s+(\w+)\s*:', stripped)
        if m:
            self._pos += 1
            parallel = stripped.startswith('parallel')
            body_lines = self._collect_block(cur_indent)
            body = self._parse_stmts_from_lines(body_lines, prog)
            return ForEachBlock(
                item_var=m.group(1),
                index_var=m.group(2),
                source=m.group(3),
                body=body,
                parallel=parallel,
            )

        # loop [until|while **condition**] [as i] [max N]:
        if stripped.startswith('loop'):
            return self._parse_loop(stripped, cur_indent, prog)

        # try:
        if stripped == 'try:':
            return self._parse_try(cur_indent, prog)

        # throw
        if stripped.startswith('throw'):
            self._pos += 1
            m = re.match(r'throw(?:\s+"([^"]*)")?', stripped)
            return ThrowStmt(message=m.group(1) if m else None)

        # choice **criteria**:
        m = re.match(r'choice\s+(\*+)(.*?)(\*+)\s*:', stripped, re.DOTALL)
        if m:
            return self._parse_choice(stripped, cur_indent, prog)

        # if **condition**:
        m = re.match(r'if\s+\*\*(.+?)\*\*\s*:', stripped)
        if m:
            return self._parse_if(stripped, cur_indent, prog)

        # session
        if stripped.startswith('session'):
            return self._parse_session(stripped, cur_indent, prog)

        # resume
        m = re.match(r'resume\s*:\s*(\w+)', stripped)
        if m:
            self._pos += 1
            agent = m.group(1)
            prop_lines = self._collect_block(cur_indent)
            props = self._parse_props(prop_lines, cur_indent)
            return ResumeStmt(agent=agent, props=props)

        # pipeline: source | map:
        m = re.match(r'(\w+)\s*\|\s*(map|filter|reduce|pmap)\s*(?:\(([^)]+)\))?\s*:', stripped)
        if m:
            return self._parse_pipeline(stripped, cur_indent, prog)

        # skip unknown
        self._pos += 1
        return Comment(f"[unparsed] {stripped}")

    def _parse_inline_expr(self, rest: str, indent: int, prog: ProseProgram) -> Stmt:
        if rest.startswith('session'):
            return self._parse_session_inline(rest, indent, prog)
        return Comment(rest)

    def _parse_session(self, stripped: str, cur_indent: int, prog: ProseProgram) -> SessionStmt:
        self._pos += 1
        # session "prompt"
        m = re.match(r'session\s+"(.*?)"(?:\s*->\s*(.*))?$', stripped)
        if m:
            stmt = SessionStmt(prompt=_parse_string(f'"{m.group(1)}"'))
            prop_lines = self._collect_block(cur_indent)
            stmt.props = self._parse_props(prop_lines, cur_indent)
            # inline sequence ->
            if m.group(2):
                next_stmt = self._parse_session_inline(m.group(2).strip(), cur_indent, prog)
                return InlineSequence(steps=[stmt, next_stmt])
            return stmt

        # session: agentName  OR  session name: agentName
        m = re.match(r'session(?:\s+(\w+))?\s*:\s*(\w+)', stripped)
        if m:
            bind_name = m.group(1)
            agent = m.group(2)
            prop_lines = self._collect_block(cur_indent)
            props = self._parse_props(prop_lines, cur_indent)
            return SessionStmt(agent=agent, bind_name=bind_name, props=props)

        # session """multi"""
        m = re.match(r'session\s+(""".*?""")', stripped, re.DOTALL)
        if m:
            return SessionStmt(prompt=_parse_string(m.group(1)))

        return SessionStmt(prompt=stripped[7:].strip())

    def _parse_session_inline(self, text: str, indent: int, prog: ProseProgram) -> SessionStmt:
        m = re.match(r'session\s+"(.*?)"', text)
        if m:
            return SessionStmt(prompt=_parse_string(f'"{m.group(1)}"'))
        m = re.match(r'session\s*:\s*(\w+)', text)
        if m:
            return SessionStmt(agent=m.group(1))
        return SessionStmt(prompt=text)

    def _parse_parallel(self, stripped: str, cur_indent: int, prog: ProseProgram) -> ParallelBlock:
        self._pos += 1
        # parallel ("first"|"any"):  or parallel (on-fail: "continue"):
        strategy = "all"
        on_fail = "fail"
        m = re.search(r'\(\s*"(first|any|all)"\s*\)', stripped)
        if m:
            strategy = m.group(1)
        m = re.search(r'on-fail:\s*"(\w+)"', stripped)
        if m:
            on_fail = m.group(1)
        body_lines = self._collect_block(cur_indent)
        branches = self._parse_stmts_from_lines(body_lines, prog)
        return ParallelBlock(branches=branches, strategy=strategy, on_fail=on_fail)

    def _parse_loop(self, stripped: str, cur_indent: int, prog: ProseProgram) -> LoopBlock:
        self._pos += 1
        condition = None
        condition_kind = "while"
        max_iter = None
        index_var = None

        m = re.search(r'loop\s+until\s+\*\*(.+?)\*\*', stripped)
        if m:
            condition = m.group(1)
            condition_kind = "until"

        m = re.search(r'loop\s+while\s+\*\*(.+?)\*\*', stripped)
        if m:
            condition = m.group(1)
            condition_kind = "while"

        m = re.search(r'max\s+(\d+)', stripped)
        if m:
            max_iter = int(m.group(1))

        m = re.search(r'as\s+(\w+)', stripped)
        if m:
            index_var = m.group(1)

        body_lines = self._collect_block(cur_indent)
        body = self._parse_stmts_from_lines(body_lines, prog)
        return LoopBlock(
            body=body,
            condition=condition,
            condition_kind=condition_kind,
            max_iterations=max_iter,
            index_var=index_var,
        )

    def _parse_try(self, cur_indent: int, prog: ProseProgram) -> TryCatch:
        self._pos += 1
        try_lines = self._collect_block(cur_indent)
        try_body = self._parse_stmts_from_lines(try_lines, prog)

        catch_body = []
        catch_var = None
        finally_body = None

        item = self._peek()
        if item and item[1].strip().startswith('catch'):
            self._pos += 1
            m = re.match(r'catch(?:\s+as\s+(\w+))?\s*:', item[1].strip())
            if m:
                catch_var = m.group(1)
            catch_lines = self._collect_block(cur_indent)
            catch_body = self._parse_stmts_from_lines(catch_lines, prog)

        item = self._peek()
        if item and item[1].strip() == 'finally:':
            self._pos += 1
            finally_lines = self._collect_block(cur_indent)
            finally_body = self._parse_stmts_from_lines(finally_lines, prog)

        return TryCatch(
            try_body=try_body,
            catch_body=catch_body,
            catch_var=catch_var,
            finally_body=finally_body,
        )

    def _parse_choice(self, stripped: str, cur_indent: int, prog: ProseProgram) -> ChoiceBlock:
        self._pos += 1
        m = re.match(r'choice\s+\*+(.+?)\*+\s*:', stripped, re.DOTALL)
        criteria = m.group(1).strip() if m else stripped
        body_lines = self._collect_block(cur_indent)

        options = []
        i = 0
        while i < len(body_lines):
            _, line = body_lines[i]
            stripped_line = line.strip()
            mo = re.match(r'option\s+"([^"]+)"\s*:', stripped_line)
            if mo:
                opt_indent = _indent(line)
                label = mo.group(1)
                opt_lines = []
                i += 1
                while i < len(body_lines):
                    _, next_line = body_lines[i]
                    if next_line.strip() and _indent(next_line) <= opt_indent:
                        break
                    opt_lines.append(body_lines[i])
                    i += 1
                opt_body = self._parse_stmts_from_lines(opt_lines, prog)
                options.append({"label": label, "body": opt_body})
            else:
                i += 1

        return ChoiceBlock(criteria=criteria, options=options)

    def _parse_if(self, stripped: str, cur_indent: int, prog: ProseProgram) -> IfBlock:
        self._pos += 1
        m = re.match(r'if\s+\*\*(.+?)\*\*\s*:', stripped)
        condition = m.group(1) if m else ""
        body_lines = self._collect_block(cur_indent)
        then_body = self._parse_stmts_from_lines(body_lines, prog)

        elif_branches = []
        else_body = None

        while True:
            item = self._peek()
            if not item:
                break
            _, line = item
            ls = line.strip()
            if ls.startswith('elif'):
                self._pos += 1
                m = re.match(r'elif\s+\*\*(.+?)\*\*\s*:', ls)
                ec = m.group(1) if m else ""
                el = self._collect_block(cur_indent)
                eb = self._parse_stmts_from_lines(el, prog)
                elif_branches.append({"condition": ec, "body": eb})
            elif ls == 'else:':
                self._pos += 1
                el = self._collect_block(cur_indent)
                else_body = self._parse_stmts_from_lines(el, prog)
                break
            else:
                break

        return IfBlock(
            condition=condition,
            then_body=then_body,
            elif_branches=elif_branches,
            else_body=else_body,
        )

    def _parse_pipeline(self, stripped: str, cur_indent: int, prog: ProseProgram) -> PipelineOp:
        self._pos += 1
        m = re.match(r'(\w+)\s*\|(.*)', stripped)
        source = m.group(1) if m else ""
        ops_raw = m.group(2) if m else ""
        # Parse the first op
        ops = []
        om = re.match(r'\s*(p?map|filter|reduce)\s*(?:\(([^)]+)\))?\s*:', ops_raw)
        if om:
            kind = om.group(1)
            extra = om.group(2)
            body_lines = self._collect_block(cur_indent)
            body = self._parse_stmts_from_lines(body_lines, prog)
            ops.append({"kind": kind, "extra": extra, "body": body, "parallel": kind == "pmap"})
        return PipelineOp(source=source, ops=ops)

    def _parse_stmts_from_lines(self, lines: list[tuple[int, str]], prog: ProseProgram) -> list[Stmt]:
        """Re-parse a list of (lineno, content) pairs as statements."""
        sub_parser = ProseParser.__new__(ProseParser)
        sub_parser._lines = lines
        sub_parser._pos = 0
        stmts = []
        while not sub_parser._at_end:
            item = sub_parser._peek()
            if not item:
                break
            stmt = sub_parser._parse_stmt(prog)
            if stmt is not None and not isinstance(stmt, Comment):
                stmts.append(stmt)
        return stmts


def parse_prose(source: str) -> ProseProgram:
    """Parse a .prose source string and return a ProseProgram AST."""
    return ProseParser(source).parse()
