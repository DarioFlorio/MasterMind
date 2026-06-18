"""
Skill: chain_of_memories
Chain of Memories – Monte Carlo tree search over the agent's own memory traces.

This skill helps the agent walk back through its own actions, steps, and past
reasoning in a structured, search-based way. Given a query about what happened,
why something occurred, or how a previous situation was handled, it builds a
tree of possible memory chains, evaluates their relevance, and returns the
most coherent causal/temporal path – similar to how a Monte Carlo method
explores a game tree, but applied to episodic memories.

Use this skill when:
- You need to answer "What did I do when …?" or "How did we get here?"
- You need to reconstruct a chain of events from incomplete or noisy memory
- You need to trace back from an outcome to the actions that caused it
- You want to verify the consistency of a sequence of past steps
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from skills.base_skill import BaseSkill

DESCRIPTION = (
    "Chain-of-Memories reasoning. Uses a Monte Carlo tree search over the agent's "
    "own memory store to reconstruct the most likely sequence of past actions, "
    "events, or thoughts relevant to a query. Balances exploration of unfamiliar "
    "memory branches with exploitation of high-relevance paths. Returns a synthesized "
    "chain and the final answer."
)


# ── Memory entry representation ────────────────────────────────────────────────

@dataclass
class MemoryNode:
    """A single memory item."""
    id: str
    content: str               # text of the memory
    timestamp: float = 0.0     # optional ordering cue
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Monte Carlo tree node for memory search ────────────────────────────────────

@dataclass
class MCTNode:
    memory: MemoryNode
    parent: Optional[MCTNode] = None
    children: List[MCTNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    depth: int = 0


# ── Relevance scorer ───────────────────────────────────────────────────────────

def _relevance(query: str, memory: MemoryNode) -> float:
    """
    Simple keyword-overlap relevance (0-1). In a real system this would be
    an embedding cosine similarity or a reranker model.
    """
    query_words = set(query.lower().split())
    mem_words = set(memory.content.lower().split())
    if not query_words or not mem_words:
        return 0.0
    overlap = len(query_words & mem_words)
    jaccard = overlap / len(query_words | mem_words)
    return jaccard


# ── Memory store mock (replace with actual agent memory retrieval) ─────────────

_MOCK_MEMORIES: List[MemoryNode] = [
    MemoryNode("m1", "User asked to implement login feature."),
    MemoryNode("m2", "I chose JWT for authentication."),
    MemoryNode("m3", "Wrote auth middleware in Node.js."),
    MemoryNode("m4", "Tested token expiry – it worked correctly."),
    MemoryNode("m5", "User reported login broken after 24h."),
    MemoryNode("m6", "Found bug: refresh token was not being stored."),
    MemoryNode("m7", "Fixed by adding refresh token rotation."),
    MemoryNode("m8", "Deployed fix – login works again."),
]


def _retrieve_memories(query: str = "", limit: int = 20) -> List[MemoryNode]:
    """
    Pull from real persistent stores via reflector_agent.real_retrieve_memories().
    Falls back to mock data if reflector is not initialised.
    """
    try:
        from reflector_agent import real_retrieve_memories
        raw = real_retrieve_memories(query=query, limit=limit)
        if raw:
            return [
                MemoryNode(
                    id=m["id"],
                    content=m["content"],
                    timestamp=m.get("timestamp", 0.0),
                )
                for m in raw
            ]
    except Exception:
        pass
    return _MOCK_MEMORIES[:limit]


# ── Monte Carlo Tree Search logic ──────────────────────────────────────────────

class MemoryMCTS:
    """
    Runs a Monte Carlo tree search over the memory graph.
    The graph is fully connected initially (any memory can follow any other),
    but the search prioritises transitions that make a coherent story.
    """

    def __init__(self, query: str, memories: List[MemoryNode], max_iter: int = 100):
        self.query = query
        self.memories = memories
        self.max_iter = max_iter
        self.root = MCTNode(
            memory=MemoryNode(id="root", content="START"), depth=0
        )
        self._build_initial_children()

    def _build_initial_children(self):
        """Attach all memories as possible first steps."""
        for mem in self.memories:
            child = MCTNode(memory=mem, parent=self.root, depth=1)
            self.root.children.append(child)

    def _simulate(self, node: MCTNode) -> float:
        """
        Rollout policy: randomly walk forward up to a max depth,
        accumulating relevance + coherence score.
        """
        current = node
        depth = node.depth
        max_depth = min(5, len(self.memories))  # avoid infinite loops
        score = _relevance(self.query, current.memory)
        while depth < max_depth:
            # pick a random memory not already in the path
            path_ids = self._get_path_ids(current)
            candidates = [m for m in self.memories if m.id not in path_ids]
            if not candidates:
                break
            next_mem = random.choice(candidates)
            child = MCTNode(memory=next_mem, parent=current, depth=depth + 1)
            current.children.append(child)
            current = child
            depth += 1
            # Add small bonus for temporal coherence: if timestamp increases
            if next_mem.timestamp > current.parent.memory.timestamp:
                score += 0.1
            score += _relevance(self.query, next_mem)
        return score / (depth + 1)  # normalised

    def _get_path_ids(self, node: MCTNode) -> set:
        ids = set()
        while node is not None:
            ids.add(node.memory.id)
            node = node.parent
        return ids

    def _ucb1(self, node: MCTNode, parent_visits: int) -> float:
        if node.visits == 0:
            return float("inf")
        exploitation = node.total_value / node.visits
        exploration = math.sqrt(2 * math.log(parent_visits) / node.visits)
        return exploitation + exploration

    def _select(self, node: MCTNode) -> MCTNode:
        """Select a child with highest UCB1 score."""
        if not node.children:
            return node
        best = max(node.children, key=lambda c: self._ucb1(c, node.visits))
        return self._select(best)

    def search(self, iterations: int = 50) -> MCTNode:
        for _ in range(min(iterations, self.max_iter)):
            # Selection
            leaf = self._select(self.root)
            # Expansion (if leaf is not terminal and has not been fully expanded)
            if leaf.visits > 0 and leaf.children:
                # expand one random child
                path_ids = self._get_path_ids(leaf)
                candidates = [m for m in self.memories if m.id not in path_ids]
                if candidates:
                    new_mem = random.choice(candidates)
                    new_child = MCTNode(
                        memory=new_mem, parent=leaf, depth=leaf.depth + 1
                    )
                    leaf.children.append(new_child)
                    leaf = new_child
            # Simulation
            reward = self._simulate(leaf)
            # Backpropagation
            self._backpropagate(leaf, reward)
        return self.root

    def _backpropagate(self, node: MCTNode, reward: float):
        while node is not None:
            node.visits += 1
            node.total_value += reward
            node = node.parent

    def best_chain(self) -> List[MemoryNode]:
        """Return the chain of memories with the highest average value from root to leaf."""
        best_leaf = self._best_leaf(self.root)
        chain = []
        node = best_leaf
        while node is not None and node.memory.id != "root":
            chain.append(node.memory)
            node = node.parent
        chain.reverse()
        return chain

    def _best_leaf(self, node: MCTNode) -> MCTNode:
        if not node.children:
            return node
        best = max(node.children, key=lambda c: (
            c.total_value / c.visits if c.visits > 0 else 0
        ))
        return self._best_leaf(best)


# ── Synthesis of the memory chain into a coherent answer ──────────────────────

def _synthesise_memory_chain(
    query: str, chain: List[MemoryNode], steps_taken: int, total_nodes: int
) -> str:
    if not chain:
        return "No relevant memory chain found."

    steps = []
    for i, mem in enumerate(chain):
        steps.append(f"{i+1}. [{mem.id}] {mem.content}")
    steps_str = "\n".join(steps)

    conclusion = (
        f"The most probable sequence of events/memories related to "
        f"\"{query[:100]}\" is:\n\n{steps_str}\n\n"
        f"This chain was identified after exploring {total_nodes} nodes "
        f"over {steps_taken} MCTS iterations."
    )
    return conclusion


# ── Skill class ────────────────────────────────────────────────────────────────

class ChainOfMemoriesSkill(BaseSkill):
    """
    Chain of Memories – Monte Carlo tree search over agent memory.

    Instead of a static memory retrieval, this skill actively builds a
    tree of possible memory chains and searches for the path that best
    answers the given query. It balances relevance (how well each memory
    matches the question) with coherence (temporal order, logical flow)
    using an upper-confidence bound formula.

    This is especially useful for reconstructing multi-step past events,
    causal chains from symptoms to root causes, or verifying what the agent
    tried before reaching a conclusion.
    """

    @property
    def name(self) -> str:
        return "chain_of_memories"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": (
                        "The question about past actions, events, or reasoning steps. "
                        "Examples: 'What did I do when the login broke last week?', "
                        "'Trace back why I chose a NoSQL database over SQL', "
                        "'Show me the steps I took to debug the memory leak.'"
                    ),
                },
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "timestamp": {"type": "number", "default": 0.0},
                        },
                        "required": ["id", "content"],
                    },
                    "description": (
                        "Optional: a list of memory entries to search over. "
                        "If omitted, the skill uses a built-in mock memory store "
                        "(for demonstration) or the agent's real memory system."
                    ),
                },
                "iterations": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 200,
                    "default": 50,
                    "description": "Number of MCTS iterations to run.",
                },
            },
            "required": ["problem"],
        }

    @property
    def cache_results(self) -> bool:
        return False

    def execute_impl(self, problem: str, **kwargs) -> str:
        # Gather memories from input or mock
        raw_memories = kwargs.get("memories", [])
        if raw_memories:
            memories = [
                MemoryNode(
                    id=m["id"],
                    content=m["content"],
                    timestamp=m.get("timestamp", 0.0),
                )
                for m in raw_memories
            ]
        else:
            memories = _retrieve_memories(query=problem)

        iterations = int(kwargs.get("iterations", 50))

        # Handle empty memory store
        if not memories:
            return (
                "No memories available to search. Please provide a list of "
                "memory entries or ensure the agent's memory store is populated."
            )

        # Run Monte Carlo tree search
        mcts = MemoryMCTS(problem, memories, max_iter=iterations)
        mcts.search(iterations=iterations)

        # Extract best chain
        best_chain = mcts.best_chain()

        # Compute stats for transparency
        total_nodes = _count_nodes(mcts.root)

        # Synthesise output
        synthesis = _synthesise_memory_chain(
            problem, best_chain, iterations, total_nodes
        )

        return synthesis


def _count_nodes(node: MCTNode) -> int:
    """Count all nodes in the MCTS tree recursively."""
    return 1 + sum(_count_nodes(c) for c in node.children)
