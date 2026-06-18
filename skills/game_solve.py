"""
Skill: game_solve
Game theory and adversarial search: minimax, Nash equilibria, optimal strategies.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.game_solve")

DESCRIPTION = (
    "Game theory and adversarial search: optimal play, minimax, Nash equilibria, "
    "Nim, combinatorial games. Use when asked 'who wins', 'optimal strategy', "
    "'game tree', or any two-player zero-sum game."
)


def _nim(problem: str) -> str:
    return f"""**Game Solve: Nim / Token-Taking Game**
Problem: {problem}

**Winning Theory (Sprague-Grundy / Nim-sum):**

For standard Nim with piles [p1, p2, ..., pk]:
  Nim-sum = p1 XOR p2 XOR ... XOR pk

  If Nim-sum = 0  → position is LOSING for the player to move (P-position)
  If Nim-sum ≠ 0  → position is WINNING for the player to move (N-position)

**Optimal Strategy:**
  On your turn, leave the opponent in a P-position (Nim-sum = 0).
  Always possible when Nim-sum ≠ 0: find a pile where removing stones
  makes all XORs cancel.

**Example (piles = [3, 5, 7]):**
  3 XOR 5 = 6,  6 XOR 7 = 1 → Nim-sum = 1 ≠ 0 → First player WINS.
  Optimal move: reduce pile of 3 to 2 → [2, 5, 7], 2 XOR 5 XOR 7 = 0.

**Common Variants:**
  - Misère Nim (last to take LOSES): same strategy except when only piles of
    size 1 remain; reverse choice then.
  - Single-pile (Subtraction game): win iff pile size is not divisible by (k+1)
    where k = max you can take.

**Apply to this problem:**
  List the piles/positions explicitly, compute Nim-sum, determine winner,
  and trace one move ahead to confirm the winning response.
"""


def _minimax_generic(problem: str) -> str:
    return f"""**Game Solve: Minimax Tree Search**
Problem: {problem}

**Algorithm:**

```
minimax(node, depth, is_maximiser):
    if depth == 0 or node is terminal:
        return evaluate(node)
    if is_maximiser:
        best = -∞
        for child in node.children:
            best = max(best, minimax(child, depth-1, False))
        return best
    else:  # minimiser
        best = +∞
        for child in node.children:
            best = min(best, minimax(child, depth-1, True))
        return best
```

**Alpha-Beta Pruning** (same result, far fewer nodes evaluated):
  Maintain α (best maximiser can guarantee) and β (best minimiser can guarantee).
  Prune branch when α ≥ β.
  Worst-case: O(b^d). With pruning: O(b^(d/2)) — effectively doubles search depth.

**Tic-Tac-Toe:** Complete game tree is tiny (~9! = 362880 nodes).
  Minimax finds perfect play; optimal play always draws.

**Evaluation Heuristic (for deep games like Chess):**
  Piece values + positional bonuses + mobility + king safety.
  Stop at horizon depth; evaluate heuristically.

**Key Insight for this problem:**
  Identify: terminal conditions (win/loss/draw), branching factor, depth.
  Apply minimax. If the game is small enough, enumerate all paths.
"""


def _prisoners_dilemma(problem: str) -> str:
    return f"""**Game Solve: Prisoner's Dilemma**
Problem: {problem}

**Payoff Matrix (standard):**
                 B Cooperates    B Defects
  A Cooperates:   (3, 3)          (0, 5)
  A Defects:      (5, 0)          (1, 1)

**Nash Equilibrium Analysis:**
  - Cooperate is dominated by Defect for BOTH players (regardless of what the
    other does, defecting gives a higher payoff).
  - Nash Equilibrium: (Defect, Defect) → payoff (1,1) — Pareto-inferior!
  - Pareto Optimal: (Cooperate, Cooperate) → (3,3) — but not Nash stable.

**Iterated Prisoner's Dilemma:**
  Played repeatedly, cooperation can emerge. Best known strategy: Tit-for-Tat.
    Round 1: Cooperate.
    Round N: Mirror opponent's last move.
  Properties: nice, retaliatory, forgiving, clear.

**Application to this problem:**
  1. Identify each player's dominant strategy.
  2. Find Nash equilibria (where no player benefits from unilateral deviation).
  3. Compare to socially optimal outcome.
  4. Identify whether binding agreements or repetition could achieve cooperation.
"""


def _auction(problem: str) -> str:
    return f"""**Game Solve: Auction / Bidding Strategy**
Problem: {problem}

**Auction Types:**
  English (ascending): bid until only one remains → winner pays own final bid.
  Dutch (descending): clock falls until someone accepts → winner pays clock price.
  First-price sealed: highest bid wins, pays own bid.
  Second-price (Vickrey): highest bid wins, pays SECOND-highest bid.

**Dominant Strategy in Vickrey Auction:**
  Bid your true valuation v. This is weakly dominant — no strategy does better
  regardless of others' bids.
  Proof: If you win (your bid > all others), you pay the second price p.
         Overbidding can only make you win at a price above v (bad).
         Underbidding can only make you lose when p < v (missed profit).

**First-Price Sealed Bid (risk-neutral, N bidders, uniform valuations [0,1]):**
  Optimal bid = v × (N−1)/N
  (Shading factor: bid below true value to extract surplus.)

**Revenue Equivalence Theorem:**
  Under standard conditions, all four auction formats yield the same
  expected revenue to the seller.

**Apply to this problem:**
  Identify auction type → apply the corresponding equilibrium strategy.
"""


def _nash(problem: str) -> str:
    return f"""**Game Solve: Nash Equilibrium**
Problem: {problem}

**Definition:**
  A strategy profile (s1*, s2*, ..., sn*) is a Nash Equilibrium iff:
  For every player i: ui(si*, s−i*) ≥ ui(si, s−i*) for all si.
  (No player can improve their payoff by unilaterally deviating.)

**Finding Nash Equilibria — Step by Step:**

Step 1: Write out the full payoff matrix.
Step 2: For each player, find best responses to each opponent strategy.
  Mark: underline the best response payoff in each column (player 1),
        then in each row (player 2).
Step 3: A cell where BOTH payoffs are underlined = Nash Equilibrium.

**Pure vs Mixed Strategy:**
  Pure NE: each player plays a single strategy with certainty.
  Mixed NE: players randomise. Every finite game has at least one NE (Nash 1950).
  Mixed NE computation: find probabilities p such that opponent is indifferent.

**Existence Guarantee:**
  Nash's theorem: every finite game with n players has at least one NE
  (possibly in mixed strategies).

**Apply to this problem:**
  Build the matrix, find best responses, identify equilibria.
  Check for Pareto efficiency and social welfare implications.
"""


def _general_game(problem: str, players: int) -> str:
    return f"""**Game Solve: General {players}-Player Game Analysis**
Problem: {problem}

**Step 1 — Model the Game**
  - Players: identify all decision-makers.
  - Actions: enumerate each player's strategy set.
  - Payoffs: define utility/outcome for each strategy combination.
  - Information: perfect (all see everything) or imperfect?
  - Timing: simultaneous or sequential?

**Step 2 — Solve by Game Type**

  Sequential (extensive form): Use backward induction.
    Start at terminal nodes. Each player at each node picks the action
    maximising their payoff. Roll back to root.

  Simultaneous (normal form): Use best-response / Nash equilibrium.
    Build payoff matrix. Find cells where no player wants to deviate.

  Cooperative: Find the core, Shapley value, or bargaining solution.

**Step 3 — Identify Key Concepts**
  - Dominant strategies (always best regardless of opponents)
  - Dominated strategies (always worse — eliminate iteratively: IESDS)
  - Subgame Perfect Equilibrium (for sequential games)
  - Repeated game effects (cooperation, reputation, punishment)

**Step 4 — Verify and Interpret**
  Does the equilibrium match intuition? Are there multiple equilibria?
  (If so, discuss focal points / coordination mechanisms.)
  What is the social cost of the equilibrium vs the social optimum?
"""

from skills.base_skill import BaseSkill


class GameSolveSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "game_solve"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The problem to solve"},
                "depth":   {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                "players": {"type": "integer"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        players = kwargs.get("players", 2)
        low = problem.lower()
        if any(k in low for k in ("nim", "stones", "matches", "tokens", "take")):
            return _nim(problem)
        if any(k in low for k in ("chess", "tic-tac-toe", "tictactoe", "connect")):
            return _minimax_generic(problem)
        if any(k in low for k in ("prisoner", "dilemma", "cooperate", "defect")):
            return _prisoners_dilemma(problem)
        if any(k in low for k in ("auction", "bid", "sealed")):
            return _auction(problem)
        if any(k in low for k in ("nash", "equilibrium", "equilibria")):
            return _nash(problem)
        return _general_game(problem, int(players))
