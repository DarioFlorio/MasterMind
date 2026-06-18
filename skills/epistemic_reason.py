"""
Skill: epistemic_reason
Epistemic reasoning: knowledge, belief, justification, uncertainty, what can be known.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.epistemic_reason")

DESCRIPTION = (
    "Epistemic reasoning: analyse what is known vs believed vs assumed, "
    "track uncertainty, evaluate evidence quality, handle knowledge gaps. "
    "Use for 'how confident should we be?', 'what do we actually know?', "
    "'is this justified?', 'what are the limits of our knowledge here?'"
)


def _general_epistemic(problem: str, claim: str, evidence: str) -> str:
    cl_block = f"\nClaim under examination: {claim}" if claim else ""
    ev_block = f"\nEvidence: {evidence}" if evidence else ""
    return f"""**Epistemic Reasoning**
Problem: {problem}{cl_block}{ev_block}

**Framework: Knowledge, Belief, and Justification**

**Step 1 — Classify Each Statement**
  For every claim, assign it to:
  [K] Known: true, believed, and well-justified (by reliable evidence/argument)
  [B] Believed: held as true but not fully justified
  [A] Assumed: taken for granted, unexamined
  [U] Unknown: relevant but not yet determined
  [F] False: can be shown to be incorrect

**Step 2 — Examine Justification**
  For each [K] claim: what is the justification chain?
    - Direct observation / measurement
    - Deduction from other known facts
    - Induction from repeated observations
    - Testimony from reliable source
  How many steps of inference? Each step introduces potential error.

**Step 3 — Identify Epistemic Risks**
  - Confirmation bias: selectively attending to supporting evidence
  - Availability bias: overweighting vivid/recent examples
  - Authority bias: deferring to credentials over content
  - Unknown unknowns: what relevant information might exist but not be considered?

**Step 4 — Calibrate Confidence**
  Assign a credence (0–100%) to each key claim.
  Calibrated = your stated 70% claims are true ~70% of the time.
  Overconfidence is the most common epistemic error.

**Step 5 — Knowledge Gaps**
  What would we need to know to be more certain?
  What is the cost of acting vs waiting for more information?

**Step 6 — Conclusion with Epistemic Humility**
  State what is firmly established, what is likely but uncertain,
  and what remains genuinely open.
"""


def _knowledge_analysis(problem: str, claim: str, evidence: str) -> str:
    return f"""**Epistemic Reasoning: Knowledge Analysis**
Problem: {problem}

**Classical Analysis of Knowledge (JTB):**
  Knowledge = Justified True Belief
  S knows P if and only if:
    1. P is true (truth condition)
    2. S believes P (belief condition)
    3. S is justified in believing P (justification condition)

**Gettier Problem:**
  JTB is not sufficient. Counter-example:
  You believe truly (by lucky coincidence) that P, via valid-seeming but
  unreliable inference. You satisfy JTB but don't "know" P.
  Modern epistemology adds: reliability, safety, or sensitivity conditions.

**Reliability Condition (Reliabilism):**
  S knows P if S's belief that P was formed by a reliable process
  (one that generally produces true beliefs).

**Safety Condition:**
  S knows P if in nearby possible worlds where S believes P, P is true.
  (Prevents lucky true beliefs from counting as knowledge.)

**Apply to this claim:**
  1. Is it true? (What evidence determines truth value?)
  2. Does the reasoner genuinely believe it?
  3. Is the belief justified? By what method?
  4. Is the justification method reliable?
  5. Would the belief be true in similar circumstances? (Safety check)
"""


def _uncertainty_analysis(problem: str, claim: str) -> str:
    return f"""**Epistemic Reasoning: Uncertainty Analysis**
Problem: {problem}

**Types of Uncertainty:**
  Aleatory: irreducible randomness (e.g., quantum events, dice rolls).
            Cannot be reduced by more information.
  Epistemic: uncertainty due to lack of knowledge.
             CAN be reduced by gathering more evidence.
  Deep / Knightian: don't even know the probability distribution.
                    Model uncertainty, not just parameter uncertainty.

**Representing Uncertainty:**
  Point estimate: single number (hides uncertainty structure)
  Interval: [low, high] with confidence level
  Distribution: full probability distribution over possible values
  Scenario set: discrete set of possible states of the world

**Calibration Check:**
  How often were your past 90%-confident claims actually correct?
  If less than 90%: you are overconfident. If more than 90%: underconfident.

**Propagating Uncertainty:**
  When chaining inferences A → B → C:
  Uncertainty compounds. If P(A)=0.9, P(B|A)=0.9, P(C|B)=0.9:
  P(C) = 0.9³ = 0.729 — far more uncertain than each step suggests.

**Decision under Uncertainty:**
  Expected Utility: Σ P(outcome) × U(outcome)
  Maximin: choose action whose worst case is best (risk-averse)
  Maximax: choose action with best possible case (risk-seeking)
  Minimax Regret: minimise the regret from the worst-case decision error

**Apply to this problem:**
  Classify uncertainty type, quantify it, check calibration, decide.
"""


def _evidence_evaluation(problem: str, evidence: str) -> str:
    ev_block = f"\nEvidence to evaluate: {evidence}" if evidence else ""
    return f"""**Epistemic Reasoning: Evidence Evaluation**
Problem: {problem}{ev_block}

**Evidence Quality Hierarchy (strongest to weakest):**
  1. Systematic review / meta-analysis (of well-designed RCTs)
  2. Randomised Controlled Trial (RCT)
  3. Cohort study / Natural experiment
  4. Case-control study
  5. Cross-sectional survey
  6. Expert opinion / case series
  7. Anecdote / single case report

**Checklist for Any Evidence Source:**
  □ Who produced it? (Conflict of interest? Expertise relevant?)
  □ How was the sample selected? (Representative? Selection bias?)
  □ What was the control or comparison group?
  □ How large was the effect? (Statistical vs practical significance)
  □ How precise is the estimate? (Confidence interval width)
  □ Has it been replicated? (One study = unreliable; consistent replications = strong)
  □ Pre-registered? (Registered before data collection → less p-hacking)
  □ Publication bias: might contradicting studies exist unpublished?

**Red Flags:**
  - "Studies show…" with no citation
  - Single study with huge effect size, never replicated
  - Source has financial stake in the conclusion
  - Cherry-picked data (trend starts at convenient point)
  - Absolute vs relative risk confusion

**Conclusion:**
  Rate the evidence as: Strong / Moderate / Weak / Insufficient.
  State what would strengthen the evidence.
"""

from skills.base_skill import BaseSkill


class EpistemicReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "epistemic_reason"

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
                "claim":    {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        claim    = kwargs.get("claim", "")
        evidence = kwargs.get("evidence", "")
        low = problem.lower()
        if any(k in low for k in ("know", "believe", "justified", "justification", "gettier")):
            return _knowledge_analysis(problem, claim, evidence)
        if any(k in low for k in ("uncertain", "confidence", "credence", "probability")):
            return _uncertainty_analysis(problem, claim)
        if any(k in low for k in ("evidence", "source", "reliable", "trustworthy", "bias")):
            return _evidence_evaluation(problem, evidence)
        return _general_epistemic(problem, claim, evidence)
