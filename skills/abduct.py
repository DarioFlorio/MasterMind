"""
Skill: abduct
Abductive reasoning: infer the BEST explanation from incomplete evidence.
"Inference to the best explanation" — work backwards from observations.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.abduct")

DESCRIPTION = (
    "Abductive reasoning: find the best explanation for a set of observations. "
    "Use for diagnosis, detective reasoning, anomaly explanation, "
    "'what explains this?', 'why would this happen?', mystery solving."
)


def _general_abduction(problem: str, evidence: str, domain: str) -> str:
    ev_block = f"\nEvidence provided: {evidence}" if evidence else ""
    dom_block = f"\nDomain: {domain}" if domain else ""
    return f"""**Abductive Reasoning — Inference to the Best Explanation**
Problem: {problem}{ev_block}{dom_block}

**Phase 1 — Observe**
  List all facts/observations/symptoms that need explaining.
  Be precise: what is actually observed vs inferred?

**Phase 2 — Generate Hypotheses**
  Brainstorm every hypothesis H that would explain the observations O:
    "If H were true, then O would be expected."
  Aim for 3–7 candidate explanations. Include unlikely ones initially.

**Phase 3 — Evaluate Each Hypothesis**
  Score each hypothesis on:

  a) Explanatory Power: Does H fully explain ALL observations?
     Partial coverage = weaker hypothesis.

  b) Prior Probability: How likely was H before seeing the evidence?
     Common causes are usually more probable than rare ones.
     (Occam: prefer simpler explanations when explanatory power is equal.)

  c) Predictive Success: Does H predict additional facts we can check?
     A hypothesis that makes novel confirmed predictions is stronger.

  d) Coherence: Is H consistent with background knowledge?
     A hypothesis that requires many auxiliary assumptions is weaker.

  e) Uniqueness: Does H explain things the alternatives cannot?

**Phase 4 — Rank and Select**
  Best explanation = highest combined score across (a)–(e).
  Document the ranking explicitly:
    H1: [score rationale]
    H2: [score rationale]
    ...
  Select the best; acknowledge runner-up and what evidence would distinguish them.

**Phase 5 — Test and Update**
  What observation or experiment would MOST DECISIVELY distinguish H1 from H2?
  If testable: prescribe that test. If not: express residual uncertainty.

**Key Abductive Pitfalls:**
  - Post hoc ergo propter hoc (correlation ≠ causation)
  - Anchoring on the first plausible explanation
  - Ignoring base rates (rare explanations need strong evidence)
  - Forgetting that multiple causes can co-occur
"""


def _debug_abduction(problem: str, evidence: str) -> str:
    ev_block = f"\nAdditional evidence: {evidence}" if evidence else ""
    return f"""**Abductive Reasoning: Software Debugging**
Problem: {problem}{ev_block}

**Step 1 — Characterise the Failure**
  - What is the exact error message / unexpected behaviour?
  - When did it start? (What changed recently?)
  - Is it deterministic or intermittent?
  - What inputs/conditions trigger it?

**Step 2 — Generate Hypotheses (most → least likely)**
  H1: Recent code change introduced the bug
  H2: Dependency version change broke an assumption
  H3: Environment difference (OS, config, env vars)
  H4: Data / input edge case not previously hit
  H5: Race condition / concurrency issue
  H6: Resource exhaustion (memory, file handles, DB connections)

**Step 3 — Rank by Explanatory Power**
  For each H: would it produce EXACTLY these symptoms?
  Check: does the stack trace / error code point to a specific layer?

**Step 4 — Rapid Falsification**
  Cheapest tests first:
  - Check git log for recent changes in the affected module
  - Reproduce with minimal input
  - Binary search: does reverting the last change fix it?
  - Add logging/assertions around the suspect area

**Step 5 — Confirm the Root Cause**
  The best explanation is confirmed when:
  (a) Removing/fixing H causes the bug to disappear
  (b) Re-introducing H reliably brings it back
"""


def _medical_abduction(problem: str, evidence: str) -> str:
    ev_block = f"\nPresenting evidence: {evidence}" if evidence else ""
    return f"""**Abductive Reasoning: Differential Diagnosis**
Problem: {problem}{ev_block}

**Step 1 — Symptom Inventory**
  List: chief complaint, associated symptoms, timeline, severity, modifying factors.
  Vital signs, lab values, imaging findings.

**Step 2 — Differential Diagnosis (broad to narrow)**
  Common diseases presenting with these symptoms:
  1. [Most common cause]
  2. [Second most common]
  3. [Must-not-miss / dangerous cause]
  4. [Rare but fitting cause]

  Use: "VITAMIN C D" mnemonic (Vascular, Infection, Trauma, Autoimmune,
  Metabolic/Idiopathic, Neoplastic, Congenital, Degenerative)

**Step 3 — Discriminating Features**
  For each pair of top diagnoses, identify the single test or finding that
  best discriminates between them. High sensitivity test rules OUT; high
  specificity test rules IN.

**Step 4 — Bayesian Weighting**
  Adjust probabilities by: base rate × likelihood ratio for each finding.
  LR+ = sensitivity / (1 − specificity)
  LR− = (1 − sensitivity) / specificity

**Step 5 — Working Diagnosis + Next Step**
  Best explanation = highest posterior probability.
  Prescribe the confirmatory test; plan treatment while awaiting results
  if the working diagnosis is high-stakes.

**Disclaimer:** This is a reasoning framework. Clinical decisions require
  qualified medical professionals.
"""


def _detective_abduction(problem: str, evidence: str) -> str:
    ev_block = f"\nClues: {evidence}" if evidence else ""
    return f"""**Abductive Reasoning: Detective / Whodunit Analysis**
Problem: {problem}{ev_block}

**Step 1 — Establish Facts**
  Separate confirmed facts from assumptions and hearsay.
  Build a timeline: what happened, in what order, where, who was present?

**Step 2 — Identify Suspects and Motives**
  For each person of interest:
  - Motive: reason to commit the act
  - Means: capability to do it
  - Opportunity: access at the time

**Step 3 — Evaluate Each Suspect**
  Score each on Motive × Means × Opportunity.
  Which suspect best explains ALL the clues together?
  (Not just one clue — the best explanation covers them all.)

**Step 4 — Alibi and Elimination**
  Confirmed alibis eliminate suspects. Check each alibi's source.
  What evidence would exist if a given suspect were guilty — and is it present?

**Step 5 — Converging Evidence Test**
  The correct suspect:
  (a) Has no confirmed alibi
  (b) Has clear motive + means + opportunity
  (c) Is consistent with ALL physical evidence
  (d) Explains why innocent-looking clues are misleading

**Step 6 — State Best Explanation with Confidence**
  "The most probable explanation is [X] because [evidence A, B, C] all point
  to [X], and no other hypothesis accounts for [evidence C] without additional
  implausible assumptions."
"""

from skills.base_skill import BaseSkill


class AbductSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "abduct"

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
                "evidence": {"type": "string", "description": "Supporting evidence"},
                "domain": {"type": "string", "description": "Domain context"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        evidence = kwargs.get("evidence", "")
        domain   = kwargs.get("domain", "")
        low = problem.lower()
        if any(k in low for k in ("bug", "crash", "error", "exception", "fail")):
            return _debug_abduction(problem, evidence)
        if any(k in low for k in ("symptom", "disease", "diagnosis", "patient", "medical")):
            return _medical_abduction(problem, evidence)
        if any(k in low for k in ("murder", "crime", "suspect", "detective", "whodunit", "clue")):
            return _detective_abduction(problem, evidence)
        return _general_abduction(problem, evidence, domain)
