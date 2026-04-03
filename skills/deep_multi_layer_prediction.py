"""
Skill: deep_multi_layer_prediction
Deep multi-layer prediction: stack predictions on predictions, model
emergent properties that arise only at higher levels of abstraction.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.deep_multi_layer_prediction")

DESCRIPTION = (
    "Deep multi-layer prediction: build predictions in layers where higher layers "
    "depend on lower-layer forecasts, like a prediction stack. Use for "
    "'deep future prediction', 'how does X affect Y which affects Z?', "
    "complex interdependent forecasts, emergent future phenomena."
)


def _deep_predict(problem: str, layers: int) -> str:
    layer_names = [
        "Physical / Empirical Layer",
        "Behavioural / Agent Layer",
        "Institutional / Systemic Layer",
        "Emergent / Cultural Layer",
        "Meta / Civilisational Layer",
    ]
    layer_descs = [
        "What will happen at the level of measurable facts, data, and direct observations?",
        "How will individuals and organisations behave in response to Layer 1 changes?",
        "How will institutions, markets, governments, and systems adapt to Layer 2 behaviour?",
        "What norms, cultures, narratives, and emergent phenomena arise from Layer 3 changes?",
        "What are the long-arc civilisational or meta-level implications?",
    ]

    layer_blocks = "\n\n".join([
        f"""**Layer {i+1}: {layer_names[i] if i < len(layer_names) else f'Level {i+1}'}**
  {layer_descs[i] if i < len(layer_descs) else 'How do effects from the previous layer cascade and compound?'}

  Predictions at this layer:
    a) [Most likely outcome at this level]
    b) [Alternative outcome if Layer {i} prediction was wrong]

  Confidence at this layer: [%] — Note: uncertainty accumulates across layers.
  Key assumption from Layer {i}: [what must be true from below for this to hold]
  Emergent properties: [what appears at THIS level that was invisible at lower levels]"""
        for i in range(min(layers, 5))
    ])

    return f"""**Deep Multi-Layer Prediction**
Problem: {problem}
Layers: {layers}

**Philosophy: Prediction Stacks**
  Complex futures are not flat — they unfold in layers.
  Layer 1 predictions feed into Layer 2, which feeds into Layer 3, etc.
  Each layer has its own logic, timescale, and uncertainty.
  Emergent properties appear only at higher layers and cannot be predicted
  from lower layers alone.

  Uncertainty COMPOUNDS across layers. A 5-layer stack with 80% confidence
  per layer gives: 0.8^5 ≈ 33% overall confidence.
  Be honest about this — don't false-precision aggregate into false certainty.

**The Prediction Stack:**

{layer_blocks}

**Cross-Layer Feedback Loops**
  Layers don't just flow downward — feedback operates upward too.
  Identify any loops where higher-layer outcomes FEED BACK into lower layers:

  Example: Layer 3 institutional change → alters Layer 1 incentive structures
           → changes Layer 2 individual behaviour → reinforces Layer 3 change.

  Positive feedback loop: [describe if present — these cause tipping points]
  Negative feedback loop: [describe if present — these cause stabilisation]

**Tipping Points and Phase Transitions**
  At what threshold does the system shift qualitatively?
  (Not just "more of the same" but a fundamentally different state.)
  Identify any bifurcation points in the prediction stack.

**Error Propagation**
  Which layer is the most uncertain?
  → This layer is the weakest link — error here invalidates all downstream layers.
  What would we need to know to reduce uncertainty at that layer?

**Final Synthesis**
  Integrating all {layers} layers:
  Most likely deep outcome: [describe]
  Alternative trajectory (if Layer 2 prediction fails): [describe]
  Black swan risk (if Layer 1 assumption is wrong): [describe]

  Overall confidence: [%] (noting compounded uncertainty across all layers)
  Time horizon to observable confirmation: [when will we know if this is right?]
"""

from skills.base_skill import BaseSkill


class DeepMultiLayerPredictionSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "deep_multi_layer_prediction"

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
                "layers": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        layers = int(kwargs.get("layers", 4))
        return _deep_predict(problem, layers)
