# Self‑Model module for EVE
import json
from pathlib import Path

MODEL_PATH = Path(__file__).parents[1] / "memdir" / "self_model.json"

DEFAULT_MODEL = {
    "capabilities": {},
    "limitations": {},
    "goals": [],
    "last_reward": null,
    "confidence": {},
    "metrics": {
        "tasks_completed": 0,
        "errors": 0,
        "avg_latency_ms": 0
    }
}

def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        save_model(DEFAULT_MODEL)
        return DEFAULT_MODEL

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

def update_model(updates: dict):
    model = load_model()
    model.update(updates)
    save_model(model)
    return model

if __name__ == "__main__":
    print("Current self‑model:")
    print(json.dumps(load_model(), indent=2))
