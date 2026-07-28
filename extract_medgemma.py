"""OLDCARTS extraction via self-hosted MedGemma 27B (SGLang, FP8+FP8-KV) instead of Gemini.

Reuses src/schema.py (Extraction) and src/extract_oldcarts.py (SYSTEM prompt, _clean,
_has_complaint) verbatim, so this is an apples-to-apples swap of the *model* only.

Env:
  MEDGEMMA_URL    base url  (default = the Shakti deployment below)
  MEDGEMMA_TOKEN  bearer token (deployment has Enable Auth = True)

Usage:
  python -u extract_medgemma.py --n 3            # 3 real NAS_v3 notes through MedGemma
  python -u extract_medgemma.py --n 3 --gemini   # ...and Gemini, side by side
  python -u extract_medgemma.py --health         # is the endpoint up yet?
"""
import os, sys, json, time, argparse
import requests

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)
from schema import Extraction                                    # noqa: E402
from extract_oldcarts import SYSTEM, _clean, _has_complaint      # noqa: E402
from config import EVAL_CSV, EVAL_NOTE_COL, CHAR_CAP             # noqa: E402

BASE  = os.environ.get("MEDGEMMA_URL", "https://http.aygjutg9wi.shaktistudio.shakticloud.ai").rstrip("/")
TOKEN = os.environ.get("MEDGEMMA_TOKEN", "")
MODEL = "google/medgemma-27b-text-it"

# ponytail: pydantic emits $defs/$ref; xgrammar handles refs. If it ever rejects them,
# inline with jsonref rather than hand-maintaining a second copy of the schema.
_SCHEMA = {"type": "json_schema",
           "json_schema": {"name": "Extraction", "schema": Extraction.model_json_schema()}}


def _headers():
    h = {"Content-Type": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


def health() -> str:
    r = requests.get(f"{BASE}/health", headers=_headers(), timeout=15)
    return f"{r.status_code} {r.text[:200]}"


def extract(note: str, retries: int = 3, temperature: float = 0.0) -> tuple[Extraction, dict]:
    """Same contract as src.extract_oldcarts.extract, against MedGemma. Returns (extraction, stats)."""
    want = _has_complaint(note)
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = requests.post(
                f"{BASE}/v1/chat/completions", headers=_headers(), timeout=300,
                json={"model": MODEL,
                      "messages": [{"role": "user", "content": SYSTEM + note[:CHAR_CAP]}],
                      "temperature": temperature if attempt == 0 else 0.2,
                      "max_tokens": 2048,
                      "response_format": _SCHEMA})
            r.raise_for_status()
            body = r.json()
            ex = Extraction.model_validate_json(body["choices"][0]["message"]["content"])
            if want and not ex.symptoms_reported:
                raise ValueError("empty symptoms_reported on a note with complaints")
            u = body.get("usage", {})
            return _clean(ex), {"s": round(time.time() - t0, 2),
                                "in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                                "tok_s": round(u.get("completion_tokens", 0) / max(time.time() - t0, 1e-9), 1)}
        except Exception as e:
            last = e
            if attempt == retries - 1:
                break
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"medgemma extraction failed after {retries}: {last}")


def _notes(n: int) -> list[str]:
    import pandas as pd
    df = pd.read_csv(EVAL_CSV)
    return [str(x) for x in df[EVAL_NOTE_COL].dropna().head(n).tolist()]


def demo():
    """Self-check on the same synthetic note src/extract_oldcarts.py uses, with hard asserts."""
    note = ("Gender: Female\n Age: 26 years\n Chief_complaint: Headache : Duration - 4 Days. "
            "Site - Diffuse. Severity - Mild. Onset - Acute onset (can recall exact time). "
            "Character of headache - Throbbing, Dull continuous. Radiation - pain does not radiate. "
            "Timing - Day, Night. Exacerbating factors - bending, lifting.\n"
            " Associated symptoms: Patient reports - Muscle pain, Photophobia. "
            "Patient denies - Vomiting, Nausea, Fever.")
    ex, st = extract(note)
    print(json.dumps(ex.model_dump(), indent=1, ensure_ascii=False), st, sep="\n")
    names = {s.name.lower() for s in ex.symptoms_reported}
    assert any("head" in x for x in names), f"headache not extracted: {names}"
    ha = next(s for s in ex.symptoms_reported if "head" in s.name.lower())
    assert ha.oldcarts.duration, "duration dropped"
    assert ha.oldcarts.severity == "mild", f"severity={ha.oldcarts.severity}"
    assert ha.oldcarts.radiation_none is True, "explicit-absent radiation flag missed"
    assert {"vomiting", "nausea", "fever"} <= {d.lower() for d in ex.symptoms_denied}, ex.symptoms_denied
    print("\ndemo OK — schema-valid, symptom-bound, polarity + explicit-absent all correct")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=0, help="run N real NAS_v3 notes")
    p.add_argument("--health", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--gemini", action="store_true", help="also run Gemini on the same notes")
    a = p.parse_args()

    if a.health:
        print(BASE, "->", health())
    if a.demo:
        demo()
    for i, note in enumerate(_notes(a.n)):
        print(f"\n{'='*70}\nNOTE {i}\n{note[:600]}\n{'-'*70}")
        ex, st = extract(note)
        print("MEDGEMMA", st, json.dumps(ex.model_dump(), ensure_ascii=False, indent=1), sep="\n")
        if a.gemini:
            import extract_oldcarts as g
            t0 = time.time()
            gx = g.extract(note)
            print("\nGEMINI", {"s": round(time.time() - t0, 2)},
                  json.dumps(gx.model_dump(), ensure_ascii=False, indent=1), sep="\n")
