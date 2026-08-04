"""
Random-pairing control for BERTScore (reviewer request): shuffle hypotheses
against mismatched references and compare BERTScore F1 to the real paired
scores, to show the metric discriminates matched from mismatched pairs
rather than reflecting generic domain similarity.

Run: python src/evaluation/bertscore_control.py
"""
import json
import random
from pathlib import Path

from bert_score import score as bert_score
from bert_score.utils import get_hash, lang2model, model2layers

ROOT = Path(__file__).parents[2]
S3_DIR = ROOT / "results" / "stage3_reports"

hyps, refs = [], []
for f in sorted(S3_DIR.glob("report_*.json")):
    d = json.loads(f.read_text())
    if d.get("reference_description") and d.get("report") and not d["report"].startswith("ERROR"):
        hyps.append(d["report"])
        refs.append(d["reference_description"])

n = len(hyps)
print(f"n = {n}")

random.seed(42)
perm = list(range(n))
while True:
    random.shuffle(perm)
    if all(perm[i] != i for i in range(n)):
        break
shuffled_refs = [refs[i] for i in perm]

model = lang2model["en"]
layers = model2layers[model]
h = get_hash(model, layers, False, False, False, False)
print("bert_score hash:", h)

_, _, F1_matched = bert_score(hyps, refs, lang="en", verbose=False)
_, _, F1_matched_resc = bert_score(hyps, refs, lang="en", rescale_with_baseline=True, verbose=False)
_, _, F1_shuffled = bert_score(hyps, shuffled_refs, lang="en", verbose=False)
_, _, F1_shuffled_resc = bert_score(hyps, shuffled_refs, lang="en", rescale_with_baseline=True, verbose=False)

result = {
    "bert_score_hash": h,
    "n": n,
    "matched": {
        "raw_mean": round(float(F1_matched.mean()), 4),
        "raw_std": round(float(F1_matched.std()), 4),
        "rescaled_mean": round(float(F1_matched_resc.mean()), 4),
        "rescaled_std": round(float(F1_matched_resc.std()), 4),
    },
    "shuffled_control": {
        "raw_mean": round(float(F1_shuffled.mean()), 4),
        "raw_std": round(float(F1_shuffled.std()), 4),
        "rescaled_mean": round(float(F1_shuffled_resc.mean()), 4),
        "rescaled_std": round(float(F1_shuffled_resc.std()), 4),
    },
    "seed": 42,
    "note": "shuffled_control pairs each hypothesis with a mismatched reference via a fixed derangement (no fixed point) of the 50-item index list.",
}
print(json.dumps(result, indent=2))

out = ROOT / "results" / "metrics" / "bertscore_random_pairing_control.json"
out.write_text(json.dumps(result, indent=2))
print(f"Saved -> {out}")
