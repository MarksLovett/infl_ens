## Train partition (cap 1k / benchmark)

| Axis | Specialist | Spec NLL | Pooled NLL | Δ (spec−pool) | Spec wins |
|---|---|---:|---:|---:|:---:|
| Harm | `merge-harm` | 1.7679 | 1.7390 | +0.0289 |  |
| Hallucination | `merge-hallucination` | 1.9427 | 1.9328 | +0.0100 |  |
| Jailbreak | `merge-injection` | 1.1925 | 1.1209 | +0.0716 |  |
| Privacy | `merge-privacy` | 1.6853 | 1.6580 | +0.0274 |  |
| Over-refusal | `merge-overrefusal` | 1.8930 | 2.0097 | -0.1167 | ✓ |
| Injection | `merge-injection` | 2.7568 | 2.7057 | +0.0511 |  |
| Policy | `merge-policy` | 2.0567 | 2.0612 | -0.0045 | ✓ |

**Specialists beat pooled baseline on 2/7 axes** (lower NLL is better).
