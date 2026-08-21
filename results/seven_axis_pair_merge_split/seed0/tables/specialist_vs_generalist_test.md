## Test partition (withheld, cap 1k / benchmark)

| Axis | Specialist | Spec NLL | Pooled NLL | Δ (spec−pool) | Spec wins |
|---|---|---:|---:|---:|:---:|
| Harm | `merge-harm` | 1.8139 | 1.8178 | -0.0040 | ✓ |
| Hallucination | `merge-hallucination` | 2.0075 | 2.0344 | -0.0269 | ✓ |
| Jailbreak | `merge-injection` | 1.2336 | 1.2210 | +0.0127 |  |
| Privacy | `merge-privacy` | 1.7205 | 1.7146 | +0.0059 |  |
| Over-refusal | `merge-overrefusal` | 1.9350 | 2.0502 | -0.1153 | ✓ |
| Injection | `merge-injection` | 2.5912 | 2.5653 | +0.0259 |  |
| Policy | `merge-policy` | 2.0964 | 2.1304 | -0.0340 | ✓ |

**Specialists beat pooled baseline on 4/7 axes** (lower NLL is better).
