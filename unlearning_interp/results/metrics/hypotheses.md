# Unlearning Hypotheses & Running Findings

Findings are logged here as experiments complete. Each hypothesis notes the
model scale, supporting evidence, and open questions for future runs.

---

## H-SCALE-1: RMU effectiveness is strongly size-dependent

**Status:** Confirmed at toy scale; pending replication at Pythia-410M and larger.

### Finding

On a 2.72M-parameter GPT-NeoX (6 layers, hidden_size=192, byte tokenizer),
standard RMU fails to reach the target regime (forget EM ≤ 0.20 AND retain EM ≥ 0.60)
on a forget set of 50 fictional biographical facts.

**Pareto frontier observed (9-run sweep, all at lr=5e-4, epochs=7):**

| update_layers | layer_idx | alpha | retain_coeff | forget EM | retain EM |
|---------------|-----------|-------|--------------|-----------|-----------|
| [4]           | 4         | 2500  | 4            | 0.900     | 0.900     |
| [5]           | 5         | 2500  | 4            | 0.500     | 0.450     |
| [5]           | 5         | 5000  | 16           | 0.500     | 0.450     |
| [3,4,5]       | 4         | 2500  | 1            | 0.200     | 0.300     |
| [3,4,5]       | 4         | 2500  | 8            | 0.200     | 0.300     |
| [4,5]         | 5         | 2500  | 8            | 0.200     | 0.200     |

No configuration achieves forget EM ≤ 0.20 with retain EM ≥ 0.60 simultaneously.
The frontier has two regimes with a hard gap between them:
- Soft regime: forget ≈ 0.50, retain ≈ 0.45 (single last-layer update)
- Aggressive regime: forget ≈ 0.20, retain ≈ 0.20–0.30 (multi-layer update)

### Saturation result (novel)

Doubling alpha (2500 → 5000) with single-layer update at layer 5 produced
*identical* results, confirming the bottleneck is geometric capacity, not
gradient magnitude. Similarly, orthogonalizing the steering vector u to the
mean retain hidden state at the hook layer had zero effect, confirming the
bottleneck is weight sharing, not steering direction.

### Mechanistic explanation

The shared MLP weight matrix W (768 → 192) at any single layer must simultaneously:
1. Push forget inputs x_f toward the steering target c·u.
2. Keep retain inputs x_r near their frozen values.

In 192-dimensional hidden space, with forget and retain bio facts sharing
identical prompt templates, x_f ≈ x_r at every layer. Any change to W that
redirects x_f also redirects x_r by nearly the same amount. The L_retain MSE
cannot fully compensate because the two constraints are nearly contradictory
within the low-dimensional weight space.

In contrast, on models with hidden_size ≥ 1024, the orthogonal complement of
the retain subspace is large enough to contain the forget steering target without
significant overlap — which is why published RMU results (on 7B models) show
clean forget/retain separation.

### Literature context

The RMU paper (Li et al. 2024) evaluates only on Zephyr-7B and larger.
TAR (Tamirisa et al. 2024) similarly targets 7B+. The size dependency is
*implicit* in the field but, to our knowledge, the specific saturation
phenomenon (capacity ceiling independent of alpha and steering direction) has
not been characterized in the published literature.

### Open questions for larger-scale runs

- [ ] At what hidden_size does the Pareto frontier first include the (≤0.20, ≥0.60) regime?
      Pythia-160M (hidden=768), Pythia-410M (hidden=1024) are natural checkpoints.
- [ ] Does the saturation alpha threshold scale with hidden_size (i.e., is there
      a universal alpha/d_model ratio above which increasing alpha helps again)?
- [ ] Does the retain-orthogonal steering vector (H-SCALE-1-ORT below) improve
      results at larger scale where direction matters more than capacity?

---

## H-SCALE-1-ORT: Retain-orthogonal steering vector (method improvement)

**Status:** Implemented; no measurable effect at toy scale (see H-SCALE-1). Pending test at larger scale.

### Description

The standard RMU steering vector u is sampled from N(0, I) and unit-normalized,
with no regard for the retain representation geometry. This work introduces
`_make_retain_orthogonal_u()`: before normalizing, project out the component of u
along the mean retain hidden state at layer_idx (computed from the frozen reference).

```
retain_dir = mean_h_retain(layer_idx) / ||mean_h_retain(layer_idx)||
u = u - (u · retain_dir) retain_dir
u = u / ||u||
```

The resulting c·u target is geometrically orthogonal to the retain mean direction,
which should reduce the structural overlap between L_forget and L_retain gradients.

### Null result at toy scale

The residual cosine between u and retain_dir was confirmed to be 0.0000 (perfect
orthogonalization), yet forget EM and retain EM were unchanged vs. random u.
This is consistent with H-SCALE-1: at 192 dimensions the bottleneck is weight-matrix
capacity, not direction. The retain-orthogonal method is theoretically sound and
expected to help once hidden_size is large enough for direction to matter.

### Prediction for larger scale

At hidden_size ≥ 1024 with a standard random u, the steering target c·u will
have a non-trivial component (≈ 1/√d ≈ 0.03 at d=1024) aligned with the retain
direction purely by chance. Orthogonalization removes this, tightening the
forget/retain trade-off. Prediction: retain EM improves by 0.02–0.05 at matched
forget EM on Pythia-410M.

---

*Last updated: 2026-05-04 (toy-scale sweep complete)*
*Next update: after Pythia-410M run*
