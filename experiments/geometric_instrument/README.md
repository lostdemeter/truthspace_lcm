# Geometric Instrument

Building an LLM from first principles using the six geometric
components identified in DC 276/277. Empirically validated in DC 278.

**Result:** 15.4 KB of geometric constants reproduce the routing logic
of a 7.6B-parameter model at **6/6 accuracy** (better than baseline 5/6).
The model is 94.6% value computation and 5.4% routing — the routing
is entirely geometric, and the BOS MLP pump is a constant function.

## Findings Summary

| Finding | Phase | Key Result |
|:--------|:------|:-----------|
| F127 | 1–2 | 6/6 end-to-end instrument matches neural model |
| F128–129 | 3 | 5/6 full geometric layer at L23, softmax eliminated |
| F130 | 4 | MESH universal (112/112 rank-1), boundary at L22 |
| F131 | 4b | BOS sink 76%, content-independent routing |
| F132 | 4c | 5/6 frozen attention at ALL layers, 410M → 16 KB |
| F133 | 4d | Position-locked templates, right-align transfers |
| F134 | 4e | L3 BOS pump: rank-1 along W_down SV0, universal |
| F135 | 4f | Synthetic pump: h[0] += 7103.2 × sv0, 57,000× fewer FLOPs |
| F136 | 4g | Parametric T(N): 5 params/head, 100,000:1 compression |
| F137 | 5 | Combined assembly: 28 KB geometry + 7.6B neural, 5/6 |
| F138 | Frontier 1 | All-position attention content-independent (cos≥0.982) |
| F139 | Frontier 2 | BOS MLP cos=1.000 ALL 28 layers; sv0 replacement 6/6 |
| F140 | Combined | **Optimal assembly: 15.4 KB → 6/6. Cross-length N=5,7,9 ✓** |
| F141 | Frontier 3 | **Q/K elimination: content-independent within structure, per-structure cache = full skip** |
| F142 | Frontier 4 | **Two-phase model: L0-L19 structure encoding, L20-L27 universal extraction. Hybrid 4/4.** |
| F143 | Frontier 4b/c | **Selective head caching: cos≥0.99 → 46% cached, 10/10 PERFECT. General solver: 65% Q/K eliminated.** |
| F144 | Frontier 5 | **Sign-space navigation: signs alone ≈ random cross-structure. Gate codes universal at hourglass neck only. Levels NOT dispensable.** |
| F145 | Frontier 5b | **COMB zone Content Separator: MLP push-pull with attention (cos≈-0.2). Gyroscope strongest here. PRESERVE intermediates cross-struct cos=0.01. 7th structure.** |
| F146 | Frontier 6 | **Skip L10-L15 = 3/3 (better than baseline). Cache FAILS. Rank-1 oracle 3/3. Layer pruning > replacement. 12/28 layers geometric or skippable.** |
| F147 | Frontier 6b/c | **Signs=structure (50% flip cross-class, 2% within). Levels=content. 0/10 answer-dim sign flips for France→Germany. Structure IS binary, content IS continuous.** |
| F148 | Frontier 6d | **Rank-1 universal (97.1%), entity=0.5% level perturbation. ALL navigation fails—holistic barrier. 22/28 layers geometric or skippable, 6-7 irreducibly neural.** |
| F149 | Frontier 7 | **Sign-only COMB → Paris ✓ Berlin ✓ cos>0.91. Signs=shape (80% dir/weight). Full rank, unique/layer. Exponents=universal scale. 0 truly opaque layers.** |
| F150 | Frontier 7b-d | **MLP = rank-1 projector. Rank-1 gate ✓, W_up ✓, BOTH ✓. Gate swap: gap +7.33→-0.33. Output ⊥ v₁. COMB = bank of rank-1 projectors. 2960× compression.** |
| F151 | Frontier 8 | **5 classes: 18/20 rank-1 gate ✓. v₁ NOT orthogonal (cos 0.20–0.52). Filters NOT unique (cos 0.62–0.85). Wrong v₁ also works! Weight = hologram. BOTH rank-1: 10/10.** |
| F152 | Frontier 9 | **Workbench tools on weights. SV crystalline (ρ<0.01). Refinement UNIFIES (cos 0.97). Disparity: 4.3% sensitive neurons. Residuals autocorrelated. 3 classes = 0.14% energy. Hologram is DEEP.** |
| F153 | Frontier 10 | **Read-only barrier. Full swap → Berlin ✓. Rank-1 weight edit ✗ (gap -7.10). MLP delta ✗ (U-shaped). Hologram READ-ONLY at component level. MLP = amplifier, attention = reader.** |
| F154 | Frontier 11 | **Attention = reader CONFIRMED. Entity-pos swap (3584 nums) → Berlin emb–L20. Attn swap L22-23 → Berlin (+4.27). KV group 0 only group that matters. 0.0003% edit redirects answer.** |
| F155 | Frontier 12 | **4D IS ALL YOU NEED. Entity SVD 4D: 4/4 ✓✓✓✓. Entity diffs 3-dimensional. Gate ⊥ selector. 8D general: 71 ops (20M× reduction). 112 KB (20,857× compression). Just directions interfering.** |

## Directory Structure

```
geometric_instrument/
├── plan/
│   └── ARCHITECTURE.md        ← Build plan, phases, progress log
├── components/
│   ├── waveguide.py            ← Residual stream (carries signals by ⊕)
│   ├── stabilizer.py           ← Geometric Gyroscope (self-correction)
│   ├── decomposer.py           ← Geometric Spectrometer (spectral channels)
│   ├── selector.py             ← Geometric Selector (directional filter)
│   ├── resonator.py            ← Geometric Resonator (rank-1 locking)
│   ├── lens.py                 ← Geometric Lens (knowledge projection)
│   └── amplifier.py            ← Geometric Amplifier (coherent boosting)
├── instrument.py               ← Assembles components into full pipeline
├── verify_component.py         ← Per-component isolation tests
├── verify_instrument.py        ← End-to-end instrument test (6/6)
├── verify_geometric.py         ← Geometric replacement tests (5/6)
├── phase4_survey_layers.py     ← MESH survey: 112/112 rank-1
├── phase4_allayer_routing.py   ← All-layer routing test
├── phase4_routing_followup.py  ← Routing analysis
├── phase4b_distributed.py      ← Distributed attention analysis
├── phase4c_explore.py          ← Fixed-template attention
├── phase4d_length_gen.py       ← Template length generalization
├── phase4e_l3_explosion.py     ← L3 BOS mechanism investigation
├── phase4e_l3_weights.py       ← L3 weight structure analysis
├── phase4f_synth_pump.py       ← Synthetic BOS pump + initial templates
├── phase4g_param_template.py   ← Parametric template generator
├── phase5_assembly.py          ← Full geometric model assembly
├── frontier1_allpos_templates.py  ← F138: All-position content-independence
├── frontier1c_param_allpos.py     ← T(N,q) parametric attempt (0/6)
├── frontier2_mlp_geometry.py      ← F139: MLP geometry initial survey
├── frontier2b_mlp_deep.py         ← Deep MLP investigation
├── frontier2c_sv0_analysis.py     ← sv0 direction analysis (6/6)
├── frontier_combined.py           ← Full-template + sv0 tests
├── frontier_optimal.py            ← F140: Best combined (6/6)
├── frontier3_qk_elimination.py    ← F141: Q/K elimination diagnostic
├── frontier3b_scope_test.py       ← F141: Content-independence scope
├── frontier4_cross_structure.py   ← F142: Cross-structure & two-phase model
├── frontier4b_selective_heads.py  ← F143: Per-head sensitivity map & selective caching
├── frontier4c_token_qk_cache.py   ← F143: Token Q/K cache & full pipeline
├── frontier5_sign_navigation.py   ← F144: Sign-space navigation experiment
├── frontier5b_comb_zone.py        ← F145: COMB zone anatomy & Content Separator
├── frontier6_engineer_comb.py     ← F146: Engineering the COMB zone (skip/cache/rank)
├── frontier6b_phi_basis_comb.py   ← F147: φ-basis sign flips in raw 3584-d
├── frontier6c_knowledge_subspace.py ← F147: φ-basis in 128-d knowledge subspace
├── frontier6d_rank1_level.py      ← F148: Rank-1 × φ-level holistic barrier
├── frontier7_weight_shapes.py     ← F149: Weight sign shapes carry computation
├── frontier7b_shape_translation.py ← F150: Gate activation patterns & directions
├── frontier7c_rank1_manifold.py    ← F150: Rank-1 manifold & scalar navigation
├── frontier7d_gate_vs_up.py        ← F150: Gate/up decomposition & content swap
├── frontier8_multi_class_rank1.py  ← F151: Multi-class rank-1 & superposition test
├── frontier9_holographic_analysis.py ← F152: Holographic workbench tools on weight matrices
├── frontier10_hologram_writing.py   ← F153: Hologram writing — the read-only barrier
├── frontier11_attention_editing.py  ← F154: Attention editing — writing through the reader
└── frontier12_shape_computer.py     ← F155: The shape computer — 4D is all you need
```

## Quick Start

```bash
# Phase 1-2: Verify components and full instrument
python verify_component.py --component selector
python verify_instrument.py

# Phase 3: Geometric replacement at extraction layer
python verify_geometric.py

# Phase 5: Full geometric assembly (combined test)
python phase5_assembly.py
```

## See Also

- `plan/ARCHITECTURE.md` — Full build plan with phases and progress log
- `docs/design_considerations/278_the_geometric_decomposition.md` — **Synthesis of all findings**
- `docs/design_considerations/277_the_transformer_as_geometric_instrument.md` — Theoretical derivation
- `docs/design_considerations/276_geometric_structures_taxonomy.md` — Component taxonomy
- `experiments/model_reverse_engineering_v2/FINDINGS.md` — Detailed findings F127–F140
