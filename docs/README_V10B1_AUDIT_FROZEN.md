# EDC-AOPRisk V10B.1 — Audit-Frozen Publication Research Model

V10B.1 is a **packaging/audit hardening** of V10B.0. No scientific prediction model was retrained.

Key freeze rule:
- AR_AGONIST, AR_ANTAGONIST, ESR1_AGONIST use externally approved V10 activity upgrades.
- All other 38 activity endpoints are forced to the frozen V9 models.
- Potency and threshold layers remain frozen V9.

The production-facing research predictor is:

`scripts/predict_new_smiles_v10b_audit_frozen.py`

Deployment selection is registry-driven and asserted against:

- `MODEL_SELECTION_REGISTRY_V10B_FINAL.csv`
- `V10B_EXTERNAL_EVIDENCE_GATE_REGISTRY.csv`

Audit evidence is in `audit/`.

Frozen 251 prediction rerun:
- 4 CSV outputs byte-identical to V10B.0.
- No scientific output changed.

Publication wording:
This is a confidence-aware endocrine bioactivity/potency prioritization research model, not a standalone regulatory QSAR or replacement for experimental toxicology.
