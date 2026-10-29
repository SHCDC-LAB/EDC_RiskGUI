# MODEL CARD — EDC-AOPRisk V10B.1 Audit-Frozen Publication Research Model

## Status
Audit-frozen for manuscript/SI use.

## Activity deployment
- 41 activity endpoints
- 3 externally gate-approved V10 upgrades
- 38 frozen V9 endpoints

## Gate-approved upgrades
- AR_AGONIST
- AR_ANTAGONIST
- ESR1_AGONIST

## Other layers
- 40 continuous potency models
- 63 potency-threshold classifiers
- endpoint-specific applicability domain
- conformal potency uncertainty
- strict-confirmed vs usable-only observed evidence
- Core-EATS vs supporting mechanism hierarchy

## Audit
- Source package manifest: PASS
- 147 model bundles load/schema: PASS
- external metric recalculation: PASS
- 251 x 41 completeness: PASS
- hardened predictor rerun: byte-identical CSV outputs
- 38 inherited endpoints: unchanged from V9 within machine precision

## Important limitations
External validation is not claimed for all 41 endpoints. It was used to adjudicate the 8 V10-A upgrade candidates.
Some endpoints have limited internal discrimination and are constrained by evidence hierarchy/AD rather than used as standalone hazard classifiers.
