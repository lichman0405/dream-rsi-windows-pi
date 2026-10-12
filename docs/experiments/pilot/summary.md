# MOF screening experiment — pilot phase

- Protocol digest: `9f5940ed36ffbedf3b4ce3a6f43a4d65ee07435df1a4727f15e6c9620430b0f8`
- Model: `cc-switch-deep-seek` / `deepseek-v4-pro`, thinking `off`
- Arm-runs recorded: 12; unsealed candidates: 12

## Cost

- Service requests: **1091** (refused before sending: 0)
- Tokens charged: **17,414,177** (input 13,643,823, output 3,352,178, reasoning 0)
- Requests with no reported usage: 0
- Money: unknown — the provider reports no per-request cost, so no currency figure is claimed.

## Main result

- Paired blocks: 4 complete of 4
- **A − B** on sealed-pool S: mean -0.0115, exploratory 95% [-0.0334, +0.0104], 1/2 splits favouring A
- Pre-registered reading: **insufficient_evidence** (pre-registered: A-B >= 0.05 S units, same direction in >= 4/5 splits, and an exploratory interval that does not cross 0)

## Development against sealed pool

| arm | runs | mean dev S | mean test S | mean gap | failures |
| --- | --- | --- | --- | --- | --- |
| A | 4 | 0.8223 | 0.8350 | -0.0127 | 0 |
| B | 4 | 0.8057 | 0.8465 | -0.0407 | 0 |
| C | 4 | 0.8157 | 0.7597 | +0.0559 | 0 |

## Figures

- `internal_mechanism`: PNG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig5_internal_mechanism.png`, SVG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig5_internal_mechanism.svg`, source data `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig5_internal_mechanism.csv`
- `paired_final_comparison`: PNG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig3_paired_final_comparison.png`, SVG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig3_paired_final_comparison.svg`, source data `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig3_paired_final_comparison.csv`
- `quality_budget_staircase`: PNG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig1_quality_budget_staircase.png`, SVG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig1_quality_budget_staircase.svg`, source data `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig1_quality_budget_staircase.csv`
- `recommendation_explanation`: PNG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig4_recommendation_explanation.png`, SVG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig4_recommendation_explanation.svg`, source data `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig4_recommendation_explanation.csv`
- `recursive_dynamics`: PNG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig2_recursive_dynamics.png`, SVG `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig2_recursive_dynamics.svg`, source data `C:\Users\lishi\code\dream-rsi-windows-pi\.runtime\mof_experiment\experiment\pilot\unsealed\figures\fig2_recursive_dynamics.csv`

## Limits

- The development pool is a finite sample of the same material library the sealed pool comes from, and the five splits share that pool. Intervals are exploratory and are not a claim about MOFs in general.
- The provider reports no per-request money cost, so the report gives tokens and leaves the currency figure unknown rather than estimating one.
- Extended reasoning was fixed `off` for every call in every arm; the probe measurements behind that choice are recorded in `src/dream_rsi/mof/protocol.py`.
- Internal Eq(1) `V` is a policy-selection signal. It is shown for mechanism only and is never used as evidence that screening improved.
