# Tomorrow

Review the completed four-hour 7x7 Go screenings before selecting more experiments or combining any changes.

## Results to review

- Baseline.
- Learning-rate decay.
- Constant learning rate.
- Mixed search with 25% full searches.
- Any completed single-variable runs for reduced-parent FPU, restart states, remaining game length, or forced playouts.

## Implementation topics

- Calibrate adaptive search termination from full-search traces before enabling it.
- Add resignation audit instrumentation before considering active resignation.
- Design exact-state replay deduplication with aggregated targets and square-root multiplicity weighting.
- Design reanalysis around reconstructible trajectories and stable replay provenance.
- Consider a dedicated global-pooling ablation against the existing squeeze-excitation blocks.

## Experiment decisions

- Choose the next confirmation seeds from the strongest informative results.
- Decide whether a fifth new single-variable run is justified.
- Do not combine methods until each component has an informative individual result.
