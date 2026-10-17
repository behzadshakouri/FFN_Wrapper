# FFN_Wrapper

## About

FFN_Wrapper is the C++ research implementation of a time-lagged feed-forward
neural-network surrogate for Activated Sludge Model No. 1 (ASM1) simulations.
Its primary workflow trains and evaluates emulators for wastewater-treatment
outputs including nitrate (`NO`), ammonia (`NH`), soluble COD (`sCOD`),
total Kjeldahl nitrogen (`TKN`), volatile suspended solids (`VSS`), and
organic nitrogen (`ND`).

The code retains configurable ASM and non-ASM execution paths, but the
repository's principal scientific purpose is ASM1 surrogate modeling.

## Repository contents

- `ffnwrapper.*` and `ffnwrapper_multi.*`: single- and multi-output FFN
  wrappers;
- `config.*`: ASM/non-ASM data paths, target selection, and run configuration;
- `trainer.*`, `modelbuilder.*`, and `modelcreator.*`: training and model
  construction;
- `cmodelstructure*.cpp/.h`: network structure and lag configuration;
- `GADistribution.*`, `ga.*`, and `individual.h`: genetic-algorithm
  optimization tools; and
- `FFN_Wrapper.pro` and `FFN_Wrapper.sln`: Qt/qmake and Visual Studio
  projects.

## Building and data configuration

Use the qmake project or Visual Studio solution appropriate to the local
toolchain. Before building, replace the machine-specific paths in `main.cpp`
and confirm the ASM training/test files named for the selected target.

Preserve temporal train/test separation and apply the same scaling and lag
construction used during training when evaluating a saved model.

## Status

This is research software accompanying an ASM1 surrogate-modeling workflow.
Predictive performance must be evaluated on independent influent scenarios
before the emulator is used for real-time control or digital-twin applications.

## Associated publication

Shakouri, B., Murthy, S., De Clippeleir, H., Lesnik, K., & Massoudieh, A.
(2026). “Surrogate modeling of ASM1 using feedforward and LSTM neural networks.”
*Journal of Water Process Engineering, 81*, 109246.
[https://doi.org/10.1016/j.jwpe.2025.109246](https://doi.org/10.1016/j.jwpe.2025.109246)
· [Google Scholar record](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=0lTtuMoAAAAJ&citation_for_view=0lTtuMoAAAAJ:4JMBOYKVnBMC)
