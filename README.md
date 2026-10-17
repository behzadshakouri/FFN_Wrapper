# FFN_Wrapper

## About

FFN_Wrapper is a C++ research wrapper for building, training, and evaluating
feed-forward neural-network surrogate models. It includes single- and
multi-model structures, configurable training workflows, genetic-algorithm
utilities, and project files for Qt/qmake and Microsoft Visual Studio.

## Repository contents

- `ffnwrapper.*` and `ffnwrapper_multi.*`: neural-network wrappers;
- `trainer.*`, `modelbuilder.*`, and `modelcreator.*`: training and model
  construction;
- `cmodelstructure*.cpp/.h`: single- and multi-model structures;
- `GADistribution.*`, `ga.*`, and `individual.h`: optimization tools; and
- `FFN_Wrapper.pro` and `FFN_Wrapper.sln`: Qt and Visual Studio projects.

## Building

Use either the qmake project or Visual Studio solution appropriate to the local
toolchain. Review the configuration, data paths, scaling, train/test separation,
and random seeds before running an experiment.

## Status

This is research software. Predictive performance must be evaluated on
independent data before results are used outside the original experiments.
