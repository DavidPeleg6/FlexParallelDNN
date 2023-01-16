# AutoSplit

## Description
Will be linked to the EGRL AutoML service for optimization of the parallelization strategies.
Currently, the framework supports any data parallel configuration but only model parallelism in 
This library extends basic PyTorch capabilities by adding support for per-layer parallel strategies, similar to FlexFlow. 
Usability is guaranteed by adding minimal code to an existing distributed configuration (such as DDP or FSDP by fairscale).
A simple wrapper is applied to an existing PyTorch module in order to it with a parallelization strategy that could either be auto-generated or chosen by the user.

## Installation

build directly from source:

pip install .

To install FairScale, please see the following [instructions](https://github.com/facebookresearch/fairscale/blob/main/docs/source/installation_instructions.rst). You should be able to install a pip package or

## Getting Started
initializing model and data parallelism:

setting up a distributed data sampler:

gathering targets in order to feed to the loss:

finding optimal strategy:

wrapping model with optimal strategy:

once again gathering targets before each call to loss:

adding a call to notify the optimizer that gradients should be synced right after backwards call:

## Examples

Here are a few sample snippets from a subset of FairScale offerings:


At a high level, we want ML researchers to:
  * go parallel more easily (i.e. no need to manually split layers when implementing the Network)
  * potentially higher GPU efficiency (fewer steps, less networking overhead, etc.)
