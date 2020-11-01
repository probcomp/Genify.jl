# Bayesian parameter estimation for an agent-based SIR model

An example of programmable inference for an existing Julia simulators. We adapt an [agent-based Susceptible-Infected-Recovered (SIR) model](https://juliadynamics.github.io/Agents.jl/stable/examples/sir/) of virus spread written in the [`Agents.jl`](https://github.com/JuliaDynamics/Agents.jl) [[1]](#1) framework. We transform this simulator using `genify`, then use it within a Gen model with parameter uncertainty and observation noise (see [`model.jl`](./model.jl)).

Our inference task is as follows: given infection and recovery counts in three connected cities with 50 agents each, as well as location data for 75% of the agents (e.g. from opt-in contact tracing), infer the infection rate β:

![Bayesian parameter estimation for an agent-based SIR model](./images/sir-inference.png)

## Instructions

To run this example, first activate the local environment via the package manager. Assuming Julia was started in the top-level package directory `Genify.jl`,  run:
```
; cd examples/sir
] activate .
] instantiate
```

Including the top-level script [`inference.jl`](inference.jl) will run all experiments:
```
include("./inference.jl")
```

## Algorithms

We implement several different inference algorithms in Gen to solve this task:
1. [Cascading resimulation Metropolis-Hastings](inference/resimulation_mh.jl) [[2]](#2)
2. [Single-site Metropolis-Hastings](inference/single_site_mh.jl)
3. [Block resimulation Metropolis-Hastings](inference/block_mh.jl) [[3]](#3)
4. [Basic SMC with proposals from the prior](inference/basic_smc.jl)
5. [SMC with parameter rejuvenation via Gaussian drift](inference/drift_smc.jl)
6. [Data-driven SMC with custom migration proposals](inference/data_driven_smc.jl)

## References

<a id="1">[1]</a> R. Vahdati, Ali (2019). Agents.jl: agent-based modeling framework in Julia. Journal of Open Source Software, 4(42), 1611, https://doi.org/10.21105/joss.01611

<a id="2">[2]</a> M. F. Cusumano-Towner, A. Radul, D. Wingate, and V. K. Mansinghka, “Probabilistic programs for inferring the goals of autonomous agents,” arXiv:1704.04977 [cs], Apr. 2017, http://arxiv.org/abs/1704.04977.


<a id="3">[3]</a> D. J. Sargent, J. S. Hodges, and B. P. Carlin, “Structured Markov Chain Monte Carlo,” Journal of Computational and Graphical Statistics, vol. 9, no. 2, p. 217, Jun. 2000, https://doi.org/10.2307/1390651.
