# Control Barrier Function refinement with HJ Reachability

This repository contains the implementation of `refineCBF`, accompanying the paper [Refining Control Barrier Functions using HJ Reachability](https://arxiv.org/abs/2204.12507) by Sander Tonkens and Sylvia Herbert, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022.

This project combines CBFs and reachability, to refine learned or analytical CBFs and for making backup policy-based CBFs explicit.

In particular:

- The `refine_cbfs` directory contains code to define a tabular CBF (a CBF defined over a grid) and provides an interface with `hj_reachability` and `cbf_opt` to define its dynamics.
- The `examples` folder provides the simulation results for the paper mentioned above
- Don't forget to add this directory to your path to have working examples! `sys.path.append('DIR_LOC/refineCBF')`. 
- Install GUROBI
## Requirements

- `hj_reachability`: Toolbox for computing HJ reachability leveraging `jax`: `pip install --upgrade hj-reachability`.
Requires installing `jax` additionally based on available accelerator support. See [JAX installation instructions](https://github.com/google/jax#installation) for details.
- `cbf_opt`:  Toolbox for constructing CBFs and implementing them in a safety filter (using `cvxpy`). Run `pip install "cbf_opt>=0.6.0"` or install locally using the [Github link](https://github.com/stonkens/cbf_opt) and run `pip install -e .` in DIR to install.
- `experiment_wrapper`: Self-contained toolbox for running experiments that have analytically defined dynamics and measurement models. Run `pip install "experiment-wrapper>=1.1"` [Github link](https://github.com/stonkens/experiment_wrapper) and run `pip install -e .` in DIR to install
