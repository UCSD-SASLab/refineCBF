# Refining Control Barrier Functions through Hamilton Jacobi Reachability
This repository accompanies the paper "[Refining Control Barrier Functions through Hamilton Jacobi Reachability](https://arxiv.org/abs/2204.12507) by Sander Tonkens and Sylvia Herbert.

We provide an implementation of refining candidate (i.e. approximately correct CBFs) using HJ reachability in `python`. We provide a broad range of examples to which this method can be applied.

The README will be updated to contain more information at a later date.


## Requirements:
- `hj_reachability`: Toolbox for computing HJ reachability leveraging `jax`: `pip install --upgrade hj-reachability`. 
Requires installing `jax` additionally based on available accelerator support. See [JAX installation instructions](https://github.com/google/jax#installation) for details.
- `cbf_opt`:  Toolbox for constructing CBFs and implementing them in a safety filter (using `cvxpy`). [Github link](https://github.com/stonkens/cbf_opt) and run `pip install -e .` in DIR to install.
- `experiment_wrapper`: Toolbox for running simulations on self defined dynamics and pretty logging. [Github link](https://github.com/stonkens/experiment_wrapper)



