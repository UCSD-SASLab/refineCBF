# Control Barrier Function refinement with HJ Reachability
Project combining CBFs and reachability, for refining learned or analytical CBFs and for making backup policies explicit.


## Requirements:
- `hj_reachability`: Toolbox for computing HJ reachability leveraging `jax`: `pip install --upgrade hj-reachability`. 
Requires installing `jax` additionally based on available accelerator support. See [JAX installation instructions](https://github.com/google/jax#installation) for details.
- `cbf_opt`:  Toolbox for constructing CBFs and implementing them in a safety filter (using `cvxpy`). [Github link](https://github.com/stonkens/cbf_opt) and run `pip install -e .` in DIR to install.


