# MaxInfoRL: Boosting exploration in RL through information gain maximization

A reimplementation of [MaxInfoRL][paper], a scalable and general reinforcement
learning algorithm that masters a wide range of applications with fixed
hyperparameters.



If you find this code useful, please reference in your paper:

```
@article{sukhija2024maxinforl,
  title={MaxInfoRL: Boosting exploration in reinforcement learning through information gain maximization},
  author={Sukhija, Bhavya and Coros, Stelian and Krause, Andreas and Abbeel, Pieter and Sferrazza, Carmelo},
  journal={ArXiv},
  year={2024}
}
```

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## MaxInfoRL

MaxInfoRL boosts exploration in RL by combining extrinsic rewards with intrinsic 
exploration bonuses derived from information gain of the underlying MDP.
MaxInfoRL naturally trades off maximization of the value function with that of the entropy over states, rewards,
and actions. MaxInfoRL is very general and can be combined with a variety
of off-policy model-free RL methods for continuous state-action spaces. We provide implementations of 
**MaxInfoSac, MaxInfoREDQ, MaxInfoDrQ, MaxInfoDrQv2**.



# Instructions

## Installation

```sh
pip install .
```

### Remark: 
The above command does not install the GPU version of [JAX][jax]. Please manually install the GPU version if needed.



Training script:

1. State-based

```sh
python examples/state_based/experiment.py \
  --project_name maxinforl \
  --entity_name wandb_entity_name \
  --alg_name maxinfosac \
  --env_name cartpole-swingup_sparse \
  --wandb_log 1
```

2. Vision-based

```sh
python examples/vision_based/experiment.py \
  --project_name maxinforl \
  --entity_name wandb_entity_name \
  --alg_name maxinfodrq \
  --env_name cartpole-swingup_sparse \
  --wandb_log 1
```

All hyperparameters are listed in the `examples/state_based//configs.yaml` and `examples/vision_based//configs.yaml` 
files. You can override them if needed.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://openreview.net/pdf?id=R4q3cY3kQf
[website]: https://sukhijab.github.io/
[tweet]: https://sukhijab.github.io/
