# Optimal Voting Package

This package allows the application of standard optimization techniques to voting rule design. Existing approaches of using neural networks to develop optimized voting rules have been critiqued due to their lack of interpretability. This package allows optimizing _interpretable_ classes of voting rule such as positional scoring rules where simply looking at the score vector provides intuition about the rule itself.

The package supports a range of existing optimization targets or allows users to provide their own custom optimization target. At the moment the package is focused on positional scoring rules, however support for additional rule types is planned for the future.

### Package Options
#### Optimization Targets:
- common social welfare functions: 
  - Utilitarian
  - Nash
  - Egalitarian/Rawlsian
  - Malfare
- two empirical measures of distortion

#### Optimizable rule types:
- positional scoring rules
- probabilistic positional scoring rules

In the future the library may support additional types of voting rule, such as:
- functions of (weighted) tournaments, i.e., C2 rules (soon)
- sequential rules, e.g., Instant-Runoff Voting (soon)
- sequential Thiele rules (eventual)
- Thiele rules (perhaps)

### Optimization techniques:
- simulated annealing: due to ease-of-use across domains this is intended to be the primary optimization method
- gradient descent: partially implemented at the moment. Early experiments show that this results in outcomes of a similar quality to simulated annealing but requires more compute. The eventual goal is to support GD with Torch and Jax but annealing is likely to remain preferable.

### Use cases:

The library can be used for a wide variety of tasks. The included social welfare functions may help users select alternatives which better reflect their preferred social welfare, or might be useful for understanding social welfare functions analytically. Additionally, user-specified optimization targets allow for novel use cases, such as:

- finding rules which minimize axiom violation rate (https://arxiv.org/abs/2508.06454)
- finding rules which produce rankings that are most consistent with each other (https://arxiv.org/abs/2508.17177)


The expected most common use case for the package is roughly as follows:

1. Generate preference profile(s) empirically or by sampling one or more distributions.
2. (Optional) generate utilities corresponding to the preference profiles.
3. Select an optimization target (i.e., egalitarian social welfare).
4. Send profiles, utilities, target to the library.
5. Optimal-Voting returns a positional scoring vector which maximizes egalitarian social welfare on the provided profiles. 

### Usage Examples

See `example_usage.py` for several examples outlining the main use cases of the package. These examples show how to:

1. Generate sample preferences and utilities.
2. Find an optimized score vector based on either ordinal or cardinal preferences.
3. Compare several score vectors without optimization.
4. Define a custom optimization target and find a rule which maximizes social welfare on the target.



If you are interested in using the package or have suggestions for possible features you are encouraged to reach out to 'research at BenArmstrong dot ca'!