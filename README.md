# Optimal Voting Package

This package allows the application of standard optimization techniques to voting rule design. Existing approaches of using neural networks to develop optimized voting rules have been critiqued due to their lack of interpretability. This package allows optimizing _interpretable_ classes of voting rule such as positional scoring rules where simply looking at the score vector provides intuition about the rule itself.

The package aims to support a wide range of existing and user-specifiable optimization targets, as well as several classes of voting rule.

Possible optimization targets:
- common utility functions (current): 
  - Utilitarian
  - Nash
  - Egalitarian/Rawlsian
  - Malfare
- measures of distortion
- axiom violation rate (https://arxiv.org/abs/2508.06454)
- ranking consistency (https://arxiv.org/abs/2508.17177)

Optimizable rule types:
- positional scoring rules (current)
- probabilistic positional scoring rules (soon)
- functions of (weighted) tournaments, i.e., C2 rules (soon)
- sequential rules, i.e., Instant-Runoff Voting (soon)
- sequential Thiele rules (eventual)
- Thiele rules (perhaps)

Optimization techniques:
- simulated annealing: due to ease-of-use across domains this is intended to be the primary optimization method
- gradient descent: partially implemented at the moment. Early experiments show that this results in outcomes of a similar quality to simulated annealing but requires more compute. The eventual goal is to support GD with Torch and Jax but annealing is likely to remain preferable.


Proper documentation will be developed as the package matures. A rough overview of package use is:

1. Generate preference profile(s) empirically or by sampling one or more distributions.
2. (Optional) generate utilities corresponding to the preference profiles.
3. Select an optimization target (i.e., egalitarian social welfare).
4. Send profiles, utilities, target to Optimal-Voting.
5. Optimal-Voting returns a positional scoring vector which maximizes egalitarian social welfare on the provided profiles. 


NOTE: At the moment, the package is in active development. Changes that break compatibilty should be expected.

If you are interested in using the package or have suggestions for possible features you are encouraged to reach out at BenArmstrong dot ca