# Optimal Voting Package

This package allows the application of standard optimization techniques to voting rule design.

Initially, the package uses simulated annealing to find optimal positional scoring rules.
The package includes functionality for:
- generating underlying voter preferences
- converting preferences to utility values
- common utility functions (utilitarian, Nash, egalitarian/Rawlsian, malfare)
- custom scoring functions to allow generating rules optimized for novel targets

NOTE: At the moment, the package is quite basic. Expect changes that break compatibility. If you encounter this and want to use the package in your work you are welcome to email Ben Armstrong and inquire about how best to do so. There are also significant updates planned: different rule types, different optimization methods (especially gradient descent), and more documentation.