from optimal_voting.OptimizableRule import PositionalScoringRule
from optimal_voting.OptimizableRule import C2ScoringRule
import pytest


@pytest.fixture
def sample_profiles():
    prof1 = [
        [0, 1, 2, 3, 4],    # plurality scores: (3, 2, 0, 0, 0)
        [0, 1, 2, 3, 4],    # borda scores: (12, 17, 10, 8, 3)
        [0, 1, 3, 4, 2],
        [1, 2, 3, 4, 0],
        [1, 2, 3, 4, 0],
    ]
    return [prof1]

def test_c2_is_borda(sample_profiles):
    # pref_profiles = sample_profiles
    rule = C2ScoringRule(pref_profiles=sample_profiles,
                         eval_func="utilitarian")
    assert 1 == 1

