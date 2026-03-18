import math
import numpy as np
from optimal_voting.OptimizableRule import PositionalScoringRule
from optimal_voting.OptimizableRule import C2ScoringRule
import pref_voting.margin_based_methods as c2
import pref_voting.c1_methods as c1
import pref_voting.scoring_methods as pos
import pref_voting.profiles as pv_prof
import prefsampling as ps


def find_rule_disagreements(pv_rule, opt_rule, profiles):
    """
    Find the winner of both given rules on the provided profiles. Return data informing about whether the rules are
    different from each other.
    :param pv_rule:
    :param opt_rule:
    :param profiles:
    :return:
    """

    disagreements = {"profile_indices": [], "pref_voting_winners": [], "opt_voting_winners": []}
    opt_winners_all = opt_rule.rule_winners()

    for idx, profile in enumerate(profiles):
        pv_profile = pv_prof.Profile(profile)
        pv_winners = set(pv_rule(pv_profile))
        opt_winners = set(opt_winners_all[idx])

        if pv_winners != opt_winners:
            disagreements["profile_indices"].append(idx)
            disagreements["pref_voting_winners"].append(pv_winners)
            disagreements["opt_voting_winners"].append(opt_winners)

    print(f"On {len(profiles)} there were {len(disagreements['profile_indices'])} disagreements.")

    return disagreements


def compare_rules():
    """

    :return:
    """
    n = 10
    m = 10
    k = 1000
    profiles = [ps.ordinal.impartial(num_voters=n, num_candidates=m) for _ in range(k)]

    # arguments for different rules:
    # opt_state = [1, 0, 0, 0]  # Borda
    opt_state = [1, -math.inf, 0, 0]  # Minimax
    # opt_state = [0, 0, 0.5, 0]    # Copeland

    opt_rule = C2ScoringRule(pref_profiles=profiles, eval_func="utilitarian", initial_state=opt_state)
    pv_rule = c2.minimax

    disagreements = find_rule_disagreements(pv_rule=pv_rule, opt_rule=opt_rule, profiles=profiles)


if __name__ == "__main__":
    compare_rules()
