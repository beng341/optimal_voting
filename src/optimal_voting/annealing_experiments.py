import random

import data_utils as du
import voting_utils as vu
import numpy as np
import OptimizableRule as optr
from optimal_voting.OptimizableRule import PositionalScoringRule


def optimize_utilities(n_candidates=10, n_voters=99, profiles_per_dist=30, util_type="utilitarian",
                       rule_type="positional", **annealing_args):
    # Generate setting used by annealing process: evaluation function, pref_profiles, utility_profile
    if "seed" in annealing_args:
        seed = annealing_args["seed"]
    else:
        seed = None
    np.random.seed(seed)
    random.seed(seed)

    eval_func = vu.get_utility_eval_func_from_str(util_type)
    profiles = du.make_mixed_preference_profiles(profiles_per_distribution=profiles_per_dist, n=n_voters,
                                                 m=n_candidates)
    utilities = [du.utilities_from_profile(profile) for profile in profiles]

    if "initial_state" in annealing_args and annealing_args["initial_state"] is not None:
        initial_state = annealing_args["initial_state"]
    else:
        initial_state = [n_candidates - i - 1 for i in range(n_candidates)]
        initial_state = vu.normalize_score_vector(initial_state)

    if "profile_score_agg_metric" in annealing_args:
        profile_score_agg_metric = annealing_args["profile_score_agg_metric"]
    else:
        profile_score_agg_metric = np.mean

    if "job_name" in annealing_args:
        job_name = annealing_args["job_name"]
    else:
        job_name = du.default_job_name(**annealing_args)

    if "n_steps" in annealing_args:
        n_steps = annealing_args["n_steps"]
    else:
        n_steps = 0

    if "num_history_updates" in annealing_args:
        num_history_updates = annealing_args["num_history_updates"]
    else:
        num_history_updates = min(100, n_steps)

    if rule_type == "positional":
        rule = optr.PositionalScoringRule(eval_func=eval_func,
                                          m=n_candidates,
                                          k=None,
                                          initial_state=initial_state,
                                          utilities=utilities,
                                          profile_score_aggregation_metric=profile_score_agg_metric,
                                          keep_history=True,
                                          history_path="../results/annealing_history",
                                          job_name=job_name,
                                          num_history_updates=num_history_updates,
                                          pref_profile_lists=profiles
                                          )
    elif rule_type == "C2":
        rule = optr.C2ScoringRule(pref_profiles=profiles,
                                  eval_func=eval_func,
                                  m=n_candidates,
                                  k=None,
                                  initial_state=initial_state,
                                  utilities=utilities,
                                  profile_score_aggregation_metric=profile_score_agg_metric,
                                  keep_history=True,
                                  history_path="../results/annealing_history",
                                  job_name=job_name,
                                  num_history_updates=num_history_updates
                                  )

    result = rule.optimize(n_steps=n_steps)
    vector = result["state"]
    if rule.history is not None and n_steps > 0:
        print(f"Current Energy: {rule.history['current_energy'][-1]}")
        print(f"Best Energy: {rule.history['best_energy'][-1]}")

    if initial_state is None and n_steps == 0:
        vector = rule.state
    elif n_steps == 0:
        vector = initial_state
    mean_sw = rule.rule_score()
    return mean_sw, vector


if __name__ == "__main__":
    optimization_steps = 100
    # annealing_runs = 3
    n = 10
    m = 10
    profiles_per_dist = 50

    seed = 0

    util_type = "malfare"

    all_profiles = du.make_impartial_culture_profiles(n_profiles=profiles_per_dist,
                                                      n=n, m=m, seed=seed)
    all_utilities = [du.utilities_from_profile(profile, normalize_utilities=True, utility_type="uniform_random") for
                     profile in all_profiles]
    candidates = random.sample(range(100000), m)
    voters = random.sample(range(100000), m)
    all_utilities_dicts = [
        {
            v: {c: random.randint(0, 10000) for c in candidates}
            for v in voters
        }
        for _ in range(profiles_per_dist)   # one for each profile
    ]

    args = {
        "n_steps": optimization_steps,
        # "utility_profile_lists": all_utilities,
        # "pref_profile_lists": all_profiles,
        "utility_profile_dicts": all_utilities_dicts,
        "optimization_method": "annealing",
        # "initial_state": [1.0] + [0.0 for _ in range(m-1)],
        "initial_state": [(m-i-1)/(m-i) for i in range(m)],
        # "gd_opt_target": util_type,
        "return_candidate_scores": True,
        "verbose": True
    }
    annealing = PositionalScoringRule(
                                      eval_func=vu.get_utility_eval_func_from_str(util_type),
                                      m=m,
                                      **args
                                      )

    anneal_dict = annealing.optimize(n_steps=optimization_steps)
    print(f"Annealing vector is: {anneal_dict['state']} with loss {anneal_dict['best_energy']}")

    # args["optimization_method"] = "gradient_descent"
    # gradient_descender = PositionalScoringRule(
    #                                   eval_func=vu.get_utility_eval_func_from_str(util_type),
    #                                   m=m,
    #                                   **args
    #                                   )
    #
    # gd_dict = gradient_descender.optimize(n_steps=optimization_steps)
    # print(f"GD vector is: {gd_dict['state']} with loss {gd_dict['best_energy']}")

    anneal_sw, _ = optimize_utilities(n_candidates=m,
                                         n_voters=n,
                                         profiles_per_dist=profiles_per_dist,
                                         util_type=util_type,
                                         rule_type="positional",
                                         initial_state=anneal_dict['state'],
                                         profile_score_agg_metric=np.mean,
                                         # job_name=f"util_measurement-initial_state={vec_name}-utility={util_type}",
                                         n_steps=0,
                                         seed=seed
                                         )

    # gd_sw, _ = optimize_utilities(n_candidates=m,
    #                                      n_voters=n,
    #                                      profiles_per_dist=profiles_per_dist,
    #                                      util_type=util_type,
    #                                      rule_type="positional",
    #                                      initial_state=gd_dict['state'],
    #                                      profile_score_agg_metric=np.mean,
    #                                      # job_name=f"util_measurement-initial_state={vec_name}-utility={util_type}",
    #                                      n_steps=0,
    #                                      seed=seed
    #                                      )

    print("\n===================\n")

    print(f"Validation test on anneal vector gives {anneal_sw} utility.")
    # print(f"Validation test on gradient descent vector gives {gd_sw} utility.")
