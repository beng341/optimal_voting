import random

import data_utils as du
import voting_utils as vu
import pandas as pd
import numpy as np
from OptimizableRule import _optimize_and_report_score
import OptimizableRule as optr
from optimal_voting.OptimizableRule import OptimizableRule, PositionalScoringRule


def get_utility_eval_func_from_str(util_type):
    if util_type == "utilitarian":
        eval_func = vu.utilitarian_social_welfare
    elif util_type == "egalitarian":
        eval_func = vu.egalitarian_social_welfare
    elif util_type == "nash":
        eval_func = vu.nash_social_welfare
    elif util_type == "malfare":
        eval_func = vu.malfare_social_welfare
    elif util_type == "utilitarian_distortion":
        eval_func = vu.utilitarian_distortion
    elif util_type == "egalitarian_distortion":
        eval_func = vu.egalitarian_distortion
    else:
        raise ValueError("Didn't make other eval functions yet")
    return eval_func


def optimize_utilities(n_candidates=10, n_voters=99, profiles_per_dist=30, util_type="utilitarian",
                       rule_type="positional", **annealing_args):
    # Generate setting used by annealing process: evaluation function, profiles, utilities
    if "seed" in annealing_args:
        seed = annealing_args["seed"]
    else:
        seed = None
    np.random.seed(seed)
    random.seed(seed)

    eval_func = get_utility_eval_func_from_str(util_type)
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
        rule = optr.PositionalScoringRule(profiles,
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
    elif rule_type == "C2":
        rule = optr.C2ScoringRule(profiles=profiles,
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
    optimization_steps = 200
    # annealing_runs = 3
    n = 10
    m = 10
    profiles_per_dist = 50

    seed = 0

    util_type = "malfare"

    all_profiles = vu.make_mixed_preference_profiles(profiles_per_distribution=profiles_per_dist,
                                                     n=n,
                                                     m=m)
    all_utilities = [vu.utilities_from_profile(profile, normalize_utilities=True, utility_type="uniform_random") for
                     profile in all_profiles]

    args = {
        "n_steps": optimization_steps,
        "utilities": all_utilities,
        "optimization_method": "annealing",
        # "initial_state": [1.0] + [0.0 for _ in range(m-1)],
        "initial_state": [(m-i-1)/(m-i) for i in range(m)],
        "gd_opt_target": util_type
    }
    annealing = PositionalScoringRule(profiles=all_profiles,
                                      eval_func=get_utility_eval_func_from_str(util_type),
                                      m=m,
                                      **args
                                      )

    anneal_dict = annealing.optimize(n_steps=optimization_steps)
    print(f"Annealing vector is: {anneal_dict['state']} with loss {anneal_dict['best_energy']}")

    args["optimization_method"] = "gradient_descent"
    gradient_descender = PositionalScoringRule(profiles=all_profiles,
                                               eval_func=get_utility_eval_func_from_str(util_type),
                                               m=m,
                                               **args
                                               )

    gd_dict = gradient_descender.optimize(n_steps=optimization_steps)
    print(f"GD vector is: {gd_dict['state']} with loss {gd_dict['best_energy']}")

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

    gd_sw, _ = optimize_utilities(n_candidates=m,
                                         n_voters=n,
                                         profiles_per_dist=profiles_per_dist,
                                         util_type=util_type,
                                         rule_type="positional",
                                         initial_state=gd_dict['state'],
                                         profile_score_agg_metric=np.mean,
                                         # job_name=f"util_measurement-initial_state={vec_name}-utility={util_type}",
                                         n_steps=0,
                                         seed=seed
                                         )

    print("\n===================\n")

    print(f"Validation test on anneal vector gives {anneal_sw} utility.")
    print(f"Validation test on gradient descent vector gives {gd_sw} utility.")



    # pre_built_vectors = vu.score_vector_examples(n_candidates)
    # for vec_name, starting_state in pre_built_vectors.items():
    #     # print(f"Beginning PSR Rule for: {vec_name}")
    #     starting_state = vu.normalize_score_vector(starting_state)
    #
    #     mean_sw, vector = optimize_utilities(n_candidates=n_candidates,
    #                                          n_voters=n_voters,
    #                                          profiles_per_dist=profiles_per_dist,
    #                                          util_type=util_type,
    #                                          rule_type="positional",
    #                                          initial_state=starting_state,
    #                                          profile_score_agg_metric=np.mean,
    #                                          job_name=f"util_measurement-initial_state={vec_name}-utility={util_type}",
    #                                          n_steps=0,
    #                                          seed=seed
    #                                          )
    #     print(f"{vec_name} - energy: {mean_sw}")

    # # print(f"Testing C2 as Borda")
    # mean_sw, vector = optimize_utilities(n_candidates=n_candidates,
    #                                      n_voters=n_voters,
    #                                      profiles_per_dist=profiles_per_dist,
    #                                      util_type=util_type,
    #                                      rule_type="C2",
    #                                      initial_state=[1, 0],
    #                                      profile_score_agg_metric=np.mean,
    #                                      job_name=f"util_measurement-annealing-utility={util_type}",
    #                                      n_steps=0,
    #                                      num_history_updates=50,
    #                                      verbose=True,
    #                                      seed=seed
    #                                      )
    # print(f"Borda C2 - energy: {mean_sw}")
    #
    # # print(f"Testing C2 as Copeland")
    # mean_sw, vector = optimize_utilities(n_candidates=n_candidates,
    #                                      n_voters=n_voters,
    #                                      profiles_per_dist=profiles_per_dist,
    #                                      util_type=util_type,
    #                                      rule_type="C2",
    #                                      initial_state=[0, 0.5],
    #                                      profile_score_agg_metric=np.mean,
    #                                      job_name=f"util_measurement-annealing-utility={util_type}",
    #                                      n_steps=0,
    #                                      num_history_updates=50,
    #                                      verbose=True,
    #                                      seed=seed
    #                                      )
    # print(f"Copeland C2 - energy: {mean_sw}")
    #
    # # print(f"Testing C2 between Borda and Copeland")
    # mean_sw, vector = optimize_utilities(n_candidates=n_candidates,
    #                                      n_voters=n_voters,
    #                                      profiles_per_dist=profiles_per_dist,
    #                                      util_type=util_type,
    #                                      rule_type="C2",
    #                                      initial_state=[0.5, 0.5],
    #                                      profile_score_agg_metric=np.mean,
    #                                      job_name=f"util_measurement-annealing-utility={util_type}",
    #                                      n_steps=0,
    #                                      num_history_updates=50,
    #                                      verbose=True,
    #                                      seed=seed
    #                                      )
    # print(f"Weirdo C2 - energy: {mean_sw}")
    #
    # # actually do optimization
    # print(f"Beginning C2 Optimization with Annealing")
    # mean_sw, vector = optimize_utilities(n_candidates=n_candidates,
    #                                      n_voters=n_voters,
    #                                      profiles_per_dist=profiles_per_dist,
    #                                      util_type=util_type,
    #                                      rule_type="C2",
    #                                      initial_state=[1, 0],
    #                                      profile_score_agg_metric=np.mean,
    #                                      job_name=f"util_measurement-annealing-utility={util_type}",
    #                                      n_steps=optimization_steps,
    #                                      num_history_updates=50,
    #                                      verbose=True,
    #                                      seed=seed
    #                                      )
    #
    # print(f"Best vector is: {vector}")
