import data_utils as du
import voting_utils as vu
import pandas as pd
import numpy as np
from OptimizableRule import _optimize_and_report_score
import OptimizableRule as optr


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


def generate_preference_profiles(profiles_per_distribution, n, m):
    profiles_descriptions = [
        du.ProfilesDescription("IC",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args=None),
        du.ProfilesDescription("single_peaked_conitzer",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args=None),
        du.ProfilesDescription("single_peaked_walsh",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args=None),
        du.ProfilesDescription("MALLOWS-RELPHI-R",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args=None),
        du.ProfilesDescription("URN-R",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args=None),
        du.ProfilesDescription("euclidean",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args={"num_dimensions": 3, "space": "uniform_sphere"}),
        du.ProfilesDescription("euclidean",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args={"num_dimensions": 10, "space": "uniform_sphere"}),
        du.ProfilesDescription("euclidean",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args={"num_dimensions": 3, "space": "uniform_cube"}),
        du.ProfilesDescription("euclidean",
                               num_profiles=profiles_per_distribution,
                               num_voters=n,
                               num_candidates=m,
                               args={"num_dimensions": 10, "space": "uniform_cube"}),
    ]

    profiles = du.create_profiles(profiles_descriptions=profiles_descriptions)

    return profiles


def optimize_utilities(n_candidates=10, n_voters=99, profiles_per_dist=30, util_type="utilitarian",
                       rule_type="positional", **annealing_args):
    # Generate setting used by annealing process: evaluation function, profiles, utilities
    eval_func = get_utility_eval_func_from_str(util_type)
    profiles = generate_preference_profiles(profiles_per_distribution=profiles_per_dist, n=n_voters, m=n_candidates)
    utilities = [du._utilities_from_profile(profile) for profile in profiles]

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
    # rule = OptimizableSequentialScoringRule(profiles,
    #                                         eval_func,
    #                                         m,
    #                                         utilities=utilities,
    #                                         initial_state=initial_state,
    #                                         profile_score_aggregation_metric=profile_score_agg_metric,
    #                                         changes_per_step=1,
    #                                         track_score=True,
    #                                         )
    if n_steps > 0:
        # {
        #     "state": vector,
        #     "best_energy": sw,
        #     "best_energy_history": self.best_energy_history,
        #     "current_energy_history": self.best_energy_history,
        # }

        result = rule.optimize(n_steps=n_steps)
        vector = result["state"]
        if rule.history is not None:
            print(f"Current Energy: {rule.history['current_energy'][-1]}")
            print(f"Best Energy: {rule.history['best_energy'][-1]}")
    if initial_state is None and n_steps == 0:
        vector = rule.state
    elif n_steps == 0:
        vector = initial_state
    mean_sw = rule.rule_score()
    return mean_sw, vector

    # Find utility on some hand-constructed score vectors as comparison
    # TODO: Move out of this method, use as a starting state for this method and set n_steps to zero.
    pre_built_vectors = vu.score_vector_examples(n_candidates)
    for vec_name, starting_state in pre_built_vectors.items():
        print(f"Beginning Sequential Rule for: {vec_name}")
        starting_state = vu.normalize_score_vector(starting_state)
        mean_sw, _ = _optimize_and_report_score(
            profiles=profiles,
            utilities=utilities,
            eval_func=eval_func,
            profile_score_agg_metric=np.mean,
            m=n_candidates,
            n_steps=0,
            initial_state=starting_state
        )

        best_results[vec_name] = (mean_sw, starting_state)

        # Use simulated annealing to find several scoring vectors that do well
        all_annealing_outcomes = []
        for _ in range(annealing_runs):
            mean_sw, vector = _optimize_and_report_score(
                profiles=profiles,
                utilities=utilities,
                eval_func=eval_func,
                profile_score_agg_metric=np.mean,
                m=n_candidates,
                n_steps=annealing_steps,
                initial_state=None
            )
            all_annealing_outcomes.append((mean_sw, vector))


def optimize_utilities_tmp(util_type="utilitarian", annealing_steps=500, annealing_runs=5, n_candidates=10, n_voters=99,
                           profiles_per_dist=30):
    pref_model_name = "mixture_of_distributions"
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

    file_suffix = f"{pref_model_name}-m={n_candidates}-n={n_voters}-steps={annealing_steps}-annealing_runs={annealing_runs}-testing"
    filename = f"results-{util_type}-{file_suffix}.csv"

    output_data = []

    profiles_descriptions = [
        du.ProfilesDescription("IC",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args=None),
        du.ProfilesDescription("single_peaked_conitzer",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args=None),
        du.ProfilesDescription("single_peaked_walsh",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args=None),
        du.ProfilesDescription("MALLOWS-RELPHI-R",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args=None),
        du.ProfilesDescription("URN-R",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args=None),
        du.ProfilesDescription("euclidean",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args={"num_dimensions": 3, "space": "uniform_sphere"}),
        du.ProfilesDescription("euclidean",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args={"num_dimensions": 10, "space": "uniform_sphere"}),
        du.ProfilesDescription("euclidean",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args={"num_dimensions": 3, "space": "uniform_cube"}),
        du.ProfilesDescription("euclidean",
                               num_profiles=profiles_per_dist,
                               num_voters=n_voters,
                               num_candidates=n_candidates,
                               args={"num_dimensions": 10, "space": "uniform_cube"}),
    ]

    profiles = du.create_profiles(profiles_descriptions=profiles_descriptions)
    utilities = [du._utilities_from_profile(profile) for profile in profiles]

    best_results = {}

    # Calculate social welfare for each pre-built vector
    pre_built_vectors = vu.score_vector_examples(n_candidates)
    for vec_name, starting_state in pre_built_vectors.items():
        print(f"Beginning Sequential Rule for: {vec_name}")
        starting_state = vu.normalize_score_vector(starting_state)
        mean_sw, _ = _optimize_and_report_score(
            profiles=profiles,
            utilities=utilities,
            eval_func=eval_func,
            profile_score_agg_metric=np.mean,
            m=n_candidates,
            n_steps=0,
            initial_state=starting_state
        )

        best_results[vec_name] = (mean_sw, starting_state)

    # Use simulated annealing to find several scoring vectors that do well
    all_annealing_outcomes = []
    for _ in range(annealing_runs):
        mean_sw, vector = _optimize_and_report_score(
            profiles=profiles,
            utilities=utilities,
            eval_func=eval_func,
            profile_score_agg_metric=np.mean,
            m=n_candidates,
            n_steps=annealing_steps,
            initial_state=None
        )
        all_annealing_outcomes.append((mean_sw, vector))

    # sort all annealing results so we can report average of their results/track best result
    all_annealing_outcomes.sort(key=lambda x: x[0], reverse=True)
    all_annealing_vectors = [np.round(v[1], 5).tolist() for v in all_annealing_outcomes]

    # Find the annealing vector with the highest sw
    best_annealing_vector_mean = max(all_annealing_outcomes, key=lambda x: x[0])
    best_results["annealing_best"] = best_annealing_vector_mean

    # calculate utility for the mean of all annealing outcomes
    mean_annealing_vector = np.mean([v for v in all_annealing_vectors], axis=0)
    mean_sw, vector = _optimize_and_report_score(
        profiles=profiles,
        utilities=utilities,
        eval_func=eval_func,
        profile_score_agg_metric=np.mean,
        m=n_candidates,
        n_steps=0,
        initial_state=mean_annealing_vector
    )
    best_results["annealing_mean"] = (mean_sw, vector)

    # calculate utility for the median of all annealing outcomes
    median_sw, vector = _optimize_and_report_score(
        profiles=profiles,
        utilities=utilities,
        eval_func=eval_func,
        profile_score_agg_metric=np.mean,
        m=n_candidates,
        n_steps=0,
        initial_state=mean_annealing_vector
    )
    best_results["annealing_median"] = (median_sw, vector)

    for rule_name, (sw_mean, vector) in best_results.items():
        row = {
            "pref_model": pref_model_name,
            "voters_per_profile": n_voters,
            "num_profiles": len(profiles),
            "vector_name": rule_name,
            "best_mean_sw": sw_mean,
            "best_mean_sw_vector": np.round(vector, 5).tolist(),
            "all_vectors": all_annealing_vectors if rule_name == "annealing_best" else "",
        }
        output_data.append(row)

    # Create DataFrame of all rows and save to a file called "output.csv"
    df = pd.DataFrame(output_data)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    annealing_steps = 1000
    # annealing_runs = 3
    n_voters = 49
    n_candidates = 10
    profiles_per_dist = 20

    util_type = "egalitarian"

    pre_built_vectors = vu.score_vector_examples(n_candidates)
    # for vec_name, starting_state in pre_built_vectors.items():
    #     print(f"Beginning PSR Rule for: {vec_name}")
    #     starting_state = vu.normalize_score_vector(starting_state)
    #
    #     optimize_utilities(n_candidates=n_candidates,
    #                        n_voters=n_voters,
    #                        profiles_per_dist=profiles_per_dist,
    #                        util_type=util_type,
    #                        rule_type="positional",
    #                        initial_state=starting_state,
    #                        profile_score_agg_metric=np.mean,
    #                        job_name=f"util_measurement-initial_state={vec_name}-utility={util_type}",
    #                        n_steps=0
    #                        )

    # actually do optimization
    print(f"Beginning PSR Optimization with Annealing")
    mean_sw, vector = optimize_utilities(n_candidates=n_candidates,
                       n_voters=n_voters,
                       profiles_per_dist=profiles_per_dist,
                       util_type=util_type,
                       rule_type="positional",
                       initial_state=pre_built_vectors["half_approval_degrading"],
                       profile_score_agg_metric=np.mean,
                       # job_name=f"util_measurement-annealing-utility={util_type}",
                       n_steps=annealing_steps,
                       num_history_updates=10,
                       verbose=True
                       )

    print(f"Best vector is: {vector}")



    # util_type = "nash"
    # pre_built_vectors = vu.score_vector_examples(n_candidates)
    # for vec_name, starting_state in pre_built_vectors.items():
    #     print(f"Beginning Sequential Rule for: {vec_name}")
    #     starting_state = vu.normalize_score_vector(starting_state)
    #
    #     optimize_utilities(n_candidates=n_candidates,
    #                        n_voters=n_voters,
    #                        profiles_per_dist=profiles_per_dist,
    #                        util_type=util_type,
    #                        rule_type="positional",
    #                        initial_state=starting_state,
    #                        profile_score_agg_metric=np.mean,
    #                        job_name=f"util_measurement-initial_state={vec_name}-utility={util_type}",
    #                        n_steps=0
    #                        )
    #
    # # actually do optimization
    # optimize_utilities(n_candidates=n_candidates,
    #                    n_voters=n_voters,
    #                    profiles_per_dist=profiles_per_dist,
    #                    util_type=util_type,
    #                    rule_type="positional",
    #                    # initial_state=pre_built_vectors["half_approval_degrading"],
    #                    profile_score_agg_metric=np.mean,
    #                    job_name=f"util_measurement-annealing-utility={util_type}",
    #                    n_steps=annealing_steps,
    #                    num_history_updates=69
    #                    )
