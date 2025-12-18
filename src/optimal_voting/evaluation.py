import numpy as np
import data_utils as du
import voting_utils as vu
from OptimizableRule import _optimize_and_report_score
import pandas as pd


def evaluate_many_score_vectors_on_profiles(score_vectors, profiles, utilities, eval_func,
                                            aggregation_function=np.mean):
    """
    Return a dict of score score_vector mapped to the aggregation of utility_profile (or, whatever eval_func does) for each
    provided score score_vector across all pref_profiles.
    :param score_vectors:
    :param profiles:
    :param utilities:
    :param eval_func:
    :param aggregation_function:
    :return:
    """

    results = {}

    for vector_name, vector in score_vectors.items():
        mean_sw, _ = _optimize_and_report_score(profiles=profiles,
                                                utilities=utilities, eval_func=eval_func,
                                                profile_score_agg_metric=aggregation_function,
                                                m=len(profiles[0][0]),
                                                n_steps=0,
                                                initial_state=vector)

        results[vector_name] = round(mean_sw, 3)

    return results


def evaluate_score_vectors():
    """
    Save a Dataframe showing the performance of several scoring vectors across different preference distributions.
    This should include the (copied by hand for now) result of annealing on different utility functions.
    :return:
    """
    n_voters = 99
    n_candidates = 10
    profiles_per_dist = 50
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
        # du.ProfilesDescription("single_peaked_walsh",
        #                        num_profiles=profiles_per_dist,
        #                        num_voters=n,
        #                        num_candidates=m,
        #                        args=None),
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
    ]

    evaluation_functions = {
        "utilitarian": vu.utilitarian_social_welfare,
        "egalitarian": vu.egalitarian_social_welfare,
        "nash": vu.nash_social_welfare,
        "malfare": vu.malfare_social_welfare
    }

    vectors = vu.score_vector_examples(n_candidates)

    vectors["annealed-utilitarian"] = [1.0, 0.88889, 0.77778, 0.66667, 0.55556, 0.44444, 0.33333, 0.22222, 0.11111, 0.0]
    vectors["annealed-egalitarian"] = [1.0, 0.96029, 0.89068, 0.8355, 0.82421, 0.79528, 0.73561, 0.55706, 0.43915, 0.0]
    vectors["annealed-nash"] = [1.0, 1.0, 0.72372, 0.51168, 0.51123, 0.22465, 0.22465, 0.18508, 0.14311, 0.0]
    vectors["annealed-malfare"] = [1.0, 0.02248, 0.02248, 0.02247, 0.01785, 0.01785, 0.01699, 0.01682, 0.01651, 0.0]

    for util_name, util_func in evaluation_functions.items():

        results_for_single_util_function = []
        # make one row for each distribution
        cols = ["pref distribution"] + list(vectors.keys())
        all_rows = []
        for prof_desc in profiles_descriptions:

            # create new pref_profiles for each different pref distribution
            profiles = du.create_profiles(profiles_descriptions=[prof_desc])
            utilities = [du._utilities_from_profile(profile) for profile in profiles]

            # for each utility function, evaluate quality of each score score_vector
            # get dict mapping a score_vector name to a mean social welfare for corresponding score_vector
            results = evaluate_many_score_vectors_on_profiles(score_vectors=vectors,
                                                              profiles=profiles,
                                                              utilities=utilities,
                                                              eval_func=util_func)
            results_for_single_util_function.append(results)
            new_row = [prof_desc.distribution] + [results[col] for col in cols[1:]]
            all_rows.append(new_row)
        else:
            # Also test with each profile distribution together
            # create new pref_profiles for each different pref distribution
            profiles = du.create_profiles(profiles_descriptions=profiles_descriptions)
            utilities = [du._utilities_from_profile(profile) for profile in profiles]

            # for each utility function, evaluate quality of each score score_vector
            # get dict mapping a score_vector name to a mean social welfare for corresponding score_vector
            results = evaluate_many_score_vectors_on_profiles(score_vectors=vectors,
                                                              profiles=profiles,
                                                              utilities=utilities,
                                                              eval_func=util_func)
            results_for_single_util_function.append(results)
            new_row = ["mixed_preferences"] + [results[col] for col in cols[1:]]
            all_rows.append(new_row)

        # create a dataframe with results for all different preference distributions and a single utility function
        df = pd.DataFrame(data=all_rows, columns=cols)

        df.to_csv(f"evaluation_results-{util_name}.csv", index=False)


if __name__ == "__main__":
    evaluate_score_vectors()
