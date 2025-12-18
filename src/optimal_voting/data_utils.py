import copy
import math
import os.path
import random
import numpy as np
import pandas as pd
import pref_voting.profiles
import prefsampling as ps
from scipy.stats import gamma
from pref_voting.generate_profiles import generate_profile as gen_prof
from collections import namedtuple

ProfilesDescription = namedtuple("ProfilesDescription",
                                 [
                                     "distribution",
                                     "num_profiles",
                                     "num_voters",
                                     "num_candidates",
                                     "args"
                                 ]
                                 )


def create_profiles(profiles_descriptions, seed=None):
    """
    Given a description of the desired pref_profiles create a list where each entry contains a single profile.
    :param profiles_descriptions: list of ProfilesDescription namedtuple containing all the parameters required
    to generate each sw_type of profile.
    :param seed: Value for random number generator. Due to how we interface with prefsampling this is actually used as
    a seed for rng to generate an actual seed within this method, rather than being passed directly to prefsampling.
    :return: list of pref_profiles (each of which is a list of lists of integers)
    """

    profiles = []
    rng = random.Random(seed)  # passing seed directly appears to result in all pref_profiles always being identical
    for prof in profiles_descriptions:
        for _ in range(prof.num_profiles):
            if prof.args is None:
                args = {}
            else:
                args = prof.args
            args["seed"] = rng.randint(0, 1000000)
            profile = gen_prof(num_voters=prof.num_voters,
                               num_candidates=prof.num_candidates,
                               probmodel=prof.distribution,
                               num_profiles=prof.num_profiles,
                               **args)
            # rankings = profile.rankings
            profiles += [prof.rankings for prof in profile]
            # pref_profiles.append(profile.rankings)

    return profiles


def preference_distribution_options():
    """
    Construct a dictionary mapping a string name for each available preference distribution to a dict containing two
    keys: 'function' and 'args'. 'function' contains the actual function used to generate preferences from this
    distribution. All preference distributions take in parameters 'n_profiles', 'n' (number of voters), 'm' (number
    of candidates), and 'seed' (default value of None). Some distributions accept additional arguments but always
    have default values for any additional arguments. 'args' is a list containing the string names of each optional
    additional argument allowed by a distribution.
    :return: a dictionary exposing the available options for preference distribution generation.
    """

    dists = {
        "Impartial Culture": {
            "function": make_impartial_culture_profiles,
            "args": []
        },
        "Impartial Anonymous Culture": {
            "function": make_impartial_anonymous_culture_profiles,
            "args": []
        },
        "Single-Peaked (Walsh)": {
            "function": make_sp_walsh_profiles,
            "args": []
        },
        "Single-Peaked (Conitzer)": {
            "function": make_sp_conitzer_profiles,
            "args": []
        },
        "Single-Peaked (Circle)": {
            "function": make_sp_circle_profiles,
            "args": []
        },
        "Urn": {
            "function": make_urn_profiles,
            "args": ['alpha']
        },
        "Mallow's": {
            "function": make_mallows_profiles,
            "args": ['phi']
        },
    }
    return dists


def make_impartial_culture_profiles(n_profiles, n=10, m=10, seed=None):
    rng = random.Random(seed)
    profiles = [
        ps.ordinal.impartial(num_voters=n, num_candidates=m, seed=rng.randint(0, 100000))
        for _ in range(n_profiles)
    ]
    return profiles


def make_impartial_anonymous_culture_profiles(n_profiles, n=10, m=10, seed=None):
    rng = random.Random(seed)
    profiles = [
        ps.ordinal.impartial_anonymous(num_voters=n, num_candidates=m, seed=rng.randint(0, 100000))
        for _ in range(n_profiles)
    ]
    return profiles


def make_sp_walsh_profiles(n_profiles, n=10, m=10, seed=None):
    rng = random.Random(seed)
    profiles = [
        ps.ordinal.single_peaked_walsh(num_voters=n, num_candidates=m, seed=rng.randint(0, 100000))
        for _ in range(n_profiles)
    ]
    return profiles


def make_sp_conitzer_profiles(n_profiles, n=10, m=10, seed=None):
    rng = random.Random(seed)
    profiles = [
        ps.ordinal.single_peaked_conitzer(num_voters=n, num_candidates=m, seed=rng.randint(0, 100000))
        for _ in range(n_profiles)
    ]
    return profiles


def make_sp_circle_profiles(n_profiles, n=10, m=10, seed=None):
    rng = random.Random(seed)
    profiles = [
        ps.ordinal.single_peaked_circle(num_voters=n, num_candidates=m, seed=rng.randint(0, 100000))
        for _ in range(n_profiles)
    ]
    return profiles


def make_urn_profiles(n_profiles, n=10, m=10, alpha=None, seed=None):
    """

    :param n_profiles:
    :param n:
    :param m:
    :param alpha: After sampling each individual order, return alpha*m! copies of that order into the urn. A value
    of 0 corresponds to impartial culture, large values (near infinity) approach identity preferences. Default
    value is a random amount aiming for a middle ground, which is resampled for each distinct profile.
    :param seed:
    :return:
    """
    rng = random.Random(seed)
    profiles = [
        ps.ordinal.urn(num_voters=n, num_candidates=m,
                       # alpha=alpha if alpha is not None else round(math.factorial(m) * gamma.rvs(0.8, random_state=rng)),
                       alpha=alpha if alpha is not None else gamma.rvs(0.8, random_state=rng),
                       seed=rng.randint(0, 100000))
        for _ in range(n_profiles)
    ]
    return profiles


def make_mallows_profiles(n_profiles, n=10, m=10, phi=None, seed=None):
    """

    :param n_profiles:
    :param n:
    :param m:
    :param phi: Mallow's phi parameter. Acceptable values range from 0 to 1 (inclusive). A value of 0 corresponds to
    identity preferences and a value of 1 corresponds to impartial culture preferences.
    :param seed:
    :return:
    """
    rng = random.Random(seed)
    profiles = [
        ps.ordinal.mallows(num_voters=n, num_candidates=m,
                           phi=phi if phi is not None else rng.uniform(0.001, 0.999),
                           normalise_phi=False,  # disallowed for simplicity
                           impartial_central_vote=False,
                           seed=rng.randint(0, 100000))
        for _ in range(n_profiles)
    ]
    return profiles


def make_mixed_preference_profiles(profiles_per_distribution=100, n=10, m=10, seed=None):
    profiles_descriptions = [
        ProfilesDescription("IC",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args=None),
        ProfilesDescription("single_peaked_conitzer",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args=None),
        ProfilesDescription("single_peaked_walsh",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args=None),
        ProfilesDescription("MALLOWS-RELPHI-R",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args=None),
        ProfilesDescription("URN-R",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args=None),
        ProfilesDescription("euclidean",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args={"num_dimensions": 3, "space": "uniform_sphere"}),
        ProfilesDescription("euclidean",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args={"num_dimensions": 10, "space": "uniform_sphere"}),
        ProfilesDescription("euclidean",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args={"num_dimensions": 3, "space": "uniform_cube"}),
        ProfilesDescription("euclidean",
                            num_profiles=profiles_per_distribution,
                            num_voters=n,
                            num_candidates=m,
                            args={"num_dimensions": 10, "space": "uniform_cube"}),
    ]

    profiles = create_profiles(profiles_descriptions=profiles_descriptions, seed=seed)

    return profiles


# def _get_preference_models_and_args(preference_model="all", n_profiles=20, num_profiles=10, prefs_per_profile=50, m=5):
#     if preference_model == "all":
#         preference_model = [
#             "Impartial Culture",
#             "SP by Conitzer",
#             "SP by Walsh",
#             "Single-Crossing",
#             "1D Uniform",
#             "2D Uniform",
#             "3D Uniform",
#             "5D Uniform",
#             "10D Uniform",
#             "20D Uniform",
#             "2D Sphere",
#             "3D Sphere",
#             "5D Sphere",
#             "Urn",
#             "Norm-Mallows",
#         ]
#     preference_model_short_names = {
#         "Impartial Culture": "IC",
#         "SP by Conitzer": "single_peaked_conitzer",
#         "SP by Walsh": "single_peaked_walsh",
#         "Single-Crossing": "single_crossing",
#         "1D Uniform": "euclidean",
#         "2D Uniform": "euclidean",
#         "3D Uniform": "euclidean",
#         "5D Uniform": "euclidean",
#         "10D Uniform": "euclidean",
#         "20D Uniform": "euclidean",
#         "2D Sphere": "euclidean",
#         "3D Sphere": "euclidean",
#         "5D Sphere": "euclidean",
#         "Urn": "URN-R",
#         "Norm-Mallows": "MALLOWS-RELPHI-R",
#     }
#
#     used_models = {pm: preference_model_short_names[pm] for pm in preference_model}
#
#     profiles_per_dist = math.ceil(num_profiles / len(preference_model))
#     args = {
#         "n_profiles": profiles_per_dist,
#         "prefs_per_profile": prefs_per_profile,
#         "m": m,
#         "learned_pref_model": "",
#     }
#
#     all_distribution_details = []
#
#     for model_name, short_name in used_models.items():
#         args["learned_pref_model"] = short_name
#         kwargs = {}
#         if "Sphere" in model_name:
#             dimension = model_name.split(" ")[0][:-1]
#             kwargs["num_dimensions"] = eval(dimension)
#             kwargs["space"] = "uniform_sphere"
#         if "Uniform" in model_name:
#             dimension = model_name.split(" ")[0][:-1]
#             kwargs["num_dimensions"] = eval(dimension)
#             kwargs["space"] = "uniform_cube"
#
#         all_distribution_details.append((model_name, copy.copy(args), kwargs))
#
#     return all_distribution_details


def save_profiles(profiles, out_folder="data", filename=None):
    """
    Generate some preference rankings from a variety of distributions.
    :param profiles:
    :param out_folder:
    :param filename:
    :return:
    """
    if filename is None:
        k = len(profiles)
        n = len(profiles[0])
        m = len(profiles[0][0])
        filename = f"saved_preferences_n_profiles={k}-n={n}-m={m}"

    # # convert the individual rankings to lists rather than tuples to match format of existing data
    # final_profiles = []
    # for prf in pref_profiles:
    #     new_profile = []
    #     for rnk in prf:
    #         new_profile.append(list(rnk))
    #     final_profiles.append(new_profile)

    df = pd.DataFrame({
        'profile': profiles
    })
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    df.to_csv(os.path.join(out_folder, filename, ".csv"), index=False)


def utilities_from_profile(profile, normalize_utilities=False, utility_type="uniform_random"):
    """
    Create a single utility profile consistent with the given profile. Each ranking in the profile is turned into a
    list of floats where the first alternative ranked is the highest, second ranked is second highest, etc.
    That is, profile[i][j] = c indicates voter i ranks alternative c in position j.
    This implies that utility_profile[i][c] is the j-th highest value in utility_profile[i].
    :param profile: A list of lists where each inner list corresponds to a single voter's preference order.
    :param normalize_utilities: If True, normalize each voter's utility_profile so that they sum to 1.
    Having a different maximum utility value for each voter can affect outcomes (see: malfare sw function)
    :param utility_type: Which method to use to generate utility values. Two supported options.
    uniform_random: For m alternatives, generate a list  of m random floats between 0 and 1. Assign them as utility_profile
    to alternatives in a way consistent with the voter's preferences. Sample new values for each voter.
    linear: Generate evenly spaced values such that the highest value is m (number of alternatives) and the
    lowest value is 0.
    :return: list of lists where each inner list contains the utility given to each voter for each alternative winning.
    """

    def _utility_from_ranking(ranking):
        m = len(ranking)

        if utility_type == "linear":
            util_values = list(range(m, 0, -1))
        elif utility_type == "uniform_random":
            # Generate random values, assign them to correct rankings
            util_values = np.random.uniform(low=0, high=1, size=m)
            # else:
            util_values = util_values.tolist()
            util_values.sort(reverse=True)
        else:
            raise ValueError(f"Unexpected value given for 'utility_type'. Was given {utility_type}.")

        if normalize_utilities:
            util_values = [ut / sum(util_values) for ut in util_values]

        utilities = [0.0] * m  # put in position i the utility assigned to alternative i
        for i, preference in enumerate(ranking):
            # i is index, preference is the alternative being ranked in position i
            # ex. ranking = [2, 1, 0, 4, 3]
            utilities[preference] = util_values[i]

        return utilities

    all_utility_vectors = []
    rankings = profile._rankings if isinstance(profile, pref_voting.profiles.Profile) else profile
    for ranking in rankings:
        all_utility_vectors.append(_utility_from_ranking(ranking))

    return all_utility_vectors


def profile_from_utilies(utility_profile):
    """
    Generate the preference profile induced by the given utility profile.
    :param utility_profile: List of lists or ndarray where M[i][j] = u indicates that voter i
    gets utility u if j is elected.
    :return: List of lists or ndarray (matching input value) where R[i][j] = r indicates i ranks j in position r.
    """

    use_list = True
    if isinstance(utility_profile, np.ndarray):
        use_list = False

    rankings = []

    for i in range(len(utility_profile)):
        l = np.argsort(utility_profile[i])
        rankings.append(list(reversed(l.tolist())))

    if not use_list:
        rankings = np.asarray(rankings)
    return rankings


def rank_matrix(profile):
    """
    Find the m by m rank matrix R where R[c, p] is the number of voters who put candidate c in p-th position in
    their preference order.
    :param profile: A preference profile (list of lists or PrefVoting Profile)
    :return: ndarray WT representing the rank matrix of the profile
    """
    if isinstance(profile, pref_voting.profiles.Profile):
        profile = profile.rankings
    m = len(profile[0])  # length of first preference order in profile, assume for now all orders are complete

    # for pref_profiles in pref_profiles:
    rank_matrix = np.zeros((m, m), dtype=np.int64)
    for order in profile:
        for idx, c in enumerate(order):
            rank_matrix[c, idx] += 1

    return rank_matrix.tolist()


def weighted_tournament(profile):
    """
    Find the weighted tournament graph of the profile. WT[i, j] contains the number of voters that
    prefer candidate i over candidate j.
    :param profile: A preference profile (list of lists or PrefVoting Profile)
    :return: ndarray WT representing the weighted tournament graph of the profile
    """
    if isinstance(profile, pref_voting.profiles.Profile):
        profile = profile.rankings
    wt = np.zeros((profile.num_cands, profile.num_cands))
    for v, order in enumerate(profile):
        for i_idx, i in enumerate(order):
            for j_idx, j in enumerate(order):
                if j_idx <= i_idx:
                    continue  # only count when j is above i
                if i_idx == len(order) - 1:
                    continue  # don't let i take highest value (redundant)
                wt[i, j] += 1

    return wt


def default_job_name(**kwargs):
    terms_in_name = ["profile_score_agg_metric", "n_steps"]
    job_name = "annealing-"
    job_name_terms = [f"{k}={v}" for k, v in kwargs.items() if k in terms_in_name]
    return job_name + "-".join(job_name_terms)


if __name__ == "__main__":

    pref_dist_options = preference_distribution_options()
    mallows = pref_dist_options["Mallow's"]
    kwargs = {mallows["args"][0]: 0}

    mallows_profiles = mallows["function"](n_profiles=5,
                                                     n=10,
                                                     m=10,
                                                     **kwargs)

    utility_profiles = [utilities_from_profile(prf, normalize_utilities=True) for prf in mallows_profiles]

    print(f"Generated Mallow's pref_profiles with phi = {kwargs['phi']}")