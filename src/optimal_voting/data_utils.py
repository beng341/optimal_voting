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
        wt = np.zeros((profile.num_cands, profile.num_cands))
        profile = profile.rankings
    elif isinstance(profile, list):
        m = len(profile[0])
        wt = np.zeros((m, m))
    else:
        raise TypeError(
            f"Unexpected type for profile. Expected pref_voting.profiles.Profile or list. Got {type(profile)}")
    for v, order in enumerate(profile):
        for i_idx, i in enumerate(order):
            for j_idx, j in enumerate(order):
                if j_idx <= i_idx:
                    continue  # only count when j is above i
                if i_idx == len(order) - 1:
                    continue  # don't let i take highest value (redundant)
                wt[i, j] += 1
    #
    # wt[wt == 0] = 1000

    return wt


def default_job_name(**kwargs):
    terms_in_name = ["profile_score_agg_metric", "n_steps"]
    job_name = "annealing-"
    job_name_terms = [f"{k}={v}" for k, v in kwargs.items() if k in terms_in_name]
    return job_name + "-".join(job_name_terms)


def validate_preference_profiles(profiles):
    """
    Determine whether the set of profiles meets the requirements for optimization. At the very least:
    - profiles must be a list of lists of lists of integers
    - each inner preference order must have the same candidates
    - the candidates must be numbered from 0 through m-1
    - each profile must have the same number of candidates
    :param profile:
    :return: tuple of (bool, str) indicating whether the profile is valid and (if applicable) which condition was failed
    """
    try:
        if not isinstance(profiles, list) or not isinstance(profiles[0], list) or not isinstance(profiles[0][0],
                                                                                                 list) or not (isinstance(
                profiles[0][0][0], int) or isinstance(profiles[0][0][0], np.int64)):
            reason = f"profiles must have type List[List[List[int]]]."
            return False, reason
    except Exception as e:
        reason = f"Encountered exception. Input profiles list may be empty. Exception message: {e}"
        return False, reason

    cands = None
    for prof in profiles:
        for order in prof:
            if cands is None:
                cands = set(range(len(order)))

            # must rank every candidate exactly once
            if len(order) != len(cands):
                reason = "A profile exists in which some candidate is not ranked exactly once."
                return False, reason

            # must rank candidates 0 through m-1
            if set(order) != set(cands):
                reason = f"Must rank all candidates with 0-index labels increasing consecutively. Found profile ranking: {set(order)}"
                return False, reason

    return True, ""

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
