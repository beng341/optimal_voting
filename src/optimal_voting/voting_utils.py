import numpy as np
import pref_voting.profiles
from optimal_voting.data_utils import load_utility_vectors, load_profiles, make_mixed_preference_profiles, utilities_from_profile


def weighted_tournament(profile):
    """
    Given a PrefVoting profile, return an ndarray WT representing the weighted tournament graph of the profile.
    WT[i, j] contains the number of voters that prefer candidate i over candidate j.
    :param profile:
    :return:
    """
    wt = np.zeros((profile.num_cands, profile.num_cands))
    for v, order in enumerate(profile._rankings):
        for i_idx, i in enumerate(order):
            for j_idx, j in enumerate(order):
                if j_idx <= i_idx:
                    continue    # only count when j is above i
                if i_idx == len(order)-1:
                    continue    # don't let i take highest value (redundant)
                wt[i, j] += 1

    return wt


def social_welfare_for_alternative_single_profile(utilities, alternatives, type="utilitarian"):
    """
    Given utility_type vectors and the index of an alternative, determine the social welfare (sum of utilities) for that
    alternative being elected.
    NOTE: Be sure to give this a single set of profiles rather than a list of many profiles as is often passed around.
    :param utilities:
    :param alternatives:
    :return:
    """
    if isinstance(utilities, list):
        utilities = np.array(utilities)

    if len(alternatives) != 1:
        raise ValueError(f"Unexpected number of alternatives. Expected only one. Got: {alternatives}")
    alternative = alternatives[0]

    if type == "utilitarian":
        sw = sum(utilities[:, alternative])
    elif type == "nash_welfare":
        # Keep a more scalable value by taking sum of logs rather than product
        sw = sum(np.log(utilities[:, alternative] + 0.000001))
        # sw = np.prod(utilities[:, alternative])
    elif type == "egalitarian":
        sw = min(utilities[:, alternative])
    elif type == "malfare":
        sw = max(utilities[:, alternative])
    elif type == "distortion-utilitarian":
        # find best possible utilitarian social welfare
        m = len(utilities[0])
        all_u_sws = [sum(utilities[:, a]) for a in range(m)]
        best_sw = max(all_u_sws)
        # return ratio of best utilitarian sw to actual utilitarian sw

        # we're maximizing it so we should somehow invert the value
        sw = 1/(best_sw / sum(utilities[:, alternative]))
    elif type == "distortion-egalitarian":
        # find best possible egalitarian social welfare
        m = len(utilities[0])
        all_u_sws = [min(utilities[:, a]) for a in range(m)]
        best_sw = max(all_u_sws)
        # return ratio of best egalitarian sw to actual egalitarian sw

        # we're maximizing it so we should somehow invert the value
        sw = 1/(best_sw / min(utilities[:, alternative]))
    else:
        sw = -1

    return sw


def social_welfare_for_alternative_single_profile_torch(utilities, alternative, type="utilitarian"):
    """
    Given utility_type vectors and the index of an alternative, determine the social welfare (sum of utilities) for that
    alternative being elected.
    NOTE: Be sure to give this a single set of profiles rather than a list of many profiles as is often passed around.
    :param utilities:
    :param alternatives:
    :return:
    """
    import torch
    if isinstance(utilities, list):
        utilities = np.array(utilities)

    if type == "utilitarian":
        sw = torch.sum(utilities[:, alternative])
    elif type == "nash_welfare":
        # Keep a more scalable value by taking sum of logs rather than product
        sw = torch.sum(np.log(utilities[:, alternative]))
        # sw = np.prod(utilities[:, alternative])
    elif type == "egalitarian":
        sw = torch.min(utilities[:, alternative])
    elif type == "malfare":
        sw = torch.max(utilities[:, alternative])
    elif type == "distortion-utilitarian":
        # find best possible utilitarian social welfare
        m = len(utilities[0])
        all_u_sws = [torch.sum(utilities[:, a]) for a in range(m)]
        best_sw = torch.max(all_u_sws)
        # return ratio of best utilitarian sw to actual utilitarian sw

        # we're maximizing it so we should somehow invert the value
        sw = 1/(best_sw / torch.sum(utilities[:, alternative]))
    elif type == "distortion-egalitarian":
        # find best possible egalitarian social welfare
        m = len(utilities[0])
        all_u_sws = [torch.min(utilities[:, a]) for a in range(m)]
        best_sw = torch.max(all_u_sws)
        # return ratio of best egalitarian sw to actual egalitarian sw

        # we're maximizing it so we should somehow invert the value
        sw = 1/(best_sw / torch.min(utilities[:, alternative]))
    else:
        sw = -1

    return sw


def score_vector_winner_tensor(score_vector, profile):
    """
    Return the winner of the positional scoring rule defined by the list of single number tensors in score_vector.
    :param score_vector:
    :param profile:
    :return:
    """
    import torch

    # full_score_vec = torch.atleast_2d(score_vector)
    full_score_vec = torch.cat(score_vector).unsqueeze(0)

    sorted_profiles = profile.argsort()
    scores = torch.take_along_dim(full_score_vec, sorted_profiles, dim=1)
    scores = torch.sum(scores, axis=0)

    w = torch.argmax(scores)
    return w


def score_vector_winner(score_vector, profile, return_complete_results=False, randomize=False):
    """

    :param score_vector:
    :param profile:
    :param return_complete_results: Whether to return the complete ranking/probability distribution or just a single
    winning alternative.
    :param randomize: If False, return the alternative with highest score (or scores of all alternatives), if True
    select an alternative with probability proportional to the number of
    :return:
    """
    if isinstance(profile, pref_voting.profiles.Profile):
        profile = profile._rankings
    if isinstance(profile, list):
        profile = np.asarray(profile)

    full_score_vec_test = np.atleast_2d(score_vector)
    full_score_vec = np.atleast_2d(score_vector).repeat(repeats=len(profile), axis=0)
    sorted_profiles = profile.argsort()
    scores = np.take_along_axis(full_score_vec, sorted_profiles, axis=1)
    scores = np.sum(scores, axis=0)

    if randomize:
        m = len(scores)
        if sum(scores) == 0:
            prob_normed = [1/m for _ in range(m)]
        else:
            prob_normed = [s/sum(scores) for s in scores]
        if return_complete_results:
            # return raw probabilities for each alternative being chosen
            return prob_normed
        else:
            # return single randomly chosen alternative
            return np.random.choice(list(range(m)), size=1, p=prob_normed)[0]
    else:
        if return_complete_results:
            # return the score assigned to each alternative
            return np.argsort(scores)[::-1]
        else:
            # return single highest scoring alternative
            # TODO: Allow different tie-breaking methods
            return np.argmax(scores)


def utilitarian_distortion(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utilities"][unique_id], winners, type="distortion-utilitarian")


def egalitarian_distortion(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utilities"][unique_id], winners, type="distortion-egalitarian")


def utilitarian_social_welfare(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utilities"][unique_id], winners, type="utilitarian")


def nash_social_welfare(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utilities"][unique_id], winners, type="nash_welfare")


def egalitarian_social_welfare(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utilities"][unique_id], winners, type="egalitarian")


def malfare_social_welfare(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utilities"][unique_id], winners, type="malfare")


# def social_welfare_of_score_vector_over_many_profiles(score_vector, profiles, utilities, utility_type="utilitarian"):
#     """
#     Compute the utilitarian social welfare across a list of multiple profiles/elections. Sum the
#     utility_type from each and return the result.
#     Utilitarian SW is the total sum of social welfare over all voters.
#     :param score_vector:
#     :param profiles:
#     :param utilities:
#     :param utility_type:
#     :return:
#     """
#     all_score_vector_utilities = [score_vector_social_welfare_single_profile(score_vector,
#                                                                              profiles[idx],
#                                                                              utilities[idx],
#                                                                              utility_type=utility_type)
#                                   for idx in range(len(profiles))]
#     return sum(all_score_vector_utilities), np.mean(all_score_vector_utilities)


def normalize_score_vector(vec):
    """
    Normalize so that the highest value is 1 and the lowest value is 0. This shouldn't affect the social welfare.
    :param vec: ndarray containing scores for each position
    :return:
    """
    if isinstance(vec, list):
        vec = np.asarray(vec)
    if min(vec) == max(vec):
        return np.ones(len(vec))
    vec = vec - min(vec)    # subtract this from all values to get the lowest score to zero and all values positive

    # get max value to 1 and others suitably scaled
    vec = vec / max(vec)
    # vec = [round(v/max(vec), 3) for v in vec]

    return vec


def score_vector_examples(m=5):
    """
    Generate several score vectors corresponding to well known rules and otherwise.
    :param m:
    :return:
    """
    plurality = [1] + [0 for _ in range(m-1)]
    plurality_veto = [1] + [0 for _ in range(m-2)] + [-1]
    veto = [0 for _ in range(m-1)] + [-1]
    borda = [m-idx-1 for idx in range(m)]
    squared_borda = [(m-idx-1)**2 for idx in range(m)]
    cubed_borda = [(m-idx-1)**3 for idx in range(m)]
    two_approval = [1, 1] + [0 for _ in range(m-2)]
    half_approval = [1] + [0.9 if idx < m//2 else 0 for idx in range(m-1)]
    geometric_decreasing = [1/(2**i) for i in range(m)]
    if m % 2 == 1:
        half_approval_degrading = [1] + [0.9 for _ in range(m//2)] + [1/(2**(idx+1)) for idx in range(m//2)]
    else:
        half_approval_degrading = [1] + [0.9 for _ in range(m//2-1)] + [1 / (2 ** (idx + 1)) for idx in range(m//2)]

    # all_score_vectors = [plurality, plurality_veto, veto, borda, squared_borda, cubed_borda, two_approval, symmetric,
    #                      symmetric_geometric]
    all_score_vectors = {
        "plurality": plurality,
        "plurality_veto": plurality_veto,
        "veto": veto,
        "borda": borda,
        "squared_borda": squared_borda,
        "cubed_borda": cubed_borda,
        "two_approval": two_approval,
        "half_approval": half_approval,
        "half_approval_degrading": half_approval_degrading,
        "geometric_decreasing": geometric_decreasing,
    }
    return all_score_vectors


if __name__ == "__main__":
    m = 10
    all_profiles = make_mixed_preference_profiles(profiles_per_distribution=10,
                                                  n=10,
                                                  m=10)
    all_utilities = [utilities_from_profile(profile, normalize_utilities=True, utility_type="uniform_random") for profile in all_profiles]

    vectors = score_vector_examples(m)

    for vec_name, vec in vectors.items():
        sw = social_welfare_of_score_vector_over_many_profiles(vec, profiles=all_profiles, utilities=all_utilities)
        print(f"SW for {vec} is {sw}")
        sw = social_welfare_of_score_vector_over_many_profiles(normalize_score_vector(vec), profiles=all_profiles, utilities=all_utilities)
        print(f"SW for (normed) {vec} is {sw}")