import numpy as np
import pref_voting.profiles
# from optimal_voting.data_utils import make_mixed_preference_profiles, utilities_from_profile
import optimal_voting.data_utils as du


def social_welfare_for_positional_score_vector_many_profiles(profiles, all_utilities, score_vector,
                                                             sw_type="utilitarian", aggregation_method=np.mean, normalize=False):
    """
    Determine the aggregate social welfare of a score_vector over many pref_profiles. For each profile, determine the winner
    of the positional scoring rule defined by score_vector, then calculate the social welfare of that alternative on the
    provided utility_profile.
    Use the aggregation_method (default is np.mean) to report a single value corresponding to aggregate social welfare.
    :param profiles: A list where each entry is a list of lists where profile[i][j] = c indicates that voter i ranked candidate c in position j. This profile should be consistent with utility_profile.
    :param all_utilities: A list where each entry is a list of lists where utility_profile[i][j] contains the utility voter i receives if candidate j is
    elected. Assume that this list is complete; all voters have a utility value for each alternative.
    :param score_vector: A positional scoring score_vector where score_vector[i] = p denotes that a candidate ranked in
    position i by a voter should receive p points.
    :param sw_type: A string corresponding to a pre-defined social welfare function. Supported values are 'utilitarian',
     'nash_welfare', 'egalitarian', 'malfare', 'distortion-utilitarian', 'distortion-egalitarian'.
    :param aggregation_method: Method which accepts a list of floats and returns a single numeric value representing the
    aggregate social welfare across all pref_profiles.
    :param normalize: If set to True, scale each utility result so that the best candidate possible in EACH PROFILE would receive a utility of one and other values in that profiles are reported relative to this.
    :return:
    """

    assert len(profiles) == len(all_utilities)

    individual_sws = [
        social_welfare_for_positional_score_vector_single_profile(profiles[i], all_utilities[i], score_vector, sw_type, normalize)
        for i in range(len(profiles))]
    return aggregation_method(individual_sws)


def social_welfare_for_positional_score_vector_single_profile(profile, utilities, score_vector, sw_type="utilitarian", normalize=False):
    """
    Determine the social welfare of the winner of the given score_vector. That is, determine the winner based on
    the positional scoring rule defined by score_vector on the provided profile. Then calculate the social welfare
    of that alternative given the provided utility_profile.
    NOTE: This method does not do checks to confirm that profile and utility_profile are consistent, or that all elements have
    length matching in the correct dimensions.
    :param profile: a list of lists where profile[i][j] = c indicates that voter i ranked candidate c in position j.
    This profile should be consistent with utility_profile.
    :param utilities: a list of lists where utility_profile[i][j] contains the utility voter i receives if candidate j is
    elected. Assume that this list is complete; all voters have a utility value for each alternative.
    Requires that utility_profile is consistent with profile.
    :param score_vector: A positional scoring score_vector where score_vector[i] = p denotes that a candidate ranked in
    position i by a voter should receive p points.
    :param sw_type: A string corresponding to a pre-defined social welfare function. Supported values are 'utilitarian',
     'nash_welfare', 'egalitarian', 'malfare', 'distortion-utilitarian', 'distortion-egalitarian'.
    :param normalize: If set to True, scale each utility result so that the best candidate possible would receive a utility of one and other values are reported relative to this.
    :return:
    """
    # Get winner of score_vector
    winner = score_vector_winner(score_vector, profile, randomize=False)
    # get utility of winning alternative
    sw = social_welfare_for_alternative_single_profile(utilities, winner, sw_type=sw_type)
    if normalize:
        # find utility of all candidates in order to scale result
        all_sw = [social_welfare_for_alternative_single_profile(utilities, cand, sw_type=sw_type) for cand in range(len(score_vector))]
        best_sw = max(all_sw)
    else:
        best_sw = 1

    return sw / best_sw


def social_welfare_for_alternative_many_profiles(utilities, alternatives, sw_type="utilitarian",
                                                 aggregation_method=np.mean, normalize=False):
    """
    Determine the aggregate social welfare for each alternative being elected by voters with the given utility_profile, under
    the sw_type welfare.
    Use the aggregation_method (default is np.mean) to report a single value corresponding to aggregate social welfare.
    :param utilities: a list of lists where utility_profile[i][j] contains the utility voter i receives if candidate j is
    elected. Assume that this list is complete; all voters have a utility value for each alternative.
    We assume that utility_profile[i] is the utility profile associated with alternatives[i] (e.g., alternatives[i] may have
    been the winner of an election with voters from utility_profile[i])
    :param alternatives: A list of alternatives (type: list(int)) for which social welfare should be calculated.
    :param sw_type: A string corresponding to a pre-defined social welfare function. Supported values are 'utilitarian',
     'nash_welfare', 'egalitarian', 'malfare', 'distortion-utilitarian', 'distortion-egalitarian'.
    :param aggregation_method: Method which accepts a list of floats and returns a single numeric value representing the
    aggregate social welfare across all pref_profiles.
    :param normalize: If set to True, scale each utility result so that the best candidate possible in EACH PROFILE would receive a utility of one and other values in that profiles are reported relative to this.
    :return:
    """

    individual_sws = [social_welfare_for_alternative_single_profile(utilities, alt, sw_type=sw_type, normalize=normalize) for alt in
                      alternatives]
    return aggregation_method(individual_sws)


def social_welfare_for_alternative_single_profile(utilities, alternative, sw_type="utilitarian", normalize=False):
    """
    Determine the social welfare for 'alternative' being elected by voters with the given utility_profile, under
    the sw_type welfare.
    NOTE: At the moment both distortion methods are not well-thought-out and should be used only with caution.
    :param utilities: a list of lists where utility_profile[i][j] contains the utility voter i receives if candidate j is
    elected. Assume that this list is complete; all voters have a utility value for each alternative.
    :param alternative: An integer which is the label of one of the candidates. Must be below len(utility_profile[_]).
    :param sw_type: A string corresponding to a pre-defined social welfare function. Supported values are 'utilitarian',
     'nash_welfare', 'egalitarian', 'malfare', 'distortion-utilitarian', 'distortion-egalitarian'.
    :param normalize: If set to True, scale each utility result so that the best candidate possible would receive a utility of one and other values are reported relative to this.
    :return:
    """
    if isinstance(utilities, list):
        utilities = np.array(utilities)

    def single_alternative_welfare(alt):
        if sw_type == "utilitarian":
            sw = sum(utilities[:, alt])
        elif sw_type == "nash_welfare" or sw_type == "nash":
            # Keep a more scalable value by taking sum of logs rather than product
            sw = sum(np.log(utilities[:, alt] + 0.000001))
            # sw = np.prod(utility_profile[:, alternative])
        elif sw_type == "egalitarian":
            sw = min(utilities[:, alt])
        elif sw_type == "malfare":
            sw = max(utilities[:, alt])
        elif sw_type == "distortion-utilitarian":
            # find best possible utilitarian social welfare
            m = len(utilities[0])
            all_u_sws = [sum(utilities[:, a]) for a in range(m)]
            best_sw = max(all_u_sws)
            # return ratio of best utilitarian sw to actual utilitarian sw

            # we're thinking in a maximization context; invert the value so that higher is better
            sw = 1 / (best_sw / sum(utilities[:, alt]))
        elif sw_type == "distortion-egalitarian":
            # find best possible egalitarian social welfare
            m = len(utilities[0])
            all_u_sws = [min(utilities[:, a]) for a in range(m)]
            best_sw = max(all_u_sws)
            # return ratio of best egalitarian sw to actual egalitarian sw

            # we're thinking in a maximization context; invert the value so that higher is better
            sw = 1 / (best_sw / min(utilities[:, alt]))
        else:
            raise ValueError(f"Unexpected welfare type. Received '{sw_type}'.")
        return sw

    if normalize:
        all_sw = [single_alternative_welfare(idx) for idx in range(len(utilities[0]))]
        all_sw = [sw / max(all_sw) for sw in all_sw]
        sw_alt = all_sw[alternative]
    else:
        sw_alt = single_alternative_welfare(alternative)

    return sw_alt


def social_welfare_for_alternative_single_profile_torch(utilities, alternative, type="utilitarian"):
    """
    Determine the social welfare for 'alternative' being elected by voters with the given utility_profile, under
    the sw_type welfare.
    NOTE: At the moment both distortion methods are not well-thought-out and should be used only with caution.
    :param utilities: a list of lists where utility_profile[i][j] contains the utility voter i receives if candidate j is
    elected. Assume that this list is complete; all voters have a utility value for each alternative.
    :param alternative: An integer which is the label of one of the candidates. Must be below len(utility_profile[_]).
    :param sw_type: A string corresponding to a pre-defined social welfare function. Supported values are 'utilitarian',
     'nash_welfare', 'egalitarian', 'malfare', 'distortion-utilitarian', 'distortion-egalitarian'.
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
        # sw = np.prod(utility_profile[:, alternative])
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
        sw = 1 / (best_sw / torch.sum(utilities[:, alternative]))
    elif type == "distortion-egalitarian":
        # find best possible egalitarian social welfare
        m = len(utilities[0])
        all_u_sws = [torch.min(utilities[:, a]) for a in range(m)]
        best_sw = torch.max(all_u_sws)
        # return ratio of best egalitarian sw to actual egalitarian sw

        # we're maximizing it so we should somehow invert the value
        sw = 1 / (best_sw / torch.min(utilities[:, alternative]))
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


def score_vector_winner(score_vector, profile, randomize=False):
    """
    Return a single winning alternative given a positional scoring vector and a preference profile.
    Supports randomization; if normalize=True then return a winner selected with probability related to the score of
    each alternative.
    :param score_vector: A positional scoring score_vector where score_vector[i] = p denotes that a candidate ranked in
    position i by a voter should receive p points.
    :param profile: a list of lists where profile[i][j] = c indicates that voter i ranked candidate c in position j.
    This profile should be consistent with utility_profile.
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
        if sum(scores) == 0:
            prob_normed = [1 / len(scores) for _ in range(len(scores))]
        else:
            prob_normed = [s / sum(scores) for s in scores]
        winner = np.random.choice(list(range(m)), size=1, p=prob_normed)[0]
    else:
        winner = np.argmax(scores)
    return winner


def score_vector_ranking(score_vector, profile):
    """
    Return a ranking of all alternatives given a positional scoring vector and a preference profile. The alternative
    receiving the most points is ranked first, etc.
    :param score_vector: A positional scoring score_vector where score_vector[i] = p denotes that a candidate ranked in
    position i by a voter should receive p points.
    :param profile: a list of lists where profile[i][j] = c indicates that voter i ranked candidate c in position j.
    This profile should be consistent with utility_profile.
    :return: A list where the first element is the highest scoring alternative, second is second highest ranked, etc.
    """
    if isinstance(profile, pref_voting.profiles.Profile):
        profile = profile._rankings
    if isinstance(profile, list):
        profile = np.asarray(profile)

    full_score_vec = np.atleast_2d(score_vector).repeat(repeats=len(profile), axis=0)
    sorted_profiles = profile.argsort()
    scores = np.take_along_axis(full_score_vec, sorted_profiles, axis=1)
    scores = np.sum(scores, axis=0)

    return np.argsort(scores)[::-1]


def score_vector_scores(score_vector, profile, normalize=False, voter_weights=None):
    """
    Return the total score of each alternative based on the given profile and score_vector. If normalize is True the
    list is normalized to sum to 1, making it suitable to use as a lottery over winners.
    :param score_vector: A positional scoring score_vector where score_vector[i] = p denotes that a candidate ranked in
    position i by a voter should receive p points.
    :param profile: a list of lists where profile[i][j] = c indicates that voter i ranked candidate c in position j.
    This profile should be consistent with utility_profile.
    :param normalize: If False, return the exact scores, if True normalize scores so that sum(scores) == 1.
    :param voter_weights: If set, a list or ndarray with one entry per voter. Each voter has their scores scaled
    by their weight. e.g., all set to 1 is the unweighted setting.
    :return: A list with one entry for each voter providing that voter's score.
    """
    if isinstance(profile, pref_voting.profiles.Profile):
        profile = profile._rankings
    if isinstance(profile, list):
        profile = np.asarray(profile)

    if voter_weights is not None:
        if isinstance(voter_weights, list):
            voter_weights = np.array(voter_weights)
    else:
        voter_weights = np.ones(len(profile))

    full_score_vec = np.atleast_2d(score_vector).repeat(repeats=len(profile), axis=0)
    full_score_vec = full_score_vec * voter_weights[:, None]
    sorted_profiles = profile.argsort()
    scores = np.take_along_axis(full_score_vec, sorted_profiles, axis=1)
    scores = np.sum(scores, axis=0)

    if normalize:
        if sum(scores) == 0:
            scores = [1 / len(scores) for _ in range(len(scores))]
        else:
            scores = [s / sum(scores) for s in scores]
    return scores


def utilitarian_distortion(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utility_profiles"][unique_id], winners,
                                                         sw_type="distortion-utilitarian")


def egalitarian_distortion(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utility_profiles"][unique_id], winners,
                                                         sw_type="distortion-egalitarian")


def utilitarian_social_welfare(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utility_profiles"][unique_id], winners, sw_type="utilitarian")


def nash_social_welfare(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utility_profiles"][unique_id], winners,
                                                         sw_type="nash_welfare")


def egalitarian_social_welfare(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utility_profiles"][unique_id], winners, sw_type="egalitarian")


def malfare_social_welfare(unique_id, winners, profile, **kwargs):
    return social_welfare_for_alternative_single_profile(kwargs["utility_profiles"][unique_id], winners, sw_type="malfare")


# def social_welfare_of_score_vector_over_many_profiles(score_vector, pref_profiles, utility_profile, utility_type="utilitarian"):
#     """
#     Compute the utilitarian social welfare across a list of multiple pref_profiles/elections. Sum the
#     utility_type from each and return the result.
#     Utilitarian SW is the total sum of social welfare over all voters.
#     :param score_vector:
#     :param pref_profiles:
#     :param utility_profile:
#     :param utility_type:
#     :return:
#     """
#     all_score_vector_utilities = [score_vector_social_welfare_single_profile(score_vector,
#                                                                              pref_profiles[idx],
#                                                                              utility_profile[idx],
#                                                                              utility_type=utility_type)
#                                   for idx in range(len(pref_profiles))]
#     return sum(all_score_vector_utilities), np.mean(all_score_vector_utilities)


def normalize_score_vector(vec):
    """
    Normalize the given positional score vector so that the highest value is 1 and the lowest value is 0.
    :param vec: ndarray or list containing scores for each position
    :return: ndarray or list with the normalized score vector
    """
    if isinstance(vec, list):
        if min(vec) == max(vec):
            return [1] * len(vec)
        min_v = min(vec)
        vec = [v - min_v for v in vec]
        max_v = max(vec)
        vec = [v / max_v for v in vec]

    elif isinstance(vec, np.ndarray):

        if min(vec) == max(vec):
            return np.ones(len(vec))
        vec = vec - min(vec)
        vec = vec / max(vec)

    return vec


def score_vector_examples(m=5, normalize=True):
    """
    Generate several score vectors corresponding to well known rules and otherwise.
    :param m:
    :return:
    """
    vectors = {
        "Plurality": [1] + [0 for _ in range(m - 1)],
        "Veto": [1 for _ in range(m - 1)] + [0],
        "Borda": [m - idx - 1 for idx in range(m)],
        "Harmonic": [1 / (idx + 1) for idx in range(m)],  # See Optimal SCF paper ??
        "Plurality + Veto": [1] + [0 for _ in range(m - 2)] + [-1],
        "Two Approval": [1, 1] + [0 for _ in range(m - 2)],
        "Three Approval": [1, 1, 1] + [0 for _ in range(m - 3)],
        "Geometric": [1 / (2 ** i) for i in range(m)],
        "Half-Approval": [1] + [0.9 if idx < m // 2 else 0 for idx in range(m - 1)],
        "Half-Approval Degrading": [1] + [0.9 for _ in range(m // 2)] + [1 / (2 ** (idx + 1)) for idx in range(m // 2)] if m % 2 == 1 else [1] + [0.9 for _ in range(m // 2 - 1)] + [1 / (2 ** (idx + 1)) for idx in range(m // 2)],
        # "optimal_scf_paper": [0.25, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0]
    }

    if normalize:
        vectors = {name: normalize_score_vector(vector) for (name, vector) in vectors.items()}

    return vectors


def get_utility_eval_func_from_str(util_type):
    if util_type == "utilitarian":
        eval_func = utilitarian_social_welfare
    elif util_type == "egalitarian":
        eval_func = egalitarian_social_welfare
    elif util_type == "nash":
        eval_func = nash_social_welfare
    elif util_type == "malfare":
        eval_func = malfare_social_welfare
    elif util_type == "utilitarian_distortion":
        eval_func = utilitarian_distortion
    elif util_type == "egalitarian_distortion":
        eval_func = egalitarian_distortion
    else:
        raise ValueError("Didn't make other eval functions yet")
    return eval_func


if __name__ == "__main__":
    m = 10
    n = 100
    all_profiles = du.make_impartial_culture_profiles(n_profiles=1000,
                                                      n=n,
                                                      m=m)
    all_utilities = [du.utilities_from_profile(profile, normalize_utilities=False, utility_type="linear") for
                     profile in all_profiles]

    example_vectors = score_vector_examples(m)

    for vec_name, v in example_vectors.items():
        sw = social_welfare_for_positional_score_vector_many_profiles(profiles=all_profiles,
                                                                      all_utilities=all_utilities,
                                                                      score_vector=v,
                                                                      sw_type="nash",
                                                                      normalize=True)
        print(f"SW for {vec_name} is {sw}")
        print(f"Vector is: {v}")
        print(f"Normalized vector is: {normalize_score_vector(v)}")
        print()
