from mip import Model, xsum, maximize, BINARY, CONTINUOUS, INTEGER
from optimal_voting.voting_utils import social_welfare_for_alternative_single_profile
from optimal_voting.data_utils import utilities_from_profile, make_mixed_preference_profiles, rank_matrix


def _optimize_score_vector_mip_experimental(social_welfare_lists, rank_matrices):
    """
    Optimize a single score score_vector to maximize mean social welfare across multiple pref_profiles.
    # TODO: Does it make sense to maximize median social welfare?

    Args:
        social_welfare_lists: list of utility lists, one for each profile
                        Agnostic to underlying sw function; the social welfare of each alternative winning.
                       utilities_list[p][i] = welfare from candidate i winning in profile p
        rank_matrices: list of rank matrices, one for each profile
                      rank_matrices[p][i][j] = times candidate i ranked at position j in profile p

    Returns:
        model: the optimized MIP model
        score_vector: the optimal score score_vector variable list (single, shared across pref_profiles)
        winners_list: list of binary winner variables for each profile
    """
    n_profiles = len(social_welfare_lists)
    n_cands = len(social_welfare_lists[0])
    m = len(rank_matrices[0][0])  # number of ranking positions

    # Create model
    model = Model("multi_profile_score_vector_optimization")

    # Variables - ONE score score_vector shared across all pref_profiles
    score_vector = [model.add_var(var_type=CONTINUOUS, lb=0, ub=1,
                                  name=f'score_{j}') for j in range(m)]

    # Points and winners for each profile
    points_list = []
    winners_list = []

    for p in range(n_profiles):
        points = [model.add_var(var_type=CONTINUOUS, lb=0,
                                name=f'points_p{p}_c{i}') for i in range(n_cands)]
        winners = [model.add_var(var_type=BINARY,
                                 name=f'winner_p{p}_c{i}') for i in range(n_cands)]
        points_list.append(points)
        winners_list.append(winners)

    # Big M for winner constraints (calculate per profile)
    M_list = [sum(sum(rank_matrices[p][i]) for i in range(n_cands))
              for p in range(n_profiles)]
    epsilon = 0.001

    # Objective: Maximize mean sw across all pref_profiles
    total_sw = xsum(social_welfare_lists[p][i] * winners_list[p][i]
                         for p in range(n_profiles)
                         for i in range(n_cands))
    model.objective = maximize(total_sw / n_profiles)

    # Constraints on score_vector (shared across all pref_profiles)
    model += score_vector[0] == 1, "top_rank_score"
    model += score_vector[m - 1] == 0, "bottom_rank_score"

    for j in range(m - 1):
        model += score_vector[j] >= score_vector[j + 1], f"decreasing_score_{j}"

    # Constraints for each profile
    for p in range(n_profiles):
        points = points_list[p]
        winners = winners_list[p]
        rank_matrix = rank_matrices[p]
        M = M_list[p]

        # Points calculation using the SHARED score_vector
        for i in range(n_cands):
            model += points[i] == xsum(rank_matrix[i][j] * score_vector[j]
                                       for j in range(m)), f"points_p{p}_c{i}"

        # Exactly one winner per profile
        model += xsum(winners[i] for i in range(n_cands)) == 1, f"one_winner_p{p}"

        # Winner has highest points in this profile
        # We need to find a scoring score_vector which does not result in a tied winner, winner must be strictly higher scoring
        # for i in range(n_cands):
        #     for j in range(n_cands):
        #         if i != j:
        #             model += points[i] >= points[j] + epsilon - M * (1 - winners[i]), \
        #                 f"winner_highest_p{p}_c{i}_c{j}"
        for i in range(n_cands):
            if winners[i] == 1:
                for j in range(n_cands):
                    if i == j:
                        continue
                    model += points[i] >= points[j] + epsilon, f"winner_highest_p{p}_c{i}_c{j}"

            # for j in range(n_cands):
            #     if i != j:
            #         model += points[i] >= points[j] + epsilon - M * (1 - winners[i]), \
            #             f"winner_highest_p{p}_c{i}_c{j}"

    return model, score_vector, winners_list


def _optimize_score_vector_mip(social_welfare_lists, rank_matrices):
    """
    Optimize a single score score_vector to maximize mean social welfare across multiple pref_profiles.
    # TODO: Does it make sense to maximize median social welfare?

    Args:
        social_welfare_lists: list of utility lists, one for each profile
                        Agnostic to underlying sw function; the social welfare of each alternative winning.
                       utilities_list[p][i] = welfare from candidate i winning in profile p
        rank_matrices: list of rank matrices, one for each profile
                      rank_matrices[p][i][j] = times candidate i ranked at position j in profile p

    Returns:
        model: the optimized MIP model
        score_vector: the optimal score score_vector variable list (single, shared across pref_profiles)
        winners_list: list of binary winner variables for each profile
    """
    n_profiles = len(social_welfare_lists)
    n_cands = len(social_welfare_lists[0])
    m = len(rank_matrices[0][0])  # number of ranking positions

    # Create model
    model = Model("multi_profile_score_vector_optimization")

    # Variables - ONE score score_vector shared across all pref_profiles
    score_vector = [model.add_var(var_type=CONTINUOUS, lb=0, ub=1,
                                  name=f'score_{j}') for j in range(m)]

    # Points and winners for each profile
    points_list = []
    winners_list = []

    for p in range(n_profiles):
        points = [model.add_var(var_type=CONTINUOUS, lb=0,
                                name=f'points_p{p}_c{i}') for i in range(n_cands)]
        winners = [model.add_var(var_type=BINARY,
                                 name=f'winner_p{p}_c{i}') for i in range(n_cands)]
        points_list.append(points)
        winners_list.append(winners)

    # Big M for winner constraints (calculate per profile)
    M_list = [sum(sum(rank_matrices[p][i]) for i in range(n_cands))
              for p in range(n_profiles)]
    epsilon = 0.001

    # Objective: Maximize mean sw across all pref_profiles
    total_sw = xsum(social_welfare_lists[p][i] * winners_list[p][i]
                         for p in range(n_profiles)
                         for i in range(n_cands))
    model.objective = maximize(total_sw / n_profiles)

    # Constraints on score_vector (shared across all pref_profiles)
    model += score_vector[0] == 1, "top_rank_score"
    model += score_vector[m - 1] == 0, "bottom_rank_score"

    for j in range(m - 1):
        model += score_vector[j] >= score_vector[j + 1], f"decreasing_score_{j}"

    # Constraints for each profile
    for p in range(n_profiles):
        points = points_list[p]
        winners = winners_list[p]
        rank_matrix = rank_matrices[p]
        M = M_list[p]

        # Points calculation using the SHARED score_vector
        for i in range(n_cands):
            model += points[i] == xsum(rank_matrix[i][j] * score_vector[j]
                                       for j in range(m)), f"points_p{p}_c{i}"

        # Exactly one winner per profile
        model += xsum(winners[i] for i in range(n_cands)) == 1, f"one_winner_p{p}"

        # Winner has highest points in this profile
        # We need to find a scoring score_vector which does not result in a tied winner, winner must be strictly higher scoring
        for i in range(n_cands):
            for j in range(n_cands):
                if i != j:
                    # model += points[i] >= points[j] - M * (1 - winners[i]), \
                    #     f"winner_highest_p{p}_c{i}_c{j}"
                    model += points[i] >= points[j] + epsilon - M * (1 - winners[i]), \
                        f"winner_highest_p{p}_c{i}_c{j}"

    return model, score_vector, winners_list


def optimize_score_vector_mip(profiles, utilities, sw_function, max_seconds=20, verbose=False):
    sw_lists = [
        # [social_welfare_for_alternative_single_profile(utility_profile[j], i, sw_type=sw_function) for i in
        #  range(len(utility_profile[j][0]))]
        # for j in range(len(utility_profile))
    ]
    for j in range(len(utilities)): # each profile
        sw_list = []
        for i in range(len(utilities[j][0])):  # each candidate in the profile
            sw_list.append(social_welfare_for_alternative_single_profile(utilities[j], i, sw_type=sw_function))
        sw_lists.append(sw_list)

    # 3. Make rank matrix for each profile
    rank_matrices = [rank_matrix(profile) for profile in profiles]

    # model, score_vector, winners_list = _optimize_score_vector_mip(
    #     social_welfare_lists=sw_lists,
    #     rank_matrices=rank_matrices
    # )

    model, score_vector, winners_list = _optimize_score_vector_mip_experimental(
        social_welfare_lists=sw_lists,
        rank_matrices=rank_matrices
    )

    if not verbose:
        model.verbose = False

    # Optimize
    model.optimize(max_seconds=max_seconds)

    # Print results
    if verbose and model.num_solutions:
        print(f"Optimal mean utility: {model.objective_value:.4f}")
        print(f"\nOptimal score score_vector:")
        for j, sv in enumerate(score_vector):
            print(f"  Position {j}: {sv.x:.4f}")

    final_vector = [sv.x for sv in score_vector]
    best_sw = model.objective_value
    return final_vector, best_sw


# Example usage:
if __name__ == "__main__":
    profiles_per_dist = 5
    n = 5
    m = 5

    # 1. Generate some pref_profiles and their corresponding utility_profile
    profiles = make_mixed_preference_profiles(profiles_per_distribution=profiles_per_dist,
                                              n=n,
                                              m=m,
                                              seed=4
                                              )
    utilities_for_profiles = [utilities_from_profile(profile, normalize_utilities=False, utility_type="uniform_random") for profile in profiles]


    # pref_profiles = [
    #     [
    #         (0, 1, 2, 3, 4),
    #         (0, 1, 3, 2, 4),
    #         (4, 3, 2, 1, 0),
    #     ]
    # ]
    # utilities_for_profiles = [
    #     [
    #         [1, 0.75, 0.5, 0.3, 0],
    #         [1, 0.75, 0.25, 0.5, 0],
    #         [0, 0.25, 0.5, 0.75, 1],
    #     ]
    # ]
    # utilities_for_profiles = [utilities_from_profile(profile, normalize_utilities=False, utility_type="uniform_random") for profile in pref_profiles]


    # 2. For whatever SW function, find the SW of each candidate winning across each profile
    # sw_function = "utilitarian"
    # sw_function = "malfare"
    sw_function = "egalitarian"
    # sw_function = "nash"

    max_seconds = 300   # float("inf")
    final_vector, best_sw = optimize_score_vector_mip(profiles, utilities_for_profiles, sw_function, max_seconds=max_seconds)

    print(f"Best score_vector: {final_vector}")
    print(f"Best {sw_function} social welfare: {best_sw}")