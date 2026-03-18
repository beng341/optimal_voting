import copy
import random
import numpy as np
import optimal_voting.data_utils as du
import torch
from optimal_voting.voting_utils import egalitarian_social_welfare, utilitarian_social_welfare


def social_welfare_for_alternative_single_profile_torch(utilities, alternative_weights, opt_type="utilitarian"):
    """
    Take weight score_vector instead of a single alternative index in order to maintain gradient computations.
    Returns weighted sum of utility_profile across all alternatives.
    """
    sw = None
    if opt_type == "utilitarian":
        sw = torch.sum(utilities * alternative_weights)
    elif opt_type == "nash":
        sw = torch.sum(torch.log(utilities) * alternative_weights)
    elif opt_type == "egalitarian":
        prod = utilities * alternative_weights
        winner_util, _ = torch.max(prod, dim=1)
        sw = torch.min(winner_util)
    elif opt_type == "malfare":
        prod = utilities * alternative_weights
        winner_util, _ = torch.max(prod, dim=1)     # get utility_profile from just the column of the winner
        sw = torch.max(winner_util)     # utility of the best off voter
    return sw


def score_vector_winner_tensor(score_vector, profile, temperature=0.0001):
    """
    Return the winner of the positional scoring rule defined by the list of single number tensors in score_vector.
    :param score_vector:
    :param profile:
    :return:
    """
    full_score_vec = torch.cat(score_vector).unsqueeze(0)

    sorted_profiles = profile.argsort()
    scores = torch.take_along_dim(full_score_vec, sorted_profiles, dim=1)
    scores = torch.sum(scores, axis=0)

    soft_winner = torch.softmax(scores / temperature, dim=0)
    return soft_winner


def gradient_descent(profiles, utilities, initial_state, opt_target="egalitarian", max_n_iterations=100, debug=False, verbose=False):
    """

    """

    profiles = [torch.tensor(profile, dtype=int) for profile in profiles]
    utilities = torch.as_tensor(utilities)

    m = len(profiles[0][0])
    # n = len(pref_profiles[0])
    # initial_vector = [torch.tensor([(m-1-i)/(m-1)], requires_grad=True) for i in range(m)]  # start at Borda
    # initial_vector = [torch.tensor([1.0], requires_grad=True)] + [torch.tensor([0.0], requires_grad=True) for i in range(m-1)]    # Plurality
    initial_vector = [torch.tensor([initial_state[i]], requires_grad=True) for i in range(m)]

    if verbose:
        print(f"Initial Score Vector: {[round(ai.item(), 4) for ai in initial_vector]}")

    # score_vector_tensor = copy.copy(initial_vector)
    # optimizer = torch.optim.SGD([score_vector_tensor[i] for i in range(1, m - 1)], lr=0.1)
    # optimizer = torch.optim.SGD([initial_vector[i] for i in range(1, m - 1)], lr=0.01)
    optimizer = torch.optim.Adam([initial_vector[i] for i in range(1, m - 1)], lr=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=10,
                                                           cooldown=10)

    best_loss = float('inf')
    best_vector = None
    n_iterations = 0
    # max_n_iterations = 20

    while n_iterations < max_n_iterations:

        optimizer.zero_grad()
        current_loss = 0.0

        for idx, profile in enumerate(profiles):
            # winners = vu.score_vector_winner_tensor(score_vector_tensor, profile)
            soft_winners = score_vector_winner_tensor(initial_vector, profile)

            # need loss which gets higher when the outcome is worse
            # Calculate expected social welfare (weighted by winner probabilities)
            sw = social_welfare_for_alternative_single_profile_torch(utilities[idx],
                                                                     soft_winners,
                                                                     opt_type=opt_target)

            # Loss: higher SW is better, so we want to minimize negative SW
            current_loss = current_loss + (-sw)

        current_loss = current_loss / len(profiles)

        # DEBUG: Check gradients before backward
        if debug:
            print(f"\nIteration {n_iterations}:")
            print(f"  Loss = {current_loss.item():.6f}")

        if current_loss < best_loss:
            best_loss = current_loss
            best_vector = [ai.item() for ai in initial_vector]

        current_loss.backward()
        torch.nn.utils.clip_grad_norm_([initial_vector[i] for i in range(1, m - 1)], max_norm=1.0)

        # DEBUG: Check gradients after backward
        if debug:
            print(f"  Gradients:")
            for i in range(1, m - 1):
                print(f"    initial_vector[{i}].grad = {initial_vector[i].grad}")

        optimizer.step()
        scheduler.step(current_loss)
        with torch.no_grad():
            for i in range(1, m - 1):
                initial_vector[i].data = torch.clamp(initial_vector[i].data,
                                                     min=initial_vector[m - 1].item(),
                                                     max=initial_vector[i - 1].item())


        if n_iterations % 10 == 0 and verbose:  # Print every 10 iterations
            print(f"Iteration {n_iterations}: Loss = {current_loss.item():.6f}, "
                  f"LR = {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Score Vector = {[round(ai.item(), 4) for ai in initial_vector]}")
        n_iterations += 1

    if verbose:
        print(f"Best score score_vector from gradient descent is: {best_vector} with loss {best_loss}")

    return best_vector, best_loss


if __name__ == "__main__":

    seed = 43
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    n = 50
    m = 10

    profiles = du.make_mixed_preference_profiles(profiles_per_distribution=30,
                                                 n=50,
                                                 m=10,
                                                 seed=seed)

    utilities = [du.utilities_from_profile(profile) for profile in profiles]
    # pref_profiles = [torch.tensor(profile._rankings, dtype=int) for profile in pref_profiles]
    initial_state = [(m-1-i)/(m-1) for i in range(m)]

    gradient_descent(profiles,
                     utilities=utilities,
                     initial_state=initial_state,
                     opt_target="egalitarian",
                     max_n_iterations=100,
                     verbose=True)
