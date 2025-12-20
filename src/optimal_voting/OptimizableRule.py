import copy
import math
import os.path
from abc import abstractmethod
import random
import numpy as np
import pandas as pd
import pref_voting.profiles
import abcvoting.preferences
import time
import sys
from collections import defaultdict
from simanneal import Annealer
from optimal_voting.data_utils import utilities_from_profile, weighted_tournament, profile_from_utilies
# from optimal_voting.voting_utils import normalize_score_vector, score_vector_winner, get_utility_eval_func_from_str, score_vector_scores
import optimal_voting.voting_utils as vu


class OptimizableRule(Annealer):

    def __init__(self, state, eval_func, **kwargs):
        """

        :param state:
        :param pref_profiles:
        """
        self.verbose = kwargs.get("verbose", False)

        ################################################################################
        #   Set up preferences and voter identities for whatever preference format was given
        ################################################################################
        self.vmap = dict()  # maps external voter id to internal id (0-indexed and consecutive)
        self.cmap = dict()  # maps external candidate id to internal id (0-indexed and consecutive)

        self.pref_profiles = []
        self.utility_profiles = []

        ################################################################################
        #   Set up items related to running the optimization and storing results
        ################################################################################

        self.parse_preference_profiles(kwargs)
        self.parse_utility_profiles(kwargs)

        assert len(self.pref_profiles) > 0 or len(self.utility_profiles) > 0, "Must provide at least one of ordinal or cardinal data."

        # ensure both profile formats exist
        if len(self.utility_profiles) == 0:
            if self.verbose:
                print("Utilities not provided. Generating utility_profile consistent with profile.")
            for pref_profile in self.pref_profiles:
                self.utility_profiles.append(utilities_from_profile(pref_profile,
                                           normalize_utilities=kwargs.get("normalize_utilities", True),
                                           utility_type=kwargs.get("utility_type", "linear")))
        if len(self.pref_profiles) == 0:
            if self.verbose:
                print("Utilities not provided. Generating utility_profile consistent with profile.")
            for util_profile in self.utility_profiles:
                self.pref_profiles.append(profile_from_utilies(util_profile))

        # how many winners to compute per profile
        if "num_winners" in kwargs:
            if isinstance(kwargs["num_winners"], int):
                self.num_winners = [kwargs["num_winners"]] * len(self.pref_profiles)
            elif isinstance(kwargs["num_winners"], list):
                self.num_winners = kwargs["num_winners"]
            else:
                raise ValueError("num_winners must be int or list")
        else:
            self.num_winners = [1] * len(self.pref_profiles)

        # if voter weights given as a dict, use mapping to convert to list
        if "voter_weights" in kwargs and kwargs["voter_weights"] is not None:
            if isinstance(kwargs["voter_weights"], dict):
                vw = [1 for _ in range(len(kwargs["voter_weights"]))]
                for external_name, internal_name in self.vmap.items():
                    vw[internal_name] = kwargs["voter_weights"][external_name]
                kwargs["voter_weights"] = vw


        ################################################################################
        #   Set up items related to running optimization and storing results
        ################################################################################
        if "optimization_method" not in kwargs:
            kwargs["optimization_method"] = "annealing"
        self.optimization_method = kwargs["optimization_method"]

        self.kwargs = kwargs
        self.kwargs["utility_profiles"] = self.utility_profiles
        self.kwargs["pref_profiles"] = self.pref_profiles

        self.state = state
        if isinstance(eval_func, str):
            self.evaluation_function = vu.get_utility_eval_func_from_str(eval_func)
        else:
            self.evaluation_function = eval_func

        if "job_name" in kwargs:
            self.job_name = kwargs["job_name"]
        else:
            self.job_name = "annealing_job"
        if "keep_history" in kwargs and kwargs["keep_history"]:
            # self.keep_history = True
            self.history = defaultdict(list)
            # self.history = {
            #     "current_state": [],
            #     "current_energy": [],
            #     "best_state": [],
            #     "best_energy": [],
            #     "step": []
            # }
        else:
            self.history = None

        if "num_history_updates" in kwargs:
            self.num_history_updates = kwargs["num_history_updates"]
            # self.updates = kwargs["num_history_updates"]
        else:
            # default value will update 100 times
            self.num_history_updates = 100

        if "history_path" in self.kwargs:
            if not os.path.exists(self.kwargs["history_path"]):
                try:
                    os.makedirs(self.kwargs["history_path"])
                    self.history_path = self.kwargs["history_path"]
                except Exception as e:
                    print(f"Unable to create given energy history path. Continuing without saving history. "
                          f"Was given: {self.kwargs['history_path']}", file=sys.stderr)
                    self.history_path = None
            else:
                self.history_path = self.kwargs["history_path"]
        else:
            self.history_path = None

        try:
            super().__init__(initial_state=state)
        except ValueError as e:
            # If signal.signal fails (not in main thread), manually initialize
            # temp workaround until a better annealer is created
            if "signal only works in main thread" in str(e):
                # Manually do what Annealer.__init__ does without the signal
                self.state = self.copy_state(state)
                self.user_exit = False
            else:
                raise

    def parse_preference_profiles(self, kwargs):
        """
        Extract preference profiles from kwargs given during rule creation, if they exist.
        Accepts two formats of preference data, all profiles must be in at most one format. If not given, utility
        profiles must be given and preferences will later be generated based on provided utility data.
        :param kwargs:
        :return:
        """

        assert ("pref_profile_lists" not in kwargs) or ("pref_profile_dicts" not in kwargs), "Must provide only one format of preference data"

        if "pref_profile_lists" in kwargs:
            assert len(kwargs["pref_profile_lists"]) > 0, "Cannot pass empty pref_profiles argument"
            # Voter and candidate labels require no mapping; set up default maps
            n_voters_max = 0
            n_cands = 0
            for profile in kwargs["pref_profile_lists"]:
                self.pref_profiles.append(profile)

                n_voters_max = max(len(self.pref_profiles[-1]), n_voters_max)
                n_cands = len(self.pref_profiles[0])
            self.vmap = {i: i for i in range(n_voters_max)}
            self.cmap = {i: i for i in range(n_cands)}

        if "pref_profile_dicts" in kwargs:
            # Each profile is a dict of dicts where each voter id is mapped to each candidate id is mapped to rank
            for profile in kwargs["pref_profile_dicts"]:
                util_vals = []
                for voter_id, voter_ranks in profile.items():
                    if voter_id not in self.vmap:
                        self.vmap[voter_id] = len(self.vmap)

                    pref_order = []
                    for cand_id, rank in voter_ranks.items():
                        if cand_id not in self.cmap:
                            self.cmap[cand_id] = len(self.cmap)

                        pref_order[rank] = self.cmap[cand_id]
                    util_vals.append(pref_order)

                # TODO: do we really need to use pref_voting Profile objects?
                self.pref_profiles.append(util_vals)

    def parse_utility_profiles(self, kwargs):
        """
        Extract utility profiles from kwargs given during rule creation, if they exist.
        Accepts two formats of utility data, all profiles must be in at most one format. If not given, preference
        profiles must be given and these will later be generated based on provided preference data.
        :param kwargs:
        :return:
        """

        assert ("utility_profile_lists" not in kwargs) or (
                    "utility_profile_dicts" not in kwargs), "Must provide only one format of utility data"

        if "utility_profile_lists" in kwargs:
            assert len(kwargs["utility_profile_lists"]) > 0, "Cannot pass empty utility profiles"
            # Voter and candidate labels require no mapping; set up default maps
            n_voters_max = 0
            n_cands = 0
            for profile in kwargs["utility_profile_lists"]:
                # list of lists where profile[i][j] = c indicates voter i gets c utility if j is elected
                self.utility_profiles.append(profile)
                n_voters_max = max(len(self.utility_profiles[-1]), n_voters_max)
                n_cands = len(self.utility_profiles[0])
            self.vmap = {i: i for i in range(n_voters_max)}
            self.cmap = {i: i for i in range(n_cands)}

        if "utility_profile_dicts" in kwargs:
            # Each profile is a dict of dicts where each voter id is mapped to each candidate id is mapped to rank
            for profile in kwargs["utility_profile_dicts"]:
                util_vals = []
                for voter_id, voter_utils in profile.items():
                    if voter_id not in self.vmap:
                        self.vmap[voter_id] = len(self.vmap)

                    pref_order = [0 for _ in range(len(voter_utils))]
                    for cand_id, util in voter_utils.items():
                        if cand_id not in self.cmap:
                            self.cmap[cand_id] = len(self.cmap)
                        pref_order[self.cmap[cand_id]] = util
                    util_vals.append(pref_order)

                self.utility_profiles.append(util_vals)

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def rule_winners(self):
        """
        Evaluate the current rule on all pref_profiles. Return a list with one entry per profile, in order according to
        pref_profiles list.
        Each entry in returned list should be an iterable of tied winners. Typically this should be length one but
        we prefer to not lose generality at this stage. Also applicable to multi-winner settings.
        :return:
        """
        pass

    def rule_score(self):
        """
        Calculate some aggregate score metric over all pref_profiles. Run the evaluation function provided during rule
        setup for the winner(s) of each profile. Calculate the aggregate score over all resulting evaluation values.
        If no aggregation function is provided during rule setup, default to using mean.
        If calculating utility, this might then report the mean utility over all pref_profiles for the current winner on
        each profile.
        :return:
        """
        if "profile_score_aggregation_metric" in self.kwargs:
            agg_metric = self.kwargs["profile_score_aggregation_metric"]
        else:
            agg_metric = np.mean

        all_winners = self.rule_winners()
        all_scores = [self.evaluation_function(idx, winners, profile, **self.kwargs) for idx, (winners, profile) in
                      enumerate(zip(all_winners, self.pref_profiles))]
        return agg_metric(all_scores)

    def energy(self):
        # A lower energy is considered better.
        # We aim to minimize energy and we negate the rule score so rule_score should return a higher
        # value for a better solution.
        energy = -self.rule_score()
        return energy

    def optimize(self, n_steps):
        self.steps = n_steps

        self.updates = n_steps  # call update function at every step; not always necessary

        if self.optimization_method == "annealing":
            vector, sw = self.anneal()
        elif self.optimization_method == "gradient_descent":
            from optimal_voting.gradient_descent import gradient_descent
            vector, sw = gradient_descent(profiles=self.pref_profiles,
                                          utilities=self.kwargs["utility_profiles"],
                                          initial_state=self.kwargs["initial_state"],
                                          opt_target=self.kwargs["gd_opt_target"],
                                          max_n_iterations=n_steps)

        self.post_optimization()
        results_dict = {
            "state": vector,
            "best_energy": sw,
            "history": self.history,
            "voter_map": self.vmap,
            "candidate_map": self.cmap
        }

        if "return_candidate_scores" in self.kwargs and self.kwargs["return_candidate_scores"]:
            cs = [
                vu.score_vector_scores(vector, profile,
                                       normalize=False,
                                       voter_weights=self.kwargs.get("voter_weights", None))
                for profile in self.pref_profiles
            ]
            results_dict["candidate_scores"] = cs

        return results_dict

    def post_optimization(self):
        """
        Run any post optimization tasks, such as saving results to a file.
        :return:
        """
        self.save_history_to_file()

    def save_history_to_file(self):
        if self.history_path is not None:
            df = pd.DataFrame(
                self.history
            )
            df.to_csv(os.path.join(self.history_path, f"{self.job_name}.csv"), index=False)

    def record_history(self, step, energy, temperature):
        if self.history is not None:
            self.history["step"].append(step)
            self.history["temperature"].append(temperature)
            self.history["current_energy"].append(energy)
            self.history["best_energy"].append(copy.copy(self.best_energy))
            self.history["current_state"].append(copy.copy(self.state))
            self.history["best_state"].append(copy.copy(self.best_state))

            self.save_history_to_file()

    def update(self, step, T, E, acceptance, improvement):

        updateWavelength = self.steps / self.num_history_updates
        if (step // updateWavelength) > ((step - 1) // updateWavelength):
            self.record_history(step=step, energy=E, temperature=T)

        if self.verbose:
            elapsed = time.time() - self.start
            if step == 0:
                print(' Temperature        Energy    Accept   Improve     Elapsed   Remaining', file=sys.stderr)
                print('%12.5f  %12.2f                      %s            ' % (T, E, time_string(elapsed)), file=sys.stderr, end='')
                sys.stderr.flush()
                # sys.stdout.flush()
            else:
                remain = (self.steps - step) * (elapsed / step)
                # print("\r", end='')
                print('\r%12.5f  %12.2f  %7.2f%%  %7.2f%%  %s  %s' %
                      (T, E, 100.0 * acceptance, 100.0 * improvement, time_string(elapsed), time_string(remain)), file=sys.stderr, end='')
                sys.stderr.flush()


class PositionalScoringRule(OptimizableRule):

    def __init__(self, eval_func, m, k=None, **kwargs):
        """

        :param pref_profiles: A collection of lists corresponding to each voter's ranking of alternatives.
        :param eval_func: A function which accepts preference profiles and states, and some
        numeric measure of quality for the given states.
        :param m: Total number of alternatives
        :param k: Number of alternatives ranked by each voter. If None, all voters rank all alternatives.
        :param kwargs: May contain items relevant to scoring. E.g. social welfare function, axioms to avoid violating...
        """
        # assert len(pref_profiles) > 0

        if k is None:
            k = m
        self.m = m
        self.k = k

        if "updates_per_step" in kwargs:
            self.updates_per_step = kwargs["updates_per_step"]
        else:
            self.updates_per_step = 1

        if "randomize" in kwargs:
            self.randomized = kwargs["randomize"]
        else:
            self.randomized = False

        if "initial_state" in kwargs and kwargs["initial_state"] is not None:
            state = kwargs["initial_state"]
            if isinstance(state, list):
                state = np.asarray(state)
        else:
            # Start from Borda
            state = np.asarray([k - i - 1 for i in range(k)], dtype=float)

        # normalize initial state
        state = vu.normalize_score_vector(state)

        super().__init__(state=state, eval_func=eval_func, **kwargs)

    def move(self):

        indices = random.sample(range(self.k-1), self.updates_per_step)
        for index in indices:
            # index = random.randint(0, self.m - 1)
            # sign = -1 if bool(random.getrandbits(1)) else 1
            if index > 0:
                # allow small amount of "overlap" with next index to make it possible to actually become equal
                amount = random.uniform(0, (self.state[index - 1] - self.state[index])*1.1)
                amount = min(amount, self.state[index - 1] - self.state[index])
                # amount = random.uniform(0, self.state[index - 1] - self.state[index])
            else:
                amount = random.uniform(0.1, 1)
            # self.state[index] += sign*amount

            # TODO: Could normalize after every step. May improve efficiency.
            # TODO: Add some sort of learning rate to affect size of steps
            self.state[index] += amount

            self.state = vu.normalize_score_vector(self.state)

    def rule_winners(self):
        # Get the output of the rule, as defined by the current state, on each of the pref_profiles
        if all(nw == 1 for nw in self.num_winners):
            winners = [(vu.score_vector_winner(self.state, profile, randomize=self.randomized),) for profile in
                       self.pref_profiles]
        else:
            winners = [tuple(vu.score_vector_ranking(self.state, profile)[:self.num_winners[prof_idx]])
                       for prof_idx, profile in
                       enumerate(self.pref_profiles)]

        return winners


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)


class RandomizedPositionalScoringRule(PositionalScoringRule):
    def __init__(self, pref_profiles, eval_func, m, k=None, **kwargs):
        kwargs["randomize"] = True
        super().__init__(pref_profiles, eval_func, m, k, **kwargs)


class OptimizableSequentialScoringRule(OptimizableRule):
    def __init__(self, pref_profiles, eval_func, m, **kwargs):
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # from SequentialVoting.SequentialVoting import SequentialVoting
        from SequentialVoting.SequentialRule import SequentialScoringRule as ssr

        self.m = m
        if "changes_per_step" in kwargs:
            self.changes_per_step = kwargs["changes_per_step"]
        else:
            self.changes_per_step = 1

        if "initial_state" in kwargs and kwargs["initial_state"] is not None:
            state = kwargs["initial_state"]
            if isinstance(state, list):
                state = np.asarray(state)
        else:
            # Start from Borda
            state = np.asarray([m - i - 1 for i in range(m)], dtype=float)
        # normalize initial state
        state = vu.normalize_score_vector(state)

        self.rule = ssr(score_vector=state, track_winners=False, track_losers=True, tie_break_func=None, verbose=False)

        super().__init__(state=state, pref_profiles=pref_profiles, eval_func=eval_func, **kwargs)

    def move(self):

        indices = random.sample(range(self.m), self.changes_per_step)
        for index in indices:
            if index > 0:
                amount = random.uniform(0, self.state[index - 1] - self.state[index])
            else:
                amount = random.uniform(0.1, 1)

            # TODO: Add some sort of learning rate to affect size of steps
            self.state[index] += amount
            self.state = vu.normalize_score_vector(self.state)
            self.rule.score_vector = self.state

    def rule_winners(self):
        # Get the output of the rule, as defined by the current state, on each of the pref_profiles
        winners = [(self.rule.winner(profile),) for profile in self.pref_profiles]

        return winners


class OptimizableThieleRule(OptimizableRule):

    def __init__(self, n_alternatives, n_winners, pref_profiles, eval_func, **kwargs):

        self.n_alternatives = n_alternatives
        self.n_winners = n_winners

        # TODO: Allow a flexible number of winners? Not so common in theory so probably skip.

        # Create initial state corresponding to one point for each approved alternative
        # State here is a polynomial of size n_alternatives
        if "initial_state" in kwargs and kwargs["initial_state"] is not None:
            state = kwargs["initial_state"]
            if isinstance(state, list):
                state = np.asarray(state)
        else:
            # Create random initial state
            state = [1] + [0] * (n_winners-1)
            state = np.asarray(state)

        if not all(isinstance(prof, abcvoting.preferences.Profile) for prof in pref_profiles): #"approval_profiles" in kwargs:
            raise ValueError("Current implementation requires passing abc_profile instead of ordinal preferences")

        super().__init__(state, pref_profiles, eval_func, **kwargs)

    def move(self):
        current_losers = (~self.state.astype(bool)).nonzero()[0]
        current_winners = self.state.nonzero()[0]

        if len(current_losers) == 0:
            raise ValueError("Array has no zero values")
        if len(current_winners) == 0:
            raise ValueError("Array has no one values")

        # Randomly select a winner/loser and swap them
        random_loser = np.random.choice(current_losers)
        random_winner = np.random.choice(current_winners)

        self.state[random_loser] = 1
        self.state[random_winner] = 0

    def rule_winners(self):
        pass

    def score_of_committee(self, committee, profiles):
        """
        Find the score of the given committee in current state.
        :param committee: A set of proposed winners.
        :param profiles: An abc_voting Profile object.
        :return:
        """
        score = 0
        for voter in profiles:
            n_winners_approved = len(voter.approved & committee)
            score += self.state[:n_winners_approved].sum()
        return score



def _optimize_and_report_score(profiles, utilities, eval_func, profile_score_agg_metric, m, n_steps,
                               initial_state=None):
    rule = PositionalScoringRule(profiles,
                                 eval_func=eval_func,
                                 m=m,
                                 k=None,
                                 initial_state=initial_state,
                                 utilities=utilities,
                                 profile_score_aggregation_metric=profile_score_agg_metric,
                                 keep_history=True,
                                 history_path="../results/annealing_history",
                                 job_name="psr_annealing"
                                 )
    # rule = OptimizableSequentialScoringRule(pref_profiles,
    #                                         eval_func,
    #                                         m,
    #                                         utility_profile=utility_profile,
    #                                         initial_state=initial_state,
    #                                         profile_score_aggregation_metric=profile_score_agg_metric,
    #                                         changes_per_step=1,
    #                                         track_score=True,
    #                                         )
    if n_steps > 0:
        # {
        #     "state": score_vector,
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


class C2ScoringRule(OptimizableRule):

    def __init__(self, pref_profiles, eval_func, **kwargs):

        profiles_clean = []
        self.margin_matrices = []
        for profile in pref_profiles:
            if not isinstance(profile, pref_voting.profiles.Profile):
                profiles_clean.append(pref_voting.profiles.Profile(profile))
            else:
                profiles_clean.append(profile)
            self.margin_matrices.append(weighted_tournament(profiles_clean[-1]))

        # Any state with first value set to 1 should be equivalent to Borda's rule
        # If first val is 0 and second is 0.5 we should have Copeland/Llull's rule
        if "initial_state" in kwargs and kwargs["initial_state"] is not None:
            state = kwargs["initial_state"]
            if isinstance(state, list):
                state = np.asarray(state)
        else:
            # Start from Borda
            state = np.asarray([1, 0], dtype=float)

        super().__init__(state=state,
                         pref_profiles=profiles_clean,
                         eval_func=eval_func,
                         **kwargs)

    def move(self):


        # first index has possible range in [0, 1]
        # second index has most meaningful range in [0, 1]; could be higher or lower though
        index = random.randint(0, len(self.state)-1)

        delta_min = -0.2
        delta_max = 0.2
        amount = random.uniform(delta_min, delta_max)
        self.state[index] += amount
        self.state[index] = min(max(self.state[index], 0), 1)

    def rule_winners(self):
        # def sigmoid(z):
        #     alpha = 100
        #     try:
        #         ret = 1 / (1 + np.exp(-alpha * z))
        #     except Warning as r:
        #         ret = 0
        #     return ret
        #     # return 1 / (1 + np.exp(-alpha*z))

        def sigmoid(x):
            # Safer sigmoid which avoids overflow errors (in practice so far; still technically possible)
            def _positive_sigmoid(z):
                return 1 / (1 + np.exp(-z))

            def _negative_sigmoid(z):
                # Cache exp so you won't have to calculate it twice
                exp = np.exp(z)
                return exp / (exp + 1)
            positive = x >= 0
            # Boolean array inversion is faster than another comparison
            negative = ~positive

            # empty contains junk hence will be faster to allocate
            # Zeros has to zero-out the array after allocation, no need for that
            # See comment to the answer when it comes to dtype
            result = np.empty_like(x, dtype=float)
            result[positive] = _positive_sigmoid(x[positive])
            result[negative] = _negative_sigmoid(x[negative])

            return result

        winners = []
        winners_borda = []
        winners_llull = []
        for wt in self.margin_matrices:
            n_voters = wt[0, 1] + wt[1, 0]
            a, b = self.state[0], self.state[1]
            scores = a*wt + (1-a)*(sigmoid(wt - b*n_voters))
            scores = np.sum(scores, axis=1)
            order = scores.argsort()
            ranks = order.argsort()

            a_borda, b_borda = 1, 0
            scores_borda = a_borda*wt + (1-a_borda)*(sigmoid(wt - b_borda*n_voters))
            scores_borda = np.sum(scores_borda, axis=1)
            order_borda = scores_borda.argsort()
            ranks_borda = order_borda.argsort()

            a_llull, b_llull = 0, 0.5
            scores_llull = a_llull*wt + (1-a_llull)*(sigmoid(wt - b_llull*n_voters))
            scores_llull = np.sum(scores_llull, axis=1)
            order_llull = scores_llull.argsort()
            ranks_llull = order_llull.argsort()

            if (ranks != ranks_llull).any():
                pass
            if (ranks != ranks_borda).any():
                pass
            if (ranks_llull != ranks_borda).any():
                pass

            # TODO: Really need to consider tie-breaking methods. And returning multiple winners.
            # find all tied winners for now, to see if this can find the top cycle. Only return one winner.
            # curr_winners = np.where(scores == scores.max())

            winners.append((np.argmax(scores), ))

        return winners

