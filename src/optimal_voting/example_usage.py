import data_utils as du
from optimize import optimize_psr
import numpy as np


def example_ordinal_optimization():
    """
    A basic example of the most simple use case: Finding a score vector which maximizes social welfare on a collection
    of ordinal preference profiles.
    :return:
    """
    # Generate ordinal preferences.
    # Several preference distributions can be sampled from using the library, or users can use their own data.
    profiles = du.make_impartial_culture_profiles(
        n_profiles=1000,    # This many profiles will be generated
        n=20,                # Each profile will have rankings n voters
        m=10                # Each ranking should be over exactly m candidates
    )

    # When utility values are not provided they are automatically generated. This can be controlled by the user.
    # There are two methods of generating utilities:
    # 1. utility_type='uniform_random'
    #    For each voter, sample m utility values from Uniform(0, 1).
    #    Assign these to candidates consistent with the voter's ranking. Each voter samples utility independently.
    # 2. utility_type='linear'
    #    Each voter assigns utilities to candidates in a "Borda" style, linearly decreasing from 1 to 0.

    # Set optimization preferences
    steps = 1000
    sw_fn = "egalitarian"
    save_results = False    # Set to True to save the optimization history to file.

    # Perform the optimization
    results_dict = optimize_psr(preference_profiles=profiles,
                                eval_fn=sw_fn,
                                steps=steps,
                                randomize=True,
                                utility_type="linear",
                                save_results=save_results,
                                verbose=True)

    # Format and print results
    best_state = [round(float(s), 4) for s in results_dict['state']]
    print()
    print(f"Best state: {best_state}")
    print(f"Best energy: {results_dict['best_energy']}")


def example_cardinal_optimization():
    """
    A basic example of a simple use case:
    Finding a score vector which maximizes social welfare on a collection of cardinal preference profiles.
    :return:
    """
    # Generate cardinal preferences.
    # This should be a list of lists of lists of floats or the NumPy equivalent.
    # Below we generate 1000 utility profiles, each with 20 voters that have a utility value for 10 candidates
    util_profiles = np.random.uniform(low=0, high=1, size=(100, 20, 10))

    # Set optimization preferences
    steps = 100
    sw_fn = "nash"
    save_results = False  # Set to True to save the optimization history to file.

    # Perform the optimization
    results_dict = optimize_psr(preference_profiles=None,   # Can be None iff 'utility_profiles' is provided
                                eval_fn=sw_fn,
                                steps=steps,
                                save_results=save_results,
                                utility_profiles=util_profiles
                                )

    # Format and print results
    best_state = [round(float(s), 4) for s in results_dict['state']]
    print(f"Best state: {best_state}")
    print(f"Best energy: {results_dict['best_energy']}")


def example_score_vector_analysis():
    """
    Basic example showing the use of the library for analyzing specific score vectors without performing any
    optimization.
    :return:
    """

    # Generate ordinal preferences from some distributions
    n_profiles = 1000
    n = 50
    m = 10

    prefs = {
        "IC": du.make_impartial_culture_profiles(n_profiles=n_profiles, n=n, m=m),
        "SP": du.make_sp_conitzer_profiles(n_profiles=n_profiles, n=n, m=m),
        "Mallows": du.make_mallows_profiles(n_profiles=n_profiles, n=n, m=m)
    }

    vectors = {
        "Borda": [m-i-1 / (m-1) for i in range(m)],
        "Harmonic": [1/(1+i) for i in range(m)],
        "Plurality": [1] + [0 for _ in range(m-1)]
    }

    print(" "*10, end="")
    for d in prefs.keys():
        print("{:<10}".format(d), end=" | ")
    print("")

    for rule_name, rule_vec in vectors.items():
        print(f"{rule_name:<{10}}", end="")
        for dist_name, profiles in prefs.items():
            results_dict = optimize_psr(preference_profiles=profiles,
                                        eval_fn="utilitarian",
                                        steps=0,
                                        save_results=False,
                                        initial_state=rule_vec
                                        )
            best_energy = results_dict['best_energy']
            print("{:<10}".format(round(best_energy, 5)), end=" | ")
        print("")


def example_custom_optimization_target():
    """
    An example showing the use of a custom optimization target. We say the score of a voting rule is equal to m subtract
    one for each candidate not in the
    :return:
    """
    def two_approval_sw(rule_output, pref_profile=None, utility_profile=None, **kwargs):
        """
        A basic example function which returns a social welfare of 1 if presented with the two-approval winner of
        the given profile and returns 0 otherwise.
        The two-approval winner is defined as the candidate most often ranked in one of the top two positions by voters.
        :param rule_output: A tuple (typically) representing the output of the rule being used. May also be a ranking or
        collection of multiple tied winners.
        :param pref_profile: ndarray containing a single preference profile. pref_profile[i, j] = c indicates that
        voter i ranks candidate j in position c.
        :param utility_profile: ndarray containing a utility profile consistent with pref_profile.
        utility_profile[i, j] = u indicates that voter i gets utility u if j is elected
        (utility_profile is unused in this example)
        :param kwargs:
        :return:
        """
        winner = rule_output[0]
        unique, counts = np.unique(pref_profile[:, 0:2], return_counts=True)

        # Uses lexicographic tie-breaking due to nature of np.argmax
        winner_2app = np.argmax(counts)
        winner_2app = unique[winner_2app]

        return winner == winner_2app

    n_profiles = 2000
    n = 20
    m = 10
    profiles = du.make_impartial_culture_profiles(
        n_profiles=n_profiles,  # This many profiles will be generated
        n=n,  # Each profile will have rankings n voters
        m=m  # Each ranking should be over exactly m candidates
    )

    # Set optimization preferences
    steps = 2000
    sw_fn = two_approval_sw
    save_results = False  # Set to True to save the optimization history to file

    # Perform the optimization
    results_dict = optimize_psr(preference_profiles=profiles,
                                eval_fn=sw_fn,
                                steps=steps,
                                save_results=save_results,
                                randomize=True,
                                verbose=True)

    # Format and print results
    best_state = [round(float(s), 4) for s in results_dict['state']]
    print()
    print(f"Best state after optimization: {best_state}")
    print(f"Best energy after optimization: {results_dict['best_energy']}")

    # Compare with actual two-approval utility
    results_dict = optimize_psr(preference_profiles=profiles,
                                eval_fn=sw_fn,
                                steps=0,
                                save_results=save_results,
                                initial_state=[1, 1] + [0] * (m-2))

    # Format and print results
    best_state = [round(float(s), 4) for s in results_dict['state']]
    print()
    print(f"Two approval vector: {best_state}")
    print(f"Two approval energy: {results_dict['best_energy']}")





if __name__ == "__main__":

    # example_ordinal_optimization()

    # example_cardinal_optimization()

    # example_score_vector_analysis()

    example_custom_optimization_target()