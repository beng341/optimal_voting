import copy
import math
import os.path
import random

import numpy as np
import pandas as pd
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

def create_profiles(profiles_descriptions):
    """
    Given appropriate parameters create a list where each entry contains a single profile.
    Should pass in a list of ProfilesDescription namedtuples containing all the parameters required to generate each
    type of profile. Put all generated profiles in a list that gets returned
    :param profiles_descriptions:
    :return:
    """

    profiles = []
    for prof in profiles_descriptions:
        for _ in range(prof.num_profiles):
            if prof.args is None:
                args = {}
            else:
                args = prof.args
            profile = gen_prof(num_voters=prof.num_voters,
                               num_candidates=prof.num_candidates,
                               probmodel=prof.distribution,
                               **args)
            # rankings = profile.rankings
            profiles.append(profile)
    #
    # while len(profiles) < n_profiles:
    #     profile = gen_prof(num_voters=prefs_per_profile, num_candidates=m, probmodel=pref_model, **kwargs)
    #     rankings = profile.rankings
    #     profiles.append(f"{rankings}")
    #     raw_profiles.append(profile)

    return profiles


def _get_preference_models_and_args(preference_model="all", n_profiles=20, num_profiles=10, prefs_per_profile=50, m=5):
    if preference_model == "all":
        preference_model = [
            "Impartial Culture",
            "SP by Conitzer",
            "SP by Walsh",
            "Single-Crossing",
            "1D Uniform",
            "2D Uniform",
            "3D Uniform",
            "5D Uniform",
            "10D Uniform",
            "20D Uniform",
            "2D Sphere",
            "3D Sphere",
            "5D Sphere",
            "Urn",
            "Norm-Mallows",
        ]
    preference_model_short_names = {
        "Impartial Culture": "IC",
        "SP by Conitzer": "single_peaked_conitzer",
        "SP by Walsh": "single_peaked_walsh",
        "Single-Crossing": "single_crossing",
        "1D Uniform": "euclidean",
        "2D Uniform": "euclidean",
        "3D Uniform": "euclidean",
        "5D Uniform": "euclidean",
        "10D Uniform": "euclidean",
        "20D Uniform": "euclidean",
        "2D Sphere": "euclidean",
        "3D Sphere": "euclidean",
        "5D Sphere": "euclidean",
        "Urn": "URN-R",
        "Norm-Mallows": "MALLOWS-RELPHI-R",
    }

    used_models = {pm: preference_model_short_names[pm] for pm in preference_model}

    profiles_per_dist = math.ceil(num_profiles / len(preference_model))
    args = {
        "n_profiles": profiles_per_dist,
        "prefs_per_profile": prefs_per_profile,
        "m": m,
        "learned_pref_model": "",
    }

    all_distribution_details = []

    for model_name, short_name in used_models.items():
        args["learned_pref_model"] = short_name
        kwargs = {}
        if "Sphere" in model_name:
            dimension = model_name.split(" ")[0][:-1]
            kwargs["num_dimensions"] = eval(dimension)
            kwargs["space"] = "uniform_sphere"
        if "Uniform" in model_name:
            dimension = model_name.split(" ")[0][:-1]
            kwargs["num_dimensions"] = eval(dimension)
            kwargs["space"] = "uniform_cube"

        all_distribution_details.append((model_name, copy.copy(args), kwargs))

    return all_distribution_details


def save_profiles(out_path="data", voters_per_profile=50, num_profiles=100, m=5, preference_models="all", utility_noise=True):
    """
    Generate some preference rankings from a variety of distributions.
    :param out_path:
    :param filename:
    :param profiles_per_dist:
    :return:
    """
    if not preference_models == "all" and not isinstance(preference_models, list):
        raise ValueError(f"Gave bad value for preference_model {preference_models}")

    # args = {
    #     "n_profiles": profiles_per_dist,
    #     "prefs_per_profile": 50,
    #     "m": m,
    #     "learned_pref_model": "IC",
    # }
    all_profiles = []
    all_profile_names = []

    for model_name, args, kwargs in _get_preference_models_and_args(preference_model=preference_models, num_profiles=num_profiles, prefs_per_profile=50, m=m):
        dist_name = model_name
        profiles = create_profiles(args=args, **kwargs)
        all_profiles += [profile.rankings for profile in profiles]
        all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]

    # dist_name = "Impartial Culture"
    # args["learned_pref_model"] = "IC"
    # profiles = create_profiles(args=args)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "SP by Conitzer"
    # args["learned_pref_model"] = "single_peaked_conitzer"
    # profiles = create_profiles(args=args)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "SP by Walsh"
    # args["learned_pref_model"] = "single_peaked_walsh"
    # profiles = create_profiles(args=args)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "Single-Crossing"
    # args["learned_pref_model"] = "single_crossing"
    # profiles = create_profiles(args=args)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "1D Uniform"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_cube", num_dimensions=1)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "2D Uniform"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_cube", num_dimensions=2)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "3D Uniform"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_cube", num_dimensions=3)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "5D Uniform"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_cube", num_dimensions=5)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "10D Uniform"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_cube", num_dimensions=10)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "20D Uniform"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_cube", num_dimensions=20)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "2D Sphere"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_sphere", num_dimensions=2)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "3D Sphere"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_sphere", num_dimensions=3)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "5D Sphere"
    # args["learned_pref_model"] = "euclidean"
    # profiles = create_profiles(args=args, space="uniform_sphere", num_dimensions=5)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "Urn"
    # args["learned_pref_model"] = "URN-R"
    # args["n_profiles"] = 4*profiles_per_dist
    # profiles = create_profiles(args=args)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]
    #
    # dist_name = "Norm-Mallows"
    # args["learned_pref_model"] = "MALLOWS-RELPHI-R"
    # args["n_profiles"] = 4*profiles_per_dist
    # profiles = create_profiles(args=args)
    # all_profiles += [profile.rankings for profile in profiles]
    # all_profile_names += [f"{dist_name}_{idx}" for idx in range(len(profiles))]

    # convert the individual rankings to lists rather than tuples to match format of existing data
    final_profiles = []
    for prf in all_profiles:
        new_profile = []
        for rnk in prf:
            new_profile.append(list(rnk))
        final_profiles.append(new_profile)

    df = pd.DataFrame({
        'instance_id': all_profile_names,
        'profile': final_profiles
    })
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    filename = f"profile_data-m={m}-preference_models={preference_models}-utility_noise={utility_noise}-voters_per_profile={voters_per_profile}-num_profiles={num_profiles}.csv"
    df.to_csv(os.path.join(out_path, filename), index=False)


def _utilities_from_profile(profile, normalize_utilities=False):
    """
    Super basic function to create some arbitrary but not incorrect utility_type vector for each ranking in given profile.
    Each utility_type vector sums to 1 like [0.3, 0.1, 0.2, 0.1, 0.3] and the value at index i indicates the utility_type a voter
    gets if alternative i is elected.
    NOTE: A ranking is given in the form [4, 3, 0, 1, 2] indicating 4 is most favourite, 3 2nd favourite, etc.
    :param profile:
    :param normalize_utilities: If True, normalize each voter's utilities to have a min/max of 0/1 respectively.
    Having a different maximum utility value for each voter can affect outcomes (see: malfare sw function)
    :return:
    """
    def _utility_from_ranking(ranking):
        m = len(ranking)

        # new method -- generate random values, assign them to correct rankings
        # utils = np.random.poisson(size=m)
        util_values = np.random.uniform(low=0, high=2, size=m)
        if normalize_utilities:
            util_values = util_values - min(util_values)
            util_values = [u/max(util_values) for u in util_values]
            util_values.sort(reverse=True)
        else:
            util_values = util_values.tolist()
            util_values.sort(reverse=True)

        utilities = [0.0] * m  # put in position i the utility assigned to alternative i
        for i, preference in enumerate(ranking):
            # i is index, preference is the alternative being ranked in position i
            # ex. ranking = [2, 1, 0, 4, 3]
            utilities[preference] = util_values[i]

        return utilities

    all_utility_vectors = []
    for ranking in profile._rankings:
        all_utility_vectors.append(_utility_from_ranking(ranking))

    return all_utility_vectors


def convert_rankings_to_utilities(out_path="data", voters_per_profile=50, num_profiles=100, m=5, preference_models="all", utility_noise=True):
    """

    :return:
    """
    filename = f"{out_path}/profile_data-m={m}-preference_models={preference_models}-utility_noise={utility_noise}-voters_per_profile={voters_per_profile}-num_profiles={num_profiles}.csv"
    df = pd.read_csv(filename)
    profiles = df["profile"].tolist()
    all_utilities = []
    for profile in profiles:
        profile = eval(profile)
        utilities = _utilities_from_profile(profile)
        all_utilities.append(utilities)
    df["utilities"] = all_utilities

    df.to_csv(filename)


def load_utility_vectors(out_path="data", offset=-0, **kwargs):
    """
    Load a pre-existing utility vector. Offset all values by the given amount (added for Nash welfare where being
    above 1 or between 0 and 1 makes a big difference).
    :param m:
    :param data_path:
    :param offset:
    :return:
    """
    df = load_or_make_data(out_path=out_path, **kwargs)
    profiles = df["utilities"].tolist()
    all_utilities = []
    for profile in profiles:
        util_vector = eval(profile)
        # util_vector =
        util_vector = [[offset+u for u in prof] for prof in util_vector]
        all_utilities.append(util_vector)
    return all_utilities


def load_profiles(out_path="data", **kwargs):
    df = load_or_make_data(out_path=out_path, **kwargs)
    profiles = df["profile"].tolist()
    all_profiles = []
    for profile in profiles:
        prf = eval(profile)
        all_profiles.append(prf)
    return all_profiles


def load_or_make_data(out_path="data", **kwargs):
    """

    # :param m:
    # :param data_path:
    # :param preference_models:
    # :param num_profiles:
    # :param voters_per_profile:
    :return:
    """
    if not data_exists(**kwargs):

        save_profiles(**kwargs)
        convert_rankings_to_utilities(**kwargs)

    filename = f"{out_path}/profile_data-m={kwargs['m']}-preference_models={kwargs['preference_models']}-utility_noise={kwargs['utility_noise']}-voters_per_profile={kwargs['voters_per_profile']}-num_profiles={kwargs['num_profiles']}.csv"
    return pd.read_csv(filename)


def data_exists(out_path="data", **kwargs):
    """
    Assume that if the file exists it has the right columns inside it.
    :return:
    """
    filename = f"{out_path}/profile_data-m={kwargs['m']}-preference_models={kwargs['preference_models']}-utility_noise={kwargs['utility_noise']}-voters_per_profile={kwargs['voters_per_profile']}-num_profiles={kwargs['num_profiles']}.csv"
    return os.path.exists(filename)


def default_job_name(**kwargs):
    terms_in_name = ["profile_score_agg_metric", "n_steps"]
    job_name = "annealing-"
    job_name_terms = [f"{k}={v}" for k, v in kwargs.items() if k in terms_in_name]
    return job_name + "-".join(job_name_terms)


if __name__ == "__main__":
    m = 10
    save_profiles(profiles_per_dist=1, m=m)
    convert_rankings_to_utilities(m=m)
