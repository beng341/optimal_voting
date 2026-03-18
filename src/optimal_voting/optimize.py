import OptimizableRule as optr
import numpy as np
import random
import voting_utils as vu
import data_utils as du


def optimize_psr(preference_profiles, eval_fn, steps, **kwargs):

    if "seed" in kwargs:
        seed = kwargs["seed"]
    else:
        seed = None
    np.random.seed(seed)
    random.seed(seed)

    if "utility_profiles" in kwargs:
        utility_profiles = kwargs["utility_profiles"]
    else:
        utility_profiles = None

    if preference_profiles is None:
        assert "utility_profiles" in kwargs, "Must provide one of preference_profiles or utility_profiles."
        preference_profiles = [du.profile_from_utilies(ut) for ut in utility_profiles]

    # ensure that all profiles rank same number of candidates and provide complete rankings
    valid, reason = du.validate_preference_profiles(preference_profiles)
    if not valid:
        raise ValueError(f"Validation of preference profiles failed. Reason given: {reason}")

    n_profiles = len(preference_profiles)
    n_candidates = len(preference_profiles[0][0])

    if isinstance(eval_fn, str):
        eval_fn = vu.get_utility_eval_func_from_str(eval_fn)
    elif callable(eval_fn):
        eval_fn = eval_fn
    else:
        raise ValueError(f"Expected string or callable eval_fn but got: {eval_fn}")

    # default to Borda vector if none provided
    if "initial_state" not in kwargs or kwargs["initial_state"] is None:
        initial_state = [n_candidates - i - 1 for i in range(n_candidates)]
        initial_state = vu.normalize_score_vector(initial_state)
        kwargs["initial_state"] = initial_state

    # Set function used to aggregate SW across profiles
    if "profile_score_agg_metric" not in kwargs:
        kwargs["profile_score_agg_metric"] = np.mean

    job_name = kwargs.get("job_name", du.default_job_name(**kwargs))
    history_path = kwargs.get("out_path", "results/annealing_history")
    save_results = kwargs.get("save_results", True)

    rule = optr.PositionalScoringRule(eval_func=eval_fn,
                                      m=n_candidates,
                                      keep_history=save_results,
                                      history_path=history_path,
                                      job_name=job_name,
                                      pref_profile_lists=preference_profiles,
                                      **kwargs
                                      )

    results_dict = rule.optimize(n_steps=steps)

    return results_dict
