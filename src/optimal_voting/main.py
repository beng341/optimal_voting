import data_utils as du
import voting_utils as vu
import annealing
import pprint

m = 5

# util_type = "egalitarian"
util_type = "utilitarian"
# util_type = "malfare"
# util_type = "nash_welfare"
offset = 1

total_profiles = 100
preference_models = "all"
# preference_models = ["Single-Crossing"]

# create new dataset and generate utilities
# if not du.data_exists(m=m):
if True:
    du.save_profiles(out_path="data", voters_per_profile=50, num_profiles=100, m=5, preference_models="all", utility_noise=True)
    du.convert_rankings_to_utilities(m=m, preference_models=preference_models)

# print social welfare from some pre-chosen score vectors
# filename = f"{out_path}/profile_data-m={kwargs['m']}-preference_models={kwargs['preference_models']}-utility_noise={kwargs['utility_noise']}-voters_per_profile={kwargs['voters_per_profile']}-num_profiles={kwargs['num_profiles']}.csv"
kwargs = {
    "m": m,
    "preference_models": preference_models,
    "utility_noise": True,
    "voters_per_profile": 20,
    "num_profiles": 20
}
all_utilities = du.load_utility_vectors(offset=1, **kwargs)
all_profiles = du.load_profiles(**kwargs)

vectors = vu.score_vector_examples(m)

all_vector_utilities = dict()

for vec_name, vec in vectors.items():
    # sw = vu.social_welfare_of_score_vector_over_many_profiles(vec, profiles=all_profiles, utilities=all_utilities)
    # print(f"SW for {vec} is {sw}")
    norm_vec = vu.normalize_score_vector(vec)
    sum_sw, mean_sw = vu.social_welfare_of_score_vector_over_many_profiles(norm_vec,
                                                                           profiles=all_profiles,
                                                                           utilities=all_utilities,
                                                                           utility_type=util_type)
    print(f"SW for {vec_name} / {norm_vec} is {sum_sw}")
    # print(f"Mean of SW for {norm_vec} is {sum_sw}")

    all_vector_utilities[tuple(norm_vec)] = sum_sw

annealing_runs = 10
annealing_vector_utilities = dict()
for _ in range(annealing_runs):
    # Set initial state
    vec = [0 for _ in range(m)]
    # if m % 2 == 1:
    #     vec = [1] + [0.9 for _ in range(m//2)] + [1/(2**(idx+1)) for idx in range(m//2)]
    # else:
    #     vec = [1] + [0.9 for _ in range(m//2-1)] + [1 / (2 ** (idx + 1)) for idx in range(m//2)]
    # vec = [(m - idx - 1) ** 2 for idx in range(m)]
    tsp = annealing.ScoreVectorAnnealer(vec, utility_type=util_type, **kwargs)
    tsp.steps = 1000
    vector, sw = tsp.anneal()
    print(f"Best vector found: {vector} with welfare: {round(sw, 4)}")

    # print normalized score vector
    norm_vec = vu.normalize_score_vector(vector)
    norm_sum_sw, norm_mean_sw = vu.social_welfare_of_score_vector_over_many_profiles(norm_vec,
                                                                                     profiles=all_profiles,
                                                                                     utilities=all_utilities,
                                                                                     utility_type=util_type)
    print(f"Normalized summed best value is: {norm_vec} with welfare: {norm_sum_sw}")
    print(f"Normalized mean best value is: {norm_vec} with welfare: {norm_mean_sw}")

    annealing_vector_utilities[tuple(norm_vec)] = norm_sum_sw

all_vector_utilities = {k: v for k, v in sorted(all_vector_utilities.items(), key=lambda item: item[1])}
annealing_vector_utilities = {k: v for k, v in sorted(annealing_vector_utilities.items(), key=lambda item: item[1])}

print(f"Utilities for {preference_models} preferences.")
print("\n----------------")
print("Handmade Vectors and Utilities")
print("----------------")
sorted_d = sorted(all_vector_utilities.items(), key=lambda x: x[1])
pprint.pprint(sorted_d)

print("\n----------------")
print("Annealing Vectors and Utilities")
print("\n----------------")
sorted_d = sorted(annealing_vector_utilities.items(), key=lambda x: x[1])
pprint.pprint(sorted_d)
