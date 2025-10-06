from simanneal import Annealer
import data_utils as du
import voting_utils as vu
from OptimizableRule import PositionalScoringRule, OptimizableRule


class ScoreVectorAnnealer(Annealer):

    def __init__(self, optimizable_rule: OptimizableRule, **kwargs):
        super().__init__(optimizable_rule.state)
        self.rule = optimizable_rule

    def energy(self):
        return -self.rule.energy()

    def move(self):
        """
        Adjust the current score vector slightly. Treat the score vector as the state and slightly modify it with
        :return:
        """
        self.state = self.rule.move(self.state)
        # self.state = (self.state - min(self.state)) / max(self.state)





if __name__ == "__main__":
    profiles = du.create_profiles({
        "n_profiles": 50,
        "prefs_per_profile": 20,
        "m": 10,
        "learned_pref_model": "IC",
        # "single_peaked_conitzer",
        # "single_peaked_walsh",
        # "single_crossing"
    })
    profiles = [profile.rankings for profile in profiles]
    utilities = [du._utilities_from_profile(profile) for profile in profiles]

    optimizable_rule = PositionalScoringRule(profiles=profiles,
                                             eval_func=vu.utilitarian_social_welfare,
                                             m=10,
                                             k=None,
                                             utilities=utilities,
                                             changes_per_step=1
                                             )

    tsp = ScoreVectorAnnealer(optimizable_rule)

    tsp.steps = 500

    vector, sw = tsp.anneal()

    print(f"Best vector found: {vector} with welfare: {round(sw, 4)}")
