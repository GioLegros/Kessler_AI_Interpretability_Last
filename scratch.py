from typing import Dict, Tuple

import numpy as np

from kesslergame import KesslerController, TrainerEnvironment, KesslerGame

from src.envs.radar_env import get_radar
from src.lib import parse_game_state
from src.navigation_scenario import scenario_D, scenario_E, scenario_F, simple_scenario

# Inputs
# 12 point radar
# Outputs:
# Hard Left, Left, Straight, Right, Hard Right


FUZZY_SETS = [
    # Triangular fuzzy sets can be defined as functions of their center and radius
    lambda x: np.ones_like(x),  # Don't care
    lambda x: np.maximum(1 - np.abs((0.000 - x) / 0.125), 0),  # Low
    lambda x: np.maximum(1 - np.abs((0.125 - x) / 0.125), 0),  # Medium-Low
    lambda x: np.maximum(1 - np.abs((0.250 - x) / 0.125), 0),  # Medium
    lambda x: np.maximum(1 - np.abs((0.375 - x) / 0.125), 0),  # Medium-High
    lambda x: np.maximum(1 - np.abs((0.500 - x) / 0.125), 0),  # High
    lambda x: (x < 0.01).astype(int),  # Clear
    lambda x: (x > 0.01).astype(int),  # Dangerous
]

N_FUZZY_SETS = 8
N_ATTR = 12
N_OUT = 2
N_OUT_VALS = 5
N_RULESETS = 100
STARTING_RULES_PER_RULESET = 50
MAX_RULES_PER_RULESET = 120

def ga():
    population = []
    for i in range(N_RULESETS):
        antecedents = np.random.randint(low=0, high=N_FUZZY_SETS, size=(STARTING_RULES_PER_RULESET, N_ATTR))
        outputs = np.random.randint(low=0, high=N_OUT_VALS, size=(STARTING_RULES_PER_RULESET, N_OUT))
        rules = np.concatenate((antecedents, outputs), axis=1)
        population.append(
            encode_rules(rules)
        )
    population_fitness = np.array([
        get_fitness(ruleset) for ruleset in population
    ])
    with open('../out/ga/output.txt', 'w', encoding='UTF-8') as out_f:
        for generation in range(1000):
            print(f'{np.min(population_fitness):.8f} {np.mean(population_fitness):.8f} {np.max(population_fitness):.8f}')
            out_f.write(f'{np.min(population_fitness):.8f},{np.mean(population_fitness):.8f},{np.max(population_fitness):.8f}\n')
            save_population(population, population_fitness, fn=f'../out/ga/population_gen_{generation}.txt')
            parent_A_idx = binary_tournament(population_fitness)
            parent_B_idx = binary_tournament(population_fitness)
            children = []
            for child_idx in range(N_RULESETS):
                child = crossover(population[parent_A_idx], population[parent_B_idx])
                child = mutate(child)
                child = trim(child)
                child = np.unique(child)
                children.append(child)

            children_fitness = np.array([
                get_fitness(ruleset) for ruleset in children
            ])
            best_parents = np.argsort(-1 * population_fitness)[:N_RULESETS // 2]
            best_children = np.argsort(-1 * children_fitness)[:N_RULESETS // 2]
            next_generation = []
            next_generation_fitness = []
            for i in range(N_RULESETS // 2):
                winner_parent_idx = best_parents[i]
                winner_child_idx = best_children[i]

                next_generation.append(population[winner_parent_idx])
                next_generation.append(children[winner_child_idx])

                next_generation_fitness.append(population_fitness[winner_parent_idx])
                next_generation_fitness.append(children_fitness[winner_child_idx])
            population = next_generation
            population_fitness = np.array(next_generation_fitness)
        print(f'{np.min(population_fitness):.8f} {np.mean(population_fitness):.8f} {np.max(population_fitness):.8f}')
        out_f.write(
            f'{np.min(population_fitness):.8f},{np.mean(population_fitness):.8f},{np.max(population_fitness):.8f}\n')


def save_population(population, pop_fitness, fn):
    order = np.argsort(-1 * pop_fitness)
    with open(fn, 'w', encoding='UTF-8') as f:
        for i in range(len(population)):
            idx = order[i]
            ruleset = population[idx]
            decoded = decode_rules(ruleset)
            for rule in decoded:
                f.write('[')
                for antecedent in rule:
                    f.write(f'{antecedent}, ')
                f.write('],\n')
            f.write('\n\n')


def get_fitness(ruleset):
    game = TrainerEnvironment()

    n_trials = 5
    total_time_alive = 0
    total_bonus = 0
    benchmark_scenarios = [scenario_D, scenario_E, scenario_F]
    for scenario in benchmark_scenarios:
        for j in range(n_trials):
            controller = FuzzyController(decode_rules(ruleset))
            score, _, __ = game.run(scenario=scenario(seed=j), controllers=[controller])
            total_time_alive += score.sim_time
            total_bonus += controller.get_bonus()

    n_scenarios = len(benchmark_scenarios)
    return (total_time_alive + total_bonus) / (n_trials * n_scenarios)


def binary_tournament(fitness_values):
    n = len(fitness_values)
    contestant_A = np.random.randint(low=0, high=n)
    contestant_B = np.random.randint(low=0, high=n)
    if fitness_values[contestant_A] > fitness_values[contestant_B]:
        return contestant_A
    return contestant_B

def crossover(parent_A, parent_B):
    new_rules = []
    # Rules in both will always be in the child. Likewise, rules in neither will never be in the child.
    new_rules.extend(np.intersect1d(parent_A, parent_B))

    # There is *probably* a more efficient way!
    # Take 90% of the rules which are only in parent A (pick A's "1" bit 90% of the time, else B's "0" bit)
    in_only_parent_A = np.setdiff1d(parent_A, parent_B)
    rands = np.random.random(size=in_only_parent_A.shape) > 0.1
    new_rules.extend(in_only_parent_A[rands])

    in_only_parent_B = np.setdiff1d(parent_B, parent_A)
    rands = np.random.random(size=in_only_parent_B.shape) > 0.9
    new_rules.extend(in_only_parent_B[rands])

    return np.array(new_rules, dtype=np.int64)

def mutate(subject):
    # Pick 10 fuzzy rules and add them to the ruleset.
    # Theoretically, we should remove the ones which already exist, but that is a very rare case...
    antecedents = np.random.randint(low=0, high=N_FUZZY_SETS, size=(10, N_ATTR))
    outputs = np.random.randint(low=0, high=N_OUT_VALS, size=(10, N_OUT))
    new_rules = np.concatenate((antecedents, outputs), axis=1)
    return np.union1d(subject, encode_rules(new_rules))

def trim(ruleset):
    if ruleset.shape[0] > MAX_RULES_PER_RULESET:
        return np.random.choice(ruleset, size=(MAX_RULES_PER_RULESET,), replace=False)
    return ruleset

def rule_weights(rules):
    # In this case, all fuzzy sets have the same area
    nonzero = np.count_nonzero(rules, axis=-1)
    a = 0.5 ** nonzero
    return 1 - a

def get_compatibility(decoded_rule, observation):
    antecedents = decoded_rule[:, :-2]
    ob2 = np.broadcast_to(observation, shape=antecedents.shape)
    compat = np.ones_like(antecedents, dtype=np.float64)
    for i in range(4):
        mask = antecedents == i
        lmb = FUZZY_SETS[i]
        compat[mask] = lmb(ob2[mask])
    return np.prod(compat, axis=-1)

def get_output(ruleset, observation):
    compatibility = get_compatibility(ruleset, observation)
    weights = rule_weights(ruleset)
    weighted_compatibility = compatibility * weights
    winner_idx = np.argmax(weighted_compatibility)
    return ruleset[winner_idx, -2:]

def encode_rules(rule):
    rule_length = rule.shape[-1]
    encoder = np.power(16, np.arange(rule_length), dtype=np.int64)
    return np.sum(rule * encoder, axis=-1, dtype=np.int64)

def decode_rules(rule_idx):
    rule_length = N_ATTR + N_OUT
    n_rules = rule_idx.shape[0]
    rule = np.empty(shape=(n_rules, rule_length), dtype=np.int8)
    for i in range(rule_length):
        val = rule_idx & 15
        rule[:, i] = val
        rule_idx = rule_idx >> 4
    return rule

def parse_output(fuzzy_output):
    thrust_out = fuzzy_output[0]  # Range 0 - 5
    thrust = (thrust_out - 2) * 240

    turn_out = fuzzy_output[1]  # Range 0 - 5
    turn = (turn_out - 2) * 90  # Range (-2, 2) * 90 = (-180, 180)

    return thrust, turn


class FuzzyController(KesslerController):
    def __init__(self, ruleset):
        self.rules = ruleset
        self.bonus = 0

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        state = parse_game_state(ship_state, game_state)
        radar = get_radar(state['asteroids']['polar_positions'], state['asteroids']['radii'], [100, 250, 400])
        radar[radar > 0.5] = 0.5

        threshold = 0.01
        bonus = 0.9 ** np.count_nonzero(radar > threshold)
        self.bonus += (bonus / 30)

        fuzzy_output = get_output(self.rules, radar)
        thrust, turn = parse_output(fuzzy_output)

        return thrust, turn, False, False

    def get_bonus(self):
        return self.bonus

    @property
    def name(self) -> str:
        return "Please Work"
    pass

def exp():
    ruleset = np.array([
        [5, 7, 3, 6, 5, 4, 5, 2, 7, 5, 5, 0, 0, 0, ],
        [2, 7, 2, 6, 1, 1, 7, 7, 3, 5, 5, 1, 0, 0, ],
        [4, 0, 7, 0, 7, 5, 5, 0, 2, 7, 7, 1, 0, 0, ],
        [7, 3, 5, 6, 5, 0, 6, 7, 0, 4, 3, 2, 0, 0, ],
        [0, 2, 3, 1, 5, 0, 1, 7, 7, 4, 0, 5, 0, 0, ],
        [5, 5, 5, 5, 4, 7, 1, 1, 6, 4, 6, 5, 0, 0, ],
        [1, 4, 7, 1, 3, 6, 7, 5, 6, 4, 6, 5, 0, 0, ],
        [2, 4, 5, 2, 5, 0, 6, 5, 3, 0, 3, 0, 1, 0, ],
        [6, 4, 3, 6, 3, 5, 2, 5, 6, 7, 4, 2, 1, 0, ],
        [7, 2, 2, 3, 3, 3, 7, 7, 4, 1, 1, 6, 1, 0, ],
        [5, 3, 4, 1, 7, 5, 3, 3, 0, 5, 2, 6, 1, 0, ],
        [0, 3, 0, 7, 5, 6, 0, 0, 7, 4, 7, 6, 1, 0, ],
        [1, 2, 3, 5, 6, 4, 3, 7, 5, 5, 2, 2, 2, 0, ],
        [6, 1, 1, 1, 3, 7, 2, 5, 6, 4, 0, 3, 2, 0, ],
        [0, 1, 4, 0, 6, 6, 2, 7, 1, 1, 7, 4, 2, 0, ],
        [5, 4, 7, 0, 5, 0, 4, 3, 2, 2, 7, 6, 2, 0, ],
        [7, 5, 2, 3, 2, 4, 7, 7, 1, 0, 2, 0, 3, 0, ],
        [6, 3, 5, 3, 7, 6, 4, 5, 1, 1, 2, 0, 3, 0, ],
        [1, 7, 4, 1, 5, 1, 0, 5, 7, 1, 0, 1, 3, 0, ],
        [5, 3, 5, 3, 6, 0, 2, 4, 3, 7, 2, 1, 3, 0, ],
        [0, 1, 1, 2, 3, 4, 0, 6, 0, 7, 5, 2, 3, 0, ],
        [6, 1, 1, 6, 3, 1, 5, 3, 1, 5, 7, 2, 3, 0, ],
        [7, 5, 4, 6, 2, 6, 4, 5, 3, 1, 1, 5, 3, 0, ],
        [5, 7, 4, 3, 1, 6, 6, 2, 1, 3, 3, 5, 3, 0, ],
        [4, 4, 5, 1, 3, 7, 6, 3, 1, 6, 1, 6, 3, 0, ],
        [1, 6, 3, 5, 2, 3, 3, 7, 5, 1, 6, 7, 3, 0, ],
        [0, 5, 4, 0, 3, 0, 5, 4, 0, 3, 0, 0, 4, 0, ],
        [7, 0, 1, 5, 7, 6, 5, 3, 0, 0, 4, 3, 4, 0, ],
        [4, 3, 1, 7, 7, 2, 1, 7, 7, 7, 2, 4, 4, 0, ],
        [6, 1, 5, 3, 1, 0, 4, 6, 1, 5, 0, 5, 4, 0, ],
        [3, 5, 6, 7, 7, 6, 0, 3, 2, 7, 4, 7, 4, 0, ],
        [5, 4, 3, 0, 7, 6, 1, 3, 4, 5, 3, 3, 0, 1, ],
        [2, 0, 5, 7, 0, 6, 5, 1, 0, 7, 6, 3, 0, 1, ],
        [6, 6, 1, 5, 7, 4, 2, 1, 6, 6, 4, 4, 0, 1, ],
        [2, 3, 6, 6, 7, 2, 7, 3, 3, 4, 5, 4, 0, 1, ],
        [1, 6, 5, 3, 4, 2, 0, 2, 7, 5, 7, 4, 0, 1, ],
        [5, 6, 5, 7, 7, 7, 6, 7, 3, 4, 6, 6, 0, 1, ],
        [5, 1, 0, 5, 4, 2, 6, 6, 3, 3, 4, 0, 1, 1, ],
        [5, 5, 1, 4, 2, 3, 5, 3, 3, 6, 7, 2, 1, 1, ],
        [2, 2, 7, 2, 5, 6, 6, 1, 1, 0, 4, 4, 1, 1, ],
        [2, 7, 1, 2, 3, 1, 3, 1, 4, 3, 1, 0, 2, 1, ],
        [2, 1, 3, 5, 1, 0, 0, 1, 0, 2, 1, 1, 2, 1, ],
        [2, 5, 5, 4, 5, 3, 7, 6, 3, 2, 0, 2, 2, 1, ],
        [0, 4, 6, 7, 7, 4, 7, 5, 4, 0, 7, 2, 2, 1, ],
        [6, 1, 3, 3, 7, 5, 2, 7, 2, 3, 0, 3, 2, 1, ],
        [3, 7, 5, 6, 4, 2, 1, 1, 7, 3, 5, 3, 2, 1, ],
        [7, 5, 2, 0, 4, 0, 5, 4, 2, 7, 0, 4, 2, 1, ],
        [1, 5, 4, 2, 1, 1, 6, 2, 2, 4, 4, 4, 2, 1, ],
        [7, 0, 3, 2, 5, 2, 7, 7, 1, 2, 6, 4, 2, 1, ],
        [2, 7, 2, 0, 0, 3, 1, 7, 3, 6, 4, 6, 2, 1, ],
        [0, 6, 1, 4, 7, 3, 2, 3, 2, 6, 1, 3, 3, 1, ],
        [7, 4, 1, 4, 7, 6, 6, 7, 1, 6, 2, 3, 3, 1, ],
        [5, 6, 3, 4, 4, 3, 4, 0, 4, 0, 6, 3, 3, 1, ],
        [5, 4, 1, 7, 7, 1, 3, 4, 6, 4, 2, 4, 3, 1, ],
        [3, 6, 5, 0, 1, 0, 7, 4, 4, 4, 0, 7, 3, 1, ],
        [5, 6, 5, 1, 4, 0, 6, 6, 5, 5, 5, 4, 4, 1, ],
        [6, 0, 2, 4, 1, 5, 3, 2, 7, 1, 3, 6, 4, 1, ],
        [1, 0, 4, 0, 2, 1, 3, 1, 2, 1, 1, 1, 0, 2, ],
        [2, 1, 3, 3, 7, 3, 5, 1, 0, 0, 7, 1, 0, 2, ],
        [1, 0, 3, 2, 1, 4, 4, 3, 4, 4, 2, 2, 0, 2, ],
        [1, 2, 6, 3, 3, 5, 7, 4, 5, 4, 4, 3, 0, 2, ],
        [7, 5, 6, 3, 6, 1, 3, 5, 7, 4, 7, 1, 2, 2, ],
        [0, 0, 4, 2, 5, 1, 5, 6, 1, 7, 7, 1, 2, 2, ],
        [3, 6, 7, 1, 1, 0, 1, 3, 5, 4, 1, 2, 2, 2, ],
        [0, 0, 6, 1, 5, 6, 0, 0, 4, 0, 7, 2, 2, 2, ],
        [4, 0, 2, 6, 2, 5, 7, 3, 1, 0, 6, 6, 2, 2, ],
        [0, 2, 4, 6, 6, 4, 2, 3, 0, 3, 6, 7, 2, 2, ],
        [1, 7, 7, 6, 0, 4, 1, 6, 6, 3, 1, 1, 3, 2, ],
        [5, 4, 1, 0, 7, 7, 4, 0, 3, 3, 7, 6, 3, 2, ],
        [3, 0, 2, 7, 4, 4, 2, 5, 5, 2, 3, 1, 4, 2, ],
        [1, 3, 7, 5, 2, 3, 3, 1, 3, 0, 5, 3, 4, 2, ],
        [7, 6, 1, 3, 1, 2, 6, 0, 7, 3, 5, 3, 4, 2, ],
        [0, 2, 7, 1, 3, 3, 3, 7, 0, 1, 1, 5, 4, 2, ],
        [2, 5, 3, 6, 7, 7, 2, 6, 0, 1, 3, 5, 4, 2, ],
        [2, 3, 3, 7, 7, 1, 7, 7, 4, 0, 7, 0, 0, 3, ],
        [3, 0, 2, 0, 3, 0, 7, 2, 2, 0, 4, 1, 0, 3, ],
        [1, 3, 4, 0, 4, 1, 4, 4, 3, 7, 5, 2, 0, 3, ],
        [5, 2, 5, 3, 1, 3, 2, 1, 3, 4, 1, 3, 0, 3, ],
        [6, 4, 7, 0, 3, 6, 6, 0, 3, 4, 4, 4, 0, 3, ],
        [6, 1, 7, 4, 3, 0, 5, 0, 3, 5, 6, 4, 0, 3, ],
        [0, 7, 7, 1, 0, 7, 2, 6, 3, 0, 2, 0, 1, 3, ],
        [7, 4, 6, 7, 2, 3, 3, 0, 3, 6, 3, 6, 1, 3, ],
        [5, 5, 6, 1, 1, 3, 7, 6, 6, 3, 6, 1, 2, 3, ],
        [1, 1, 0, 5, 4, 2, 4, 7, 1, 7, 2, 3, 2, 3, ],
        [0, 0, 6, 2, 4, 6, 0, 0, 2, 7, 6, 3, 2, 3, ],
        [6, 5, 1, 5, 2, 1, 4, 6, 0, 6, 0, 4, 2, 3, ],
        [1, 3, 1, 0, 4, 6, 1, 1, 7, 5, 6, 4, 2, 3, ],
        [2, 4, 2, 5, 1, 6, 2, 7, 6, 0, 7, 4, 2, 3, ],
        [1, 1, 1, 6, 5, 4, 6, 5, 7, 3, 4, 5, 2, 3, ],
        [2, 7, 3, 7, 4, 3, 0, 7, 2, 4, 4, 5, 2, 3, ],
        [2, 2, 7, 1, 5, 4, 0, 2, 4, 1, 4, 6, 2, 3, ],
        [2, 7, 7, 0, 7, 0, 6, 3, 3, 7, 2, 7, 2, 3, ],
        [6, 1, 6, 7, 4, 2, 5, 7, 4, 4, 4, 0, 3, 3, ],
        [3, 3, 4, 7, 4, 6, 6, 0, 0, 3, 7, 3, 3, 3, ],
        [3, 2, 6, 7, 3, 1, 0, 0, 5, 0, 5, 4, 3, 3, ],
        [4, 0, 5, 6, 0, 2, 2, 3, 7, 5, 2, 5, 3, 3, ],
        [2, 4, 2, 4, 4, 6, 4, 0, 1, 0, 4, 4, 4, 3, ],
        [2, 0, 5, 5, 4, 3, 6, 0, 0, 0, 3, 5, 4, 3, ],
        [4, 2, 1, 0, 0, 6, 3, 7, 3, 0, 3, 5, 4, 3, ],
        [7, 1, 4, 7, 1, 6, 5, 5, 0, 7, 6, 6, 4, 3, ],
        [5, 6, 5, 5, 2, 6, 3, 7, 4, 1, 7, 7, 4, 3, ],
        [7, 4, 7, 1, 6, 0, 3, 4, 2, 6, 5, 1, 0, 4, ],
        [1, 1, 1, 6, 0, 0, 2, 2, 1, 0, 4, 3, 0, 4, ],
        [2, 1, 4, 3, 2, 1, 4, 0, 4, 4, 0, 6, 0, 4, ],
        [4, 2, 2, 0, 0, 4, 7, 7, 3, 2, 5, 7, 0, 4, ],
        [4, 1, 4, 4, 7, 1, 0, 4, 1, 7, 6, 0, 1, 4, ],
        [3, 1, 6, 4, 4, 4, 7, 4, 3, 0, 6, 3, 1, 4, ],
        [1, 5, 1, 6, 5, 3, 1, 6, 4, 1, 5, 6, 1, 4, ],
        [1, 4, 1, 4, 4, 2, 2, 3, 3, 6, 1, 7, 1, 4, ],
        [3, 2, 5, 3, 0, 2, 3, 7, 4, 0, 7, 0, 2, 4, ],
        [0, 0, 5, 4, 3, 3, 1, 7, 5, 3, 3, 2, 2, 4, ],
        [7, 4, 5, 0, 1, 6, 4, 5, 1, 3, 4, 3, 2, 4, ],
        [0, 4, 6, 0, 7, 1, 3, 1, 3, 5, 2, 7, 2, 4, ],
        [5, 0, 0, 7, 0, 1, 1, 6, 4, 6, 0, 0, 3, 4, ],
        [4, 5, 3, 1, 3, 1, 3, 5, 7, 5, 1, 3, 3, 4, ],
        [5, 4, 5, 0, 6, 5, 3, 0, 1, 7, 7, 5, 3, 4, ],
        [1, 6, 4, 4, 4, 6, 6, 0, 5, 5, 3, 0, 4, 4, ],
        [6, 2, 3, 4, 5, 6, 7, 3, 3, 4, 4, 0, 4, 4, ],
        [6, 5, 4, 3, 0, 4, 3, 3, 2, 7, 1, 6, 4, 4, ],
        [3, 5, 5, 5, 1, 2, 2, 5, 2, 5, 7, 7, 4, 4, ],
    ])
    controller = FuzzyController(ruleset=ruleset)
    scenario = simple_scenario()
    game = KesslerGame()
    game.run(scenario=scenario_E(n=12), controllers=[controller])


def main():
    # antecedents = np.random.randint(low=0, high=N_FUZZY_SETS, size=(5, N_ATTR))
    # outputs = np.random.randint(low=0, high=N_OUT_VALS, size=(5, N_OUT))
    # rules = np.concatenate((antecedents, outputs), axis=1)
    # enc = encode_rules(rules)
    # print(rules)
    # print(enc)
    # print(decode_rules(enc))
    # for i in range(100):
    #     antecedents = np.random.randint(low=0, high=N_FUZZY_SETS, size=(50, N_ATTR))
    #     outputs = np.random.randint(low=0, high=N_OUT_VALS, size=(50, N_OUT))
    #     rules = np.concatenate((antecedents, outputs), axis=1)
    #     enc = encode_rules(rules)
    #     dec = decode_rules(enc)
    #     if np.any(rules != dec):
    #         print("warning")
    exp()
#    ga()


if __name__ == '__main__':
    main()
