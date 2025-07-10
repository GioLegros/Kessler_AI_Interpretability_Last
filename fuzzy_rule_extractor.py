from typing import Dict, Tuple

from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer
import numpy as np

###
# Training patterns should take the format (n, m): n training patterns, each consisting of m attributes
# Rulesets should take the format (i, j): i specifies the rule within the ruleset, j specifies the attribute
# Michigan populations should take the format (i, j, k): i specifies the ruleset, j the rule, k the attribute
###

# Triangular fuzzy.py membership functions. Defined by the center and radius of each triangle.
# Membership is 1 at the center, and decreases linearly to 0 at (center ± radius)
from kesslergame import KesslerGame, KesslerController, Scenario, TrainerEnvironment
from stable_baselines3 import PPO

from src.dummy_controller import PPODummy
from src.envs.radar_env import get_obs
from src.fuzzy import FUZZY_SETS, get_output, heuristic_weights, heuristic_set
from src.navigation_scenario import simple_scenario, scenario_D

BATCH_SIZE = 8_000
EPOCH_SIZE = 10
N_POP = 100
N_RULES = 100
N_ATTR = 24
P_DONT_CARE = 0.90
P_MUTATE = 0.10
P_CROSSOVER = 0.9
N_FUZZY = 15
FUZZY_PD = [P_DONT_CARE] + [(1 - P_DONT_CARE) / (N_FUZZY - 1)] * (N_FUZZY - 1)
THRUST_SCALE, TURN_SCALE = 480.0, 180.0
EXTRACTION_SCALE = 4.0

def fuzzy_rule_extraction(scenario):
    X_train, y_train = get_observation('C:/Users/Eric/XFC_Train/kessler-code/out/No_Ship_Speed/bookmark_9',
                                       scenario)
    p_antecedents, p_outputs, p_weights = init_population(X_train, y_train)
    with open('../out/SimpleScenario/fuzzy.txt', 'w', encoding='UTF-8') as f:
        for epoch in range(30):
            X_train, y_train = get_observation('C:/Users/Eric/XFC_Train/kessler-code/out/No_Ship_Speed/bookmark_9',
                                               scenario)
            fitness = get_fitness(p_antecedents, p_outputs, p_weights, X_train, y_train)
            for i in range(EPOCH_SIZE):
                new_generation_ante = np.zeros_like(p_antecedents)
                new_generation_outputs = np.zeros_like(p_outputs)
                for j in range(N_POP // 2):
                    parent_1_idx = binary_tournament(fitness)
                    parent_2_idx = binary_tournament(fitness)

                    child_1, child_2 = crossover(p_antecedents[parent_1_idx], p_antecedents[parent_2_idx])
                    child_1, child_2 = mutate_antecedents(child_1), mutate_antecedents(child_2)
                    new_generation_ante[j] = child_1
                    new_generation_ante[N_POP - j - 1] = child_2

                    outputs_1, outputs_2 = crossover(p_outputs[parent_1_idx], p_outputs[parent_2_idx])
                    outputs_1 = mutate_floats(outputs_1, minimum=-1, maximum=1)
                    outputs_2 = mutate_floats(outputs_2, minimum=-1, maximum=1)
                    new_generation_outputs[j] = outputs_1
                    new_generation_outputs[N_POP - j - 1] = outputs_2

                    # weights_1, weights_2 = crossover(p_weights[parent_1_idx], p_weights[parent_2_idx])
                    # weights_1 = mutate_floats(weights_1, minimum=0, maximum=1)
                    # weights_2 = mutate_floats(weights_2, minimum=0, maximum=1)
                    # new_generation_weights[j] = weights_1
                    # new_generation_weights[N_POP - j - 1] = weights_2

                new_generation_weights = heuristic_weights(new_generation_ante)
                new_generation_fitness = get_fitness(new_generation_ante, new_generation_outputs, new_generation_weights,
                                                     X_train, y_train)

                stack_fitness = np.concatenate((fitness, new_generation_fitness), axis=0)
                stack_antecedents = np.concatenate((p_antecedents, new_generation_ante), axis=0)
                stack_outputs = np.concatenate((p_outputs, new_generation_outputs), axis=0)
                stack_weights = np.concatenate((p_weights, new_generation_weights), axis=0)

                top_idx = np.argsort(stack_fitness)[:N_POP]
                fitness = stack_fitness[top_idx]
                p_antecedents = stack_antecedents[top_idx]
                p_outputs = stack_outputs[top_idx]
                p_weights = stack_weights[top_idx]

                if i % 1 == 0:
                    print(f'Generation {i}... {np.min(fitness):.8f} {np.mean(fitness):.8f} {np.max(fitness):.8f}')
                    # best_idx = np.argmin(fitness)
                    # get_loss(
                    #     get_output(p_antecedents[best_idx], p_outputs[best_idx], p_weights[best_idx], X_train),
                    #     y_train,
                    #     debug=True,
                    # )
                    f.write(f'{np.min(fitness):.8f} {np.mean(fitness):.8f} {np.max(fitness):.8f}\n')
                    save_population(i, epoch, fitness, p_antecedents, p_outputs, p_weights)

        fresh_antecedents, fresh_outputs, fresh_weights = init_population(X_train, y_train)
        n_replace = N_POP // 2
        n_keep = N_POP - n_replace
        top_idx = np.argsort(fitness)[n_keep]
        p_antecedents = np.concatenate((p_antecedents[top_idx], fresh_antecedents[:n_replace]), axis=0)
        p_outputs = np.concatenate((p_outputs[top_idx], fresh_outputs[:n_replace]), axis=0)
        p_weights = np.concatenate((p_weights[top_idx], fresh_weights[:n_replace]), axis=0)



def save_population(gen, epoch, fitness, p_antecedents, p_outputs, p_weights):
    order = np.argsort(fitness)
    for k in range(5):
        with open(f'../out/SimpleScenario/generation_{epoch}_{gen}_{k}.txt', 'w', encoding='UTF-8') as f2:
            idx = order[k]
            antecedents = p_antecedents[idx]
            for rule in antecedents:
                f2.write('[')
                for attr in rule:
                    f2.write(f'{attr:>2}, ')
                f2.write('],\n')
            f2.write('\n\n')

            outputs = p_outputs[idx]
            for rule in outputs:
                f2.write('[')
                for output in rule:
                    f2.write(f'{output:.4f}, ')
                f2.write('],\n')
            f2.write('\n\n')

            weights = p_weights[idx]
            for rule in weights:
                f2.write('[')
                for weight in rule:
                    f2.write(f'{weight:.4f}, ')
                f2.write('],\n')
            f2.write('\n\n')


def get_observation(model_path, scenario):
    X_train_buffer, y_train_buffer = np.empty(shape=(BATCH_SIZE, N_ATTR)), np.empty(shape=(BATCH_SIZE, 2))
    controller = PPODummy(model_path)
    game = TrainerEnvironment()
    generator = game.run_step(scenario=scenario, controllers=[controller])
    i = 0
    j = 0
    while i < BATCH_SIZE:
        try:
            score, perf_list, game_state = next(generator)
            obs = get_obs(game_state, forecast_frames=30, bumper_range=50, radar_zones=[100, 250, 400])
            action = controller.eval_policy(obs)
            if j == 0:
                X_train_buffer[i] = np.concatenate((obs['radar'], obs['forecast']))
                y_train_buffer[i] = action[0]
                i += 1
            j = (j + 1) % 4
        except StopIteration as exp:
            generator = game.run_step(scenario=scenario, controllers=[controller])

    X_train_buffer = np.minimum(X_train_buffer * EXTRACTION_SCALE, 1)
    return X_train_buffer * EXTRACTION_SCALE, y_train_buffer

def get_random_observation():
    return np.random.random(size=(BATCH_SIZE, N_ATTR)), np.random.random(size=(BATCH_SIZE, 2))

def init_population(X_train, y_train):
    antecedents = np.random.choice(np.arange(N_FUZZY), p=FUZZY_PD, size=(N_POP, N_RULES, N_ATTR))
    outputs = np.random.uniform(low=-1, high=1, size=(N_POP, N_RULES, 2))
    for p in range(N_POP // 2):
        for r in range(N_RULES // 2):
            idx = np.random.randint(low=0, high=X_train.shape[0])
            antecedents[p, r] = heuristic_set(X_train[idx])
            outputs[p, r] = y_train[idx]

#    weights = np.random.uniform(low=0, high=1, size=(N_POP, N_RULES, 1))
    weights = heuristic_weights(antecedents)
    return antecedents, outputs, weights

def get_fitness(population_ante, population_outputs, population_weights, X_train, y_train):
    n_pop = population_ante.shape[0]
    fitness = np.zeros(shape=(n_pop,))
    for i in range(n_pop):
        output_i = get_output(population_ante[i], population_outputs[i], population_weights[i], X_train)
        fitness[i] = get_loss(output_i, y_train)
    return fitness

def get_loss(outputs, y_train, debug=False):
    loss = outputs - y_train
    loss_mse = np.mean(np.square(loss), axis=0)
    if debug:
        print(loss_mse)
    return np.max(loss_mse)

def binary_tournament(fitness):
    idx = np.random.randint(low=0, high=fitness.shape[0], size=(2,))
    if fitness[idx[0]] > fitness[idx[1]]:
        return idx[0]
    return idx[1]


def crossover(parent1, parent2):
    p = np.random.random(size=parent1.shape)
    mask1 = p > P_CROSSOVER
    mask2 = np.logical_not(mask1)

    # Inherit most from parent1
    child1 = np.zeros_like(parent1)
    child1[mask1] = parent1[mask1]
    child1[mask2] = parent2[mask2]

    # Inherit most from parent2
    child2 = np.zeros_like(parent1)
    child2[mask1] = parent2[mask1]
    child2[mask2] = parent1[mask2]

    return child1, child2


def mutate_antecedents(antecedents):
    newchild = np.copy(antecedents)
    mutants = np.random.choice(np.arange(N_FUZZY), p=FUZZY_PD, size=antecedents.shape)
    mutate_mask = np.random.random(size=antecedents.shape) < P_MUTATE
    newchild[mutate_mask] = mutants[mutate_mask]
    return newchild

def mutate_floats(floats, minimum=0, maximum=1):
    newchild = np.copy(floats)
    alterations = np.random.normal(loc=0.0, scale=0.1, size=floats.shape)
    mutate_mask = np.random.random(size=floats.shape) < P_MUTATE
    newchild[mutate_mask] = newchild[mutate_mask] + alterations[mutate_mask]
    newchild = np.maximum(newchild, minimum)
    newchild = np.minimum(newchild, maximum)
    return newchild


from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer

def main():
    scenario = scenario_D(n=32)
    fuzzy_rule_extraction(scenario)
    # X, y = get_observation('C:/Users/Eric/XFC_Train/kessler-code/out/No_Ship_Speed/bookmark_9', scenario)
    # ss = QuantileTransformer()
    # X = ss.fit_transform(X)
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.hist(X, bins=30)
    # plt.show()


if __name__ == '__main__':
    main()
