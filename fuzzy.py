import numpy as np

FUZZY_SETS = np.array([
    [0.0, 0.0],  # 0: Placeholder -- Don't care
    [0.0, 1.0],  # 1: Low-1
    [1.0, 1.0],  # 2: High-1
    [0.0, 0.5],  # 3: Low-2
    [0.5, 0.5],  # 4: Medium-2
    [1.0, 0.5],  # 5: High-2
    [0.0, 1 / 3],  # 6: Low-3
    [1 / 3, 1 / 3],  # 7: Medium-Low-3
    [2 / 3, 1 / 3],  # 8: Medium-High-3
    [1.0, 1 / 3],  # 9: High-3
    [0.00, 0.25],  # 10: Low-4
    [0.25, 0.25],  # 11: Medium-Low-4
    [0.50, 0.25],  # 12: Medium-4
    [0.75, 0.25],  # 13: Medium-High-4
    [1.00, 0.25],  # 14: High-4
])


def get_membership(cw, x):
    """
    Membership of the value 'x' into the fuzzy.py set
    :param cw: The (center, width) definition of the fuzzy.py set
    :param x: The x value to test
    :return: The membership of x
    """
    value = 1 - np.abs((cw[0] - x) / cw[1])
    return np.maximum(value, 0)

def get_compatibility_population(population, X_train):
    """
    Compatibility between an entire Pittsburgh-style population and all training patterns.
    This implementation creates a large matrix in memory and uses vector calculations for faster calculation.
    For a large training set, the calculation is 50-100x faster, but there is a high memory requirement.
    :param population: An (N_POP, N_RULES, N_ATTR) array
    :param X_train: An (N_INPUT, N_ATTR) array
    :return: An (N_POP, N_RULE, N_INPUT) array [i, j, k]:
            The compatibility of ruleset i, rule index j, and training pattern j
    """
    # 1. Create a 4D array, [i, j, n, m]
    #    Ruleset: i, Rule: j, Pattern n, Attribute: m
    #    .. i.e. [i, j, n, m] = membership(X_train[n, m], population[i, j, m])
    # 2. For each possible fuzzy.py set, create a bitmask (i.e. identify which attributes are matched to that fuzzy.py set)
    # 3. Calculate + save the membership values for (input * attribute) pairs matched to that fuzzy.py set
    # 4. Take the product along the 'm' axis (the attributes) to create a 3D array of ruleset x rule x patterns
    n_pop, n_rules = population.shape[0], population.shape[1]
    n_input, n_attr = X_train.shape[0], X_train.shape[1]
    compat = np.ones(shape=(n_pop, n_rules, n_input, n_attr))
    X_train_bc = np.broadcast_to(X_train[np.newaxis, np.newaxis, :, :], compat.shape)
    for i in range(1, 15):
        mask = population == i
        mask = np.broadcast_to(mask[:, :, np.newaxis, :], shape=compat.shape)
        fuzzy_set = FUZZY_SETS[i]
        membership = np.maximum(1 - np.abs((fuzzy_set[0] - X_train_bc[mask]) / fuzzy_set[1]), 0)
        compat[mask] = membership
    return np.prod(compat, axis=3)


def get_compatibility_ruleset(ruleset, X_train):
    """
    Compatibility between an entire ruleset and all training patterns
    :param ruleset: An (N_RULES, N_ATTR) array
    :param X_train: An (N_INPUT, N_ATTR) array
    :return: An (N_RULE, N_INPUT) array [i, j]: The compatibility of rule i and training pattern j
    """
    # 1. Create a 3D array, [i, j, k]
    #    Rule: i, Pattern: j, Attribute: k .. i.e. [i, j, k] = membership(X_train[j, k], ruleset[i, k])
    # 2. For each possible fuzzy.py set, create a bitmask (i.e. identify which attributes are matched to that fuzzy.py set)
    # 3. Calculate + save the membership values for (input * attribute) pairs matched to that fuzzy.py set
    # 4. Take the product along the k axis (the attributes) to create a 2D array of rule x patterns
    n_rules, n_input, n_attr = ruleset.shape[0], X_train.shape[0], X_train.shape[1]
    compat = np.ones(shape=(n_rules, n_input, n_attr))
    X_train_bc = np.broadcast_to(X_train, compat.shape)
    for i in range(1, 15):
        mask = ruleset == i
        mask = np.broadcast_to(mask[:, np.newaxis, :], shape=compat.shape)
        fuzzy_set = FUZZY_SETS[i]
        membership = np.maximum(1 - np.abs((fuzzy_set[0] - X_train_bc[mask]) / fuzzy_set[1]), 0)
        compat[mask] = membership
    return np.prod(compat, axis=2)


def get_compatibility_rule(rule, X_train):
    """
    Compatibility between a single rule and all training patterns
    :param rule:
    :param X_train:
    :return:
    """
    # 1. Create a 2D array, [i, j]
    #   (The membership of pattern 'i', attribute 'j' into the fuzzy.py set specified by the rule)
    # 2. For each possible fuzzy.py set, create a bitmask (i.e. identify which attributes are matched to that fuzzy.py set)
    # 3. Calculate + save the membership values for (input * attribute) pairs matched to that fuzzy.py set
    # 4. To get the compatibility of each rule with each pattern, take the product along each column
    compat = np.ones_like(X_train)
    for i in range(1, 15):
        mask = np.broadcast_to(rule == i, shape=X_train.shape)
        fuzzy_set = FUZZY_SETS[i]
        membership = np.maximum(1 - np.abs((fuzzy_set[0] - X_train[mask]) / fuzzy_set[1]), 0)
        compat[mask] = membership
    return np.prod(compat, axis=1)

def get_output(rule_ante, rule_outputs, rule_weights, X_train):
    weighted_compat = get_compatibility_ruleset(rule_ante, X_train) * rule_weights
    winner = np.argmax(weighted_compat, axis=0)
    return rule_outputs[winner]

    # outputs = np.matmul(weighted_compat.T, rule_outputs)
    # total_weights = np.sum(weighted_compat, axis=0)
    # # If no rule has any compatibility for a certain training pattern, output zero
    # no_compat_mask = total_weights == 0
    # total_weights[no_compat_mask] = np.inf
    # return outputs / total_weights[:, np.newaxis]

# A really simple heuristic weighting scheme.
# Don't care = 0 points
# Fuzzy sets = 2, 3, 4, or 5 points
def heuristic_weights(rule_ante):
    score_1 = (rule_ante > 0).astype(int)
    score_2 = (rule_ante > 2).astype(int)
    score_3 = (rule_ante > 5).astype(int)
    score_4 = (rule_ante > 9).astype(int)
    s = np.sum(score_1 + score_2 + score_3 + score_4, axis=-1)
    return s[:, :, np.newaxis]

def heuristic_set(inputs, n=3):
    heuristic_rule = np.zeros_like(inputs)
    nonzero = np.nonzero(inputs)[0]
    n_nonzero = nonzero.shape[0]
    n_pick = min(n_nonzero, n)
    attr_idx = np.random.choice(nonzero, size=(n_pick,), replace=False)
    for i in attr_idx:
        best_idx, best_compat = 0, 0
        for j in range(1, FUZZY_SETS.shape[0]):
            compat = get_membership(FUZZY_SETS[j], inputs[i])
            if compat > best_compat:
                best_idx = j
                best_compat = compat
        heuristic_rule[i] = best_idx
    return heuristic_rule
