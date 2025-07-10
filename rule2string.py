import numpy as np

def rule2string(fuzzy_rule):
    attr_labels = [
        'FAR-FRONT',
        'FAR-LEFT',
        'FAR-BEHIND',
        'FAR-RIGHT',
        'MIDDLE-FRONT',
        'MIDDLE-LEFT',
        'MIDDLE-BEHIND',
        'MIDDLE-RIGHT',
        'NEAR-FRONT',
        'NEAR-LEFT',
        'NEAR-BEHIND',
        'NEAR-RIGHT',
    ]
    fuzzy_labels = [
        "DON'T CARE",
        'LOW',
        'MEDIUM-LOW',
        'MEDIUM',
        'MEDIUM-HIGH',
        'HIGH',
        'CLEAR',
        'DANGEROUS'
    ]
    thrust_labels = [
        'FULL-REVERSE',
        'HALF-REVERSE',
        'NO-THRUST',
        'HALF-FORWARD',
        'FULL-FORWARD',
    ]
    turn_labels = [
        'FULL-RIGHT',
        'HALF-RIGHT',
        'NO-TURN',
        'HALF-LEFT',
        'FULL-LEFT',
    ]

    output = 'IF '
    for i in range(12):
        fuzzy_set = fuzzy_rule[i]
        if fuzzy_set == 0:
            continue
        output += f'{attr_labels[i]} IS {fuzzy_labels[fuzzy_set]} '
        if i < 11:
            output += 'AND '
    output += f'THEN THRUST IS {thrust_labels[fuzzy_rule[12]]} '
    output += f'AND TURN IS {turn_labels[fuzzy_rule[13]]}'
    return output

def main():
    antecedents = np.random.randint(low=0, high=8, size=(5,12))
    outputs = np.random.randint(low=0, high=5, size=(5, 2))
    rules = np.concatenate((antecedents, outputs), axis=1)
    print(rules)
    for rule in rules:
        print(rule2string(rule))


if __name__ == '__main__':
    main()