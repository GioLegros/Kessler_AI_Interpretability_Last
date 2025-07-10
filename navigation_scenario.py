import numpy as np

from kesslergame import Scenario, TrainerEnvironment


###
# Scenario A: Large asteroids, starting far from the ship
###
def scenario_A(seed=None, n=24):
    asteroid_states = []
    for i in range(n):
        asteroid_states.append({
            'position': (0, 0),
            'size': 4,
        })

    return Scenario(
        map_size=(1000, 800),
        ship_states=[{
            'position': (500, 400),
            'lives': 1,
        }],
        asteroid_states=asteroid_states,
        time_limit=240,
        seed=seed,
    )


###
# Scenario D: Asteroids of mixed sizes, starting near the ship
###
def scenario_D(seed=None, n=32):
    asteroid_states = []
    for i in range(n):
        asteroid_states.append({
            'position': (0, 0),
            'size': (i % 4) + 1
        })

    return Scenario(
        map_size=(1000, 800),
        ship_states=[{
            'position': (200, 150),
            'lives': 1,
        }],
        asteroid_states=asteroid_states,
        time_limit=240,
        seed=seed,
    )

###
# Scenario E: Asteroids of mixed sizes arranged in a wall, all moving roughly towards the ship
###
def scenario_E(seed=None, n=32):
    rng = np.random.RandomState(seed=seed)
    asteroid_states = []
    yc = np.floor(np.linspace(0, 1000, n))
    speed = rng.randint(low=90, high=110, size=(n,))
    angle = rng.randint(low=-30, high=30, size=(n,))
    for i in range(n):
        asteroid_states.append({
            'position': (50, yc[i]),
            'angle': angle[i],
            'speed': speed[i],
            'size': (i % 4) + 1
        })

    return Scenario(
        map_size=(1000, 800),
        ship_states=[{
            'position': (300, 400),
            'lives': 1,
        }],
        asteroid_states=asteroid_states,
        time_limit=240,
        seed=seed,
    )

###
# Scenario F: Medium asteroids forming an almost-circle around the ship which slowly closes in.
###
def scenario_F(seed=None, n=32):
    rng = np.random.RandomState(seed=seed)
    asteroid_states = []
    r = 225
    theta = np.linspace(0.25 * np.pi, 1.75 * np.pi, n)
    x = r * np.cos(theta) + 500
    y = r * np.sin(theta) + 400
    speed = rng.randint(low=15, high=25, size=(n,))
    for i in range(n):
        asteroid_states.append({
            'position': (x[i], y[i]),
            'angle': np.degrees(theta[i] + np.pi),
            'speed': speed[i],
            'size': 3,
        })
    return Scenario(
        map_size=(1000, 800),
        ship_states=[{
            'position': (500, 400),
            'lives': 1,
        }],
        asteroid_states=asteroid_states,
        time_limit=240,
        seed=seed,
    )

def simple_scenario():
    asteroid_states = [{
        'position': (50, 400),
        'angle': 0,
        'speed': 0,
        'size': 4
    }]

    s = Scenario(
        map_size=(1000, 800),
        ship_states=[{
            'position': (200, 400),
            'angle': 0,
            'lives': 1,
        }],
        asteroid_states=asteroid_states,
        time_limit=60,
    )
    return s


def benchmark(controller, n_trials=20):
    game = TrainerEnvironment()

    benchmark_scenarios = [scenario_D, scenario_E, scenario_F]
    scores = np.zeros(shape=(len(benchmark_scenarios), n_trials), dtype=np.float64)

    for i, scenario in enumerate(benchmark_scenarios):
        for j in range(n_trials):
            score, _, __ = game.run(scenario=scenario(seed=j), controllers=[controller])
            scores[i, j] = score.sim_time

    return scores
