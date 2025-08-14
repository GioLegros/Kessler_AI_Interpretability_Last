from src.scenario import Scenario
from src.kessler_game import KesslerGame
from src.graphics import GraphicsType
from headmaster import HeadMaster
from hybrid_controller import HybridController
import math

#model path for the trained neural network
MODEL_PATH = "models/kessler_ppo/ppo_kessler_final.zip"

# create a scenario with ships in a circle, avoid collision with other ships
W, H = 1000, 800
cx, cy = W/2, H/2
R = 200
N = 10
ship_states = []
for i in range(N):
    ang = 2*math.pi * i / N
    x = cx + R * math.cos(ang)
    y = cy + R * math.sin(ang)
    heading = (ang * 180.0 / math.pi + 180.0) % 360.0
    ship_states.append({"position": (x, y), "angle": heading})

scenario = Scenario(name="HeadMaster Demo", num_asteroids=3, ship_states=ship_states, map_size=(W, H), time_limit=60.0)


# HeadMaster : 2 en black box toutes les 0.5 sec
hm = HeadMaster(top_k=2, interval_s=0.5, w_dist=1.0, w_approach=1.0)

# hybrid controller per ship
controllers = [HybridController(hm, nn_model_path=MODEL_PATH, device="cpu") for _ in range(10)]

game = KesslerGame(settings={"graphics_type": GraphicsType.Tkinter, "frequency": 30.0})
score, perf, state = game.run(scenario, controllers)
