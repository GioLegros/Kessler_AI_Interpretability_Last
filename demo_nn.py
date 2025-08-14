
# demo_nn.py
from src.scenario import Scenario
from src.kessler_game import KesslerGame
from src.graphics import GraphicsType
from nn_controller import NeuralFleeController

MODEL_PATH = "models/kessler_ppo/ppo_kessler_final.zip"

if __name__ == "__main__":
    scenario = Scenario(name="NN Demo", num_asteroids=6, map_size=(1000,800), time_limit=30.0)
    controllers = [NeuralFleeController(model_path=MODEL_PATH)]
    game = KesslerGame(settings={"graphics_type": GraphicsType.Tkinter, "frequency": 30.0})
    score, perf, state = game.run(scenario, controllers)
