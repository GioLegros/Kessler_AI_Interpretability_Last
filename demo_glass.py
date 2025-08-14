
# demo_glass.py
from src.scenario import Scenario
from src.kessler_game import KesslerGame
from src.graphics import GraphicsType
from glass_controller import GlassBoxFleeController

if __name__ == "__main__":
    scenario = Scenario(name="Glass Demo", num_asteroids=6, map_size=(1000,800), time_limit=30.0)
    controllers = [GlassBoxFleeController()]
    game = KesslerGame(settings={"graphics_type": GraphicsType.Tkinter, "frequency": 30.0})
    score, perf, state = game.run(scenario, controllers)
