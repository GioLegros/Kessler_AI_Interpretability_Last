
# demo_glass.py
from kessler_game.scenario import Scenario
from kessler_game.kessler_game import KesslerGame
from kessler_game.graphics import GraphicsType
from kessler_game.glass_controller import GlassBoxFleeController

if __name__ == "__main__":
    scenario = Scenario(name="Glass Demo", num_asteroids=6, map_size=(1000,800), time_limit=30.0)
    controllers = [GlassBoxFleeController()]
    game = KesslerGame(settings={"graphics_type": GraphicsType.Tkinter, "frequency": 30.0})
    score, perf, state = game.run(scenario, controllers)
