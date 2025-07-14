from src.kessler_game import KesslerGame
from src.scenario import Scenario
from Controller.controller_gamepad import GamepadController
from src.graphics.graphics_handler import GraphicsType

from Controller.controller_keyboard import KeyboardController


ships = []
def main():
    for i in range(9):  # asteroid start pos
        ships.append({"position": (100 + i * 50, 150 + i * 50)})
    scenario = Scenario(
        name="ManetteTest",
        num_asteroids=8,
        map_size=(1000, 800),
        ammo_limit_multiplier=10, # mun limit
        ship_states=ships,  # ship start pos    
        stop_if_no_ammo=True     
    )

    #need controller to playz
    keyboardController = KeyboardController()
    #glassboxController = GlassBoxController()

    game = KesslerGame(settings={
        "graphics_type": GraphicsType.Tkinter,  # change with pyplot if doesnt work
        "prints_on": True,                      # console info
        "perf_tracker": True                   # perf track
    })

    #idée, faire un diagramme visiteur avec un head master, regarde les vesseaux et les asteroide et use une fonction update
    #dans le controleur qui aura le code des deux ia et qui switche entre les deux

    # Game lauch with 1 controller
    game.run(scenario, [keyboardController,keyboardController,keyboardController,keyboardController,keyboardController,
                        keyboardController,keyboardController,keyboardController,keyboardController,keyboardController])

if __name__ == "__main__":
    main()
