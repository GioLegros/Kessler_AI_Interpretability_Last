from src.controller import KesslerController
from typing import Dict, Tuple, Any
from immutabledict import immutabledict

from pynput import keyboard, mouse # type: ignore
import threading

class KeyboardController(KesslerController):
    #initialization of the controller
    def __init__(self):
        self._pressed_keys = set()
        self._pressed_buttons = set()

        self._thrust = 0.0
        self._turn_rate = 0.0
        self._fire = False
        self._drop_mine = False

        threading.Thread(target=self._start_keyboard_listener, daemon=True).start()
        threading.Thread(target=self._start_mouse_listener, daemon=True).start()

    #start the keyboard listeners
    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                self._pressed_keys.add(key.char.lower())
            except AttributeError:
                pass

        def on_release(key):
            try:
                self._pressed_keys.discard(key.char.lower())
            except AttributeError:
                pass

        keyboard.Listener(on_press=on_press, on_release=on_release).run()

    #start the mouse listener
    def _start_mouse_listener(self):
        def on_click(x, y, button, pressed):
            if button.name == "left":
                if pressed:
                    self._pressed_buttons.add("left")
                else:
                    self._pressed_buttons.discard("left")
            elif button.name == "right":
                if pressed:
                    self._pressed_buttons.add("right")
                else:
                    self._pressed_buttons.discard("right")

        mouse.Listener(on_click=on_click).run()

    @property
    def name(self) -> str:
        return "Keyboard Controller"

    #actions to be performed by the controller
    def actions(self, ship_state: Dict[str, Any], game_state: immutabledict[Any, Any]) -> Tuple[float, float, bool, bool]:
        self._thrust = 480.0 if 'z' in self._pressed_keys else -480.0 if 's' in self._pressed_keys else 0.0
        self._turn_rate = 180.0 if 'q' in self._pressed_keys else -180.0 if 'd' in self._pressed_keys else 0.0
        self._fire = "left" in self._pressed_buttons
        self._drop_mine = "right" in self._pressed_buttons
        return self._thrust, self._turn_rate, self._fire, self._drop_mine
