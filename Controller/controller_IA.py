from Controller.model import SimpleShipNN
import torch

class ControllerIA:
    def __init__(self, model=None):
        self.model = model or SimpleShipNN()
        self.model.eval()

    @property
    def name(self):
        return "IA_NN"

    def set_forced_action(self, action_index):
        self._forced_action = action_index

    def actions(self, ship_state, game_state):
        self._ship_state = ship_state
        self._game_state = game_state
        if hasattr(self, '_forced_action'):
            return self._decode_action_vector(self._forced_action)
        return self._decode_action_vector(0)

    def _decode_action_vector(self, idx):
        mapping = [
            (False,  0.0, False, False),
            (True,   0.0, False, False),
            (False, -1.0, False, False),
            (False,  1.0, False, False),
            (True,  -1.0, False, False),
            (True,   1.0, False, False),
        ]
        return mapping[idx]

    def get_input_vector(self, ship, game):
        try:
            ship_x = ship['position']['x']
            ship_y = ship['position']['y']
            ship_vx = ship['velocity']['x']
            ship_vy = ship['velocity']['y']
            ship_angle = ship['angle']
        except Exception as e:
            print(f"[WARN] Failed to read ship state: {e}")
            return torch.zeros(1, 17, dtype=torch.float32)

        input_vec = [ship_x, ship_y, ship_vx, ship_vy, ship_angle]

        try:
            asteroids = sorted(game['asteroids'], key=lambda a:
                (a['position']['x'] - ship_x) ** 2 + (a['position']['y'] - ship_y) ** 2)
        except Exception:
            asteroids = []

        for a in asteroids[:3]:
            try:
                rel_x = a['position']['x'] - ship_x
                rel_y = a['position']['y'] - ship_y
                input_vec += [rel_x, rel_y, a['velocity']['x'], a['velocity']['y']]
            except:
                input_vec += [0.0, 0.0, 0.0, 0.0]

        while len(input_vec) < 17:
            input_vec += [0.0] * 4

        return torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
