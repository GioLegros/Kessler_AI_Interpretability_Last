import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from src.kessler_game import KesslerGame
from src.scenario import Scenario
from Controller.controller_IA import ControllerIA
from Controller.model import SimpleShipNN 


class DQNAgent:
    def __init__(self):
        self.model = SimpleShipNN()
        self.target_model = SimpleShipNN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=100_000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_target_freq = 10

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        with torch.no_grad():
            q_vals = self.model(state)
        return torch.argmax(q_vals).item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for s, a, r, s2, done in batch:
            s = s.unsqueeze(0)
            s2 = s2.unsqueeze(0)
            target = self.model(s).detach().clone()
            if done:
                target[0][a] = r
            else:
                future_q = self.target_model(s2).max().item()
                target[0][a] = r + self.gamma * future_q
            states.append(s.squeeze(0))
            targets.append(target.squeeze(0))

        self.optimizer.zero_grad()
        loss = self.loss_fn(torch.stack([self.model(s.unsqueeze(0))[0] for s in states]), torch.stack(targets))
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self):
        torch.save(self.model.state_dict(), "model.pth")

def is_alive(ship_state):
    print(f"[DEBUG] Checking if ship is alive: {ship_state}")
    return isinstance(ship_state, dict) and ship_state.get("status") == "alive"

def run_training():
    agent = DQNAgent()

    for episode in range(500):
        controller = ControllerIA(agent.model)

        scenario = Scenario(
            name=f"TrainEp_{episode}",
            num_asteroids=8,
            map_size=(1000, 800),
            ammo_limit_multiplier=10,
            ship_states=[{"position": (100, 150)}],
            stop_if_no_ammo=True
        )

        game = KesslerGame(settings={
            "headless": True,
            "frequency": 15,
            "prints_on": False
        })

        game.run(scenario, [controller])

        state = controller.get_input_vector(controller._ship_state, controller._game_state)
        total_reward = 0
        action = 0 

        while is_alive(controller._ship_state):
            print(f"[EP {episode}] Frames survécues : {total_reward + 100}") 
            action = agent.act(state)
            controller.set_forced_action(action)
            reward = 1
            game.update()

            next_state = controller.get_input_vector(controller._ship_state, controller._game_state)
            agent.remember(state, action, reward, next_state, False)
            total_reward += reward
            state = next_state
            
        reward = -100
        agent.remember(state, action, reward, torch.zeros_like(state), True)
        total_reward += reward

        print(f"[EP {episode}] Total reward: {total_reward}")



        agent.replay()

        if episode % agent.update_target_freq == 0:
            agent.update_target_network()

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        if episode % 50 == 0:
            agent.save()

    agent.save()


if __name__ == "__main__":
    run_training()
