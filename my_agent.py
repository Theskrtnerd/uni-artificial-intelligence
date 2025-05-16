import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
import torch
import random

STUDENT_ID = 'a1901793'
DEGREE = 'UG'

class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'
        else:
            self.mode = mode

        self.storage = []
        self.max_storage_size = 10000

        self.input_dim = 7
        self.output_dim = 2
        self.learning_rate = 0.001

        self.network = MLPRegression(input_dim=self.input_dim, output_dim=self.output_dim, learning_rate=self.learning_rate)
        self.network2 = MLPRegression(input_dim=self.input_dim, output_dim=self.output_dim, learning_rate=self.learning_rate)
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)
   
        self.epsilon_decay = 0.995
        
        if self.mode == 'train':
            self.epsilon = 1.0
        else:
            self.epsilon = 0.01

        self.n = 64
        self.discount_factor = 0.99

        self.global_step = 0

        self.previous_state = None
        self.previous_action = None

        self.epsilon_min = 0.01

        if load_model_path:
            self.load_model(load_model_path)

    def build_state(self, state: dict) -> torch.Tensor:
        bird_center_y = state['bird_y'] + (state['bird_height'] / 2)
        bird_velocity = state['bird_velocity']
        
        next_pipe_x = float('inf')
        next_pipe_width = float('inf')
        pipe_top_y = float('inf')
        pipe_bottom_y = float('inf')
        
        for pipe in state['pipes']:
            if pipe['x'] > state['bird_x'] - state['bird_width']:
                if pipe['x'] < next_pipe_x:
                    next_pipe_x = pipe['x']
                    next_pipe_width = pipe['width']
                    if 'top' in pipe and 'bottom' in pipe:
                        pipe_top_y = pipe['top']
                        pipe_bottom_y = pipe['bottom']
        
        if pipe_top_y == float('inf'):
            pipe_top_y = 0.0
            pipe_bottom_y = state['screen_height']
            pipe_middle_y = state['screen_height'] / 2
        else:
            pipe_middle_y = (pipe_top_y + pipe_bottom_y) / 2

        if next_pipe_x == float('inf'):
            next_pipe_x = state['screen_width']
        
        if next_pipe_width == float('inf'):
            next_pipe_width = 0.0

        normalized_bird_center_y = bird_center_y / state['screen_height']
        normalized_bird_velocity_y = bird_velocity / 10.0
        normalized_dist_x = (next_pipe_x - (state['bird_x'] + state['bird_width'])) / state['screen_width']
        normalized_dist_y = (pipe_middle_y - bird_center_y) / state['screen_height']
        normalized_pipe_width = next_pipe_width / state['screen_width']
        normalized_pipe_top_y = pipe_top_y / state['screen_height']
        normalized_pipe_bottom_y = pipe_bottom_y / state['screen_height']

        normalized_dist_x = max(-1.0, min(1.0, normalized_dist_x))
        normalized_dist_y = max(-1.0, min(1.0, normalized_dist_y))
        
        state_tensor = torch.tensor([
            normalized_bird_center_y,
            normalized_bird_velocity_y,
            normalized_dist_x,
            normalized_dist_y,
            normalized_pipe_width,
            normalized_pipe_top_y,
            normalized_pipe_bottom_y
        ], dtype=torch.float32).unsqueeze(0)
        
        return state_tensor

    def reward(self, state: dict) -> float:
        if state['done']:
            if state['done_type'] == 'well_done':
                return 10.0
            elif state['done_type'] == 'off_screen':
                return -1.0
            elif state['done_type'] == 'hit_pipe':
                return -1.0  # Simplified penalty for hitting pipe
            else:
                return 0.0
        else:
            survival_reward = 0.1
            
            score_bonus = 0.0
            if self.previous_state and state['score'] > self.previous_state['score']:
                score_bonus = 1.0  # Reward for passing a pipe
            
            return survival_reward + score_bonus

    def choose_action(self, state: dict, action_table: dict) -> int:
        state_tensor = self.build_state(state)
        
        explore_decision = False
        
        if self.mode == 'train' and random.random() < self.epsilon:
            explore_decision = True
        
        if explore_decision:
            action_idx = random.randint(0, 1)
            a_t = action_table['jump'] if action_idx == 1 else action_table['do_nothing']
            
        else:
            with torch.no_grad():
                q_values = self.network(state_tensor)
                a_t = torch.argmax(q_values[0][:2]).item()

        self.previous_action = a_t
        
        return a_t

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        if self.mode != 'train' or self.previous_state is None:
            self.previous_state = state
            return

        reward = self.reward(state)
        state_tensor = self.build_state(self.previous_state)
        next_state_tensor = self.build_state(state)
        done = state.get('done', False)

        self.storage.append((
            state_tensor.squeeze(0).numpy(),
            self.previous_action,
            reward,
            next_state_tensor.squeeze(0).numpy(),
            done
        ))
    
        # Limit storage size
        if len(self.storage) > self.max_storage_size:
            self.storage.pop(0)

        if len(self.storage) >= self.n:
            batch = random.sample(list(self.storage), self.n)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.vstack(states), dtype=torch.float32)
            next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            current_q_values = self.network(states)
            
            with torch.no_grad():
                next_q_values = self.network2(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

            q_targets = current_q_values.clone().detach()
            weights = torch.zeros_like(q_targets)
            
            for i in range(self.n):
                q_targets[i, actions[i].item()] = target_q_values[i].item()
                weights[i, actions[i].item()] = 1.0
            
            self.network.fit_step(states.numpy(), q_targets.numpy(), weights.numpy())
            self.global_step += 1

            if self.global_step > 0 and self.global_step % 100 == 0:
                MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)

        self.previous_state = state

    def save_model(self, path: str = 'my_model.ckpt'):
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)

    args = parser.parse_args()

    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=False)
    episodes = 10000

    scores_history = []
    mileage_history = []
    best_avg_score = -float('inf')

    for episode in range(episodes):
        env.play(player=agent)
        scores_history.append(env.score)
        mileage_history.append(env.mileage)
        score_threshold = 5

        if env.score >= score_threshold:
            print(agent.global_step)
            print(env.score)
            print(env.mileage)

        # current_avg_score = np.mean(scores_history[-100:]) if len(scores_history) >= 100 else np.mean(scores_history)
        # if current_avg_score > best_avg_score and len(scores_history) >= 50:
        #     best_avg_score = current_avg_score
        #     best_model_path = f'my_model.ckpt'
        #     agent.save_model(path=best_model_path)
        #     print(f"New best average score model saved: {best_model_path} (Avg Score: {best_avg_score:.2f})")

        recent_scores = scores_history[-100:] if len(scores_history) >= 100 else scores_history
        high_score_count = sum(score >= score_threshold for score in recent_scores)

        print(np.max(recent_scores))
        
        if high_score_count > best_avg_score and len(scores_history) >= 50:
            best_avg_score = high_score_count
            best_model_path = f'my_model.ckpt'
            agent.save_model(path=best_model_path)
            print(f"New best model saved: {best_model_path} (High scores: {high_score_count}/{len(recent_scores)})")

        agent.previous_state = None
        agent.previous_action = None

        if (episode + 1) % 10 == 0:
            MyAgent.update_network_model(net_to_update=agent.network2, net_as_source=agent.network)
        
        if (episode + 1) % 500 == 0:
            agent.storage.clear()

    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
