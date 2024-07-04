import torch
import torch.optim as optim
import random
from environment import PacmanEnv
from model import DQN
from replay import ReplayMemory
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter

class DQNAgent:

    """
    DQN Agent that interacts with the Pacman environment and learns using a Deep Q-Network.
    
    Attributes:
        env: Pacman environment instance.
        device: Device to run the model on (CPU or GPU).
        dqn:Primary DQN model.
        target_dqn:Target DQN model.
        optimizer: Optimizer for the DQN model.
        replay_memory : Replay memory buffer.
        config: Dictionary containing hyperparameters and configuration settings.
        writer : TensorBoard writer for logging.
        model_name : Name for saving the model.
    """

    def __init__(self, altirra_path, rom_path, input_shape, num_actions, config):
        #Class initializations
        self.env = PacmanEnv(altirra_path, rom_path) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(input_shape, num_actions).to(self.device) 
        self.target_dqn = DQN(input_shape, num_actions).to(self.device) 
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=config['learning_rate'])
        self.replay_memory = ReplayMemory(config['buffer_size'])
        self.altirra_path = config['altirra_path']
        self.rom_path = config['rom_path']
        
        #Hyperparameters for training
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']  # Initial epsilon value
        self.epsilon_end = config['epsilon_end']  # Minimum epsilon value
        self.epsilon_decay = config['epsilon_decay']  # Decay rate for epsilon
        self.run_id = config['run_id']
        #Model Saving

        self.models_dir = f'models/run_{self.run_id}'
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)  # Create models directory if it doesn't exist
        
        #Logging
        log_dir = f'runs/run_{self.run_id}'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_rewards = []
        self.epsilons = []
        self.model_name = "dqn_model.pt"


    def train(self, num_episodes):
        """
        Train the DQN agent.
        
        Args:
            num_episodes (int): Number of episodes to train the agent for.
        """
        start_time = time.time()
        for episode in range(num_episodes):
            state, _, original = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_loss = 0.0

            while not done:
                # Select an action, take a step and store transition
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.replay_memory.add(state, action, reward, next_state, done)
                # Move to next state
                state = next_state
                total_reward += reward
                steps += 1

                # If enough samples are available, sample a batch and train the DQN
                if len(self.replay_memory) > self.batch_size:
                    states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)
                    
                    # Convert to pytorch tensors
                    states = torch.tensor(states, dtype=torch.float32).to(self.device)
                    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                    next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

                    # Compute current Q-values using the primary DQN
                    current_q_values = self.dqn(states)
                    # Compute the next Q-values using the target DQN
                    next_q_values = self.target_dqn(next_states).max(dim=1)[0]
                    # Compute the target Q-values
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

                    # Compute the loss between the current Q-values and target Q-values
                    loss = torch.nn.functional.mse_loss(current_q_values.gather(1, actions), target_q_values.unsqueeze(1))
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    episode_loss += loss.item()

                    # Update the target network at specified frequencies
                    if episode % self.target_update_freq == 0:
                        self.update_target()
                        self.save_model(episode)
               
                if done:
                    self.log_episode_results(episode, total_reward, steps, episode_loss, start_time)
                    next_state, score, original = self.env.reset_game()
                    
            # Decay epsilon for exploration-exploitation trade-off
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def select_action(self, state):
        """
        Select an action based on the current state using an epsilon-greedy policy.
        
        Args:
            state (numpy.ndarray): Current state.
        
        Returns:
            int: Selected action.
        """
        if random.random() < self.epsilon:
            # Exploration: Randomly select an action
            return random.randrange(self.env.num_actions)
        else:
            # Exploitation: Choose the action with the highest Q-value
            with torch.no_grad():
                q_values = self.dqn(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
                return q_values.argmax().item()
    
    def log_episode_results(self, episode, total_reward, steps, episode_loss, start_time):
        average_loss = episode_loss / steps if steps > 0 else 0.0
        training_duration = time.time() - start_time

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {self.epsilon}, Average Loss: {average_loss}, Training Time: {training_duration}")
        self.writer.add_scalar('Total Reward', total_reward, episode)
        self.writer.add_scalar('Epsilon', self.epsilon, episode)
        self.writer.add_scalar('Episode Steps', steps, episode)
        self.writer.add_scalar('Average Loss', average_loss, episode)

    def save_model(self, episode):
        torch.save(self.dqn.state_dict(), f'{self.models_dir}/dqn_model_ep{episode}.pt')

    def update_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict()) 

    def close(self):
        self.env.close()
