import numpy as np

class ReplayMemory:
    """
    A class for storing and sampling experiences for training.

    Attributes:
        buffer_size (int): Maximum size of the buffer.
        buffer (list): Internal list storing the experiences as tuples.
    """
    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.buffer = []
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the memory.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Episode done flag.
        """
        state = np.asarray(state)
        next_state = np.asarray(next_state)

        # Maintain memory size limit
        if len(self.buffer) >= self.buffer_size: 
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the memory.

        Args:
            batch_size (int): Size of the sample.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        # Randomly select batch_size indices
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Unpack and convert batch to numpy arrays
        states = np.stack(states)
        next_states = np.stack(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
