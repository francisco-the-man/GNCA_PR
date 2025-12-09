import torch
import random

class SamplePool:
    """
    Implements the 'cache' (replay memory) from Section 4.3.
    Stores a history of states to prevent catastrophic forgetting.
    """
    def __init__(self, pool_size, x_0):
        """
        pool_size: Number of states to keep in memory (e.g., 1024)
        x_0: The initial seed state (batch of seeds)
        """
        self.pool = x_0.clone().repeat(pool_size, 1, 1) if x_0.ndim == 2 else x_0.repeat(pool_size, 1)
        self.size = pool_size

    def sample(self, batch_size):
        """
        Returns a batch of indices and states from the pool.
        """
        indices = torch.randint(0, self.size, (batch_size,))
        return indices, self.pool[indices]

    def commit(self, indices, new_states):
        """
        Updates the pool with the new evolved states.
        """
        # Detach is critical here to stop gradient history from exploding memory
        self.pool[indices] = new_states.detach()