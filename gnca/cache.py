import torch
import numpy as np

class StateCache:
    """
    Stores states of GNCA model for replay memory
    """
    def __init__(self, size, initial_state):
        self.size = size
        # Store initial_state to be used for resets
        # Ensure we clone it so we don't modify the original reference
        self.init_state = initial_state
        self.cache = [initial_state.clone() for _ in range(size)] 
        
    def sample(self, count):
        """Returns batch of states and the indices used."""
        # Randomly select indices
        idxs = torch.randint(0, len(self.cache), (count,))
        batch = [self.cache[i] for i in idxs]
        
        # Stack them into one tensor for the model
        return idxs, torch.stack(batch)

    def update(self, idxs, new_states):
        """
        Update cache with new states.
        """
        # Detach & loop through indices to update the list one by one
        for i, idx in enumerate(idxs):
            self.cache[idx] = new_states[i].detach().clone()
            
        # replace one random sample with the initial seed
        replace_idx = torch.randint(0, len(self.cache), (1,)).item()
        self.cache[replace_idx] = self.init_state.detach().clone()