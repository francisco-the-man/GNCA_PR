import torch
import numpy as np

class StateCache:
    """.
    Stores states of GNCA and handles the Section 4.3 logic internally.
    """
    def __init__(self, initial_state, size=1024):
        # Store initial_state to be used for resets
        # Ensure we clone it so we don't modify the original reference
        self.init_state = initial_state
        self.cache = [initial_state.clone() for _ in range(size)] 
        
    def sample(self, count):
        """Returns batch of states and the indices used."""
        # Randomly select indices
        idxs = torch.randint(0, len(self.cache), (count,))
        
        # List comprehension to retrieve items (Standard list indexing)
        batch = [self.cache[i] for i in idxs]
        
        return batch, idxs

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