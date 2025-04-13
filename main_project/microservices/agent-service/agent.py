import numpy as np
import random
import torch
import requests
import base64
import io

DQN_SERVICE_URL = "http://dqn-service:8000"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

    def get_q_values(self, state):
        """Get Q-values from DQN service. Assumes state is already a Tensor."""
        buffer = io.BytesIO()
        # state is already a tensor, ensure it's float and save directly
        torch.save(state.float(), buffer)
        buffer.seek(0) # Reset buffer position after writing
        response = requests.post(
            f"{DQN_SERVICE_URL}/predict",
            json={"state_b64": base64.b64encode(buffer.getvalue()).decode()}
        )
        buffer = io.BytesIO(base64.b64decode(response.json()["action_values_b64"]))
        return torch.load(buffer)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_values = self.get_q_values(state)
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
