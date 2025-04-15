from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.optim as optim
import torch.nn.functional as F
import base64
import numpy as np
from collections import deque, namedtuple
import random
import io

from dqn import QNetwork

# Constants from RL_Ops-main/agent.py
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network (though learning is triggered by API call here)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = FastAPI()

# --- Replay Buffer (from RL_Ops-main/agent.py) ---
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# --- DQN Service State ---
qnetwork_local = QNetwork(state_size=8, action_size=4, seed=0).to(device)
qnetwork_target = QNetwork(state_size=8, action_size=4, seed=0).to(device)
optimizer = optim.Adam(qnetwork_local.parameters(), lr=LR)
memory = ReplayBuffer(action_size=4, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0)
t_step = 0 # Counter for learning steps

# --- Helper Functions (from RL_Ops-main/agent.py) ---
def soft_update(local_model, target_model, tau):
    """Soft update model parameters."""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def learn(experiences, gamma):
    """Update value parameters using given batch of experience tuples."""
    states, actions, rewards, next_states, dones = experiences

    # Get max predicted Q values (for next states) from target model
    q_targets_next = qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    # Compute Q targets for current states
    q_targets = rewards + (gamma * q_targets_next * (1 - dones))

    # Get expected Q values from local model
    q_expected = qnetwork_local(states).gather(1, actions)

    # Compute loss
    loss = F.mse_loss(q_expected, q_targets)
    # Minimize the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    soft_update(qnetwork_local, qnetwork_target, TAU)
    return loss.item() # Return loss value

# --- API Models ---
class PredictionRequest(BaseModel):
    state_b64: str  # Base64 encoded state tensor

class ExperienceModel(BaseModel):
    state_b64: str
    action: int
    reward: float
    next_state_b64: str
    done: bool

class LearnRequest(BaseModel):
    experiences: List[ExperienceModel]

# --- Utility Function ---
def decode_state(state_b64: str) -> np.ndarray:
    """Decodes base64 encoded torch tensor back to numpy array."""
    state_bytes = base64.b64decode(state_b64)
    # Load tensor, mapping to CPU if necessary
    tensor = torch.load(io.BytesIO(state_bytes), map_location=torch.device('cpu'))
    return tensor.numpy()

# --- API Endpoints ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predicts action values for a given state."""
    try:
        state_np = decode_state(request.state_b64)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(device) # Add batch dim

        qnetwork_local.eval() # Set model to evaluation mode
        with torch.no_grad():
            action_values = qnetwork_local(state)
        qnetwork_local.train() # Set model back to train mode

        # Encode response
        buffer = io.BytesIO()
        # Detach from graph and move to CPU before saving
        torch.save(action_values.squeeze(0).cpu(), buffer)
        return {"action_values_b64": base64.b64encode(buffer.getvalue()).decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/learn")
async def handle_learn(request: LearnRequest):
    """Receives experiences, adds them to the buffer, and performs learning step."""
    global t_step
    last_loss = None
    try:
        for exp in request.experiences:
            # Decode states before adding to memory
            state = decode_state(exp.state_b64)
            next_state = decode_state(exp.next_state_b64)
            memory.add(state, exp.action, exp.reward, next_state, exp.done)

            # Learn every UPDATE_EVERY time steps.
            t_step = (t_step + 1) % UPDATE_EVERY
            if t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(memory) > BATCH_SIZE:
                    experiences = memory.sample()
                    last_loss = learn(experiences, GAMMA)

        return {"status": "experiences received", "buffer_size": len(memory), "last_loss": last_loss}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning error: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "buffer_size": len(memory)}

@app.get("/get_model_weights")
async def get_model_weights():
    """Returns the state dictionary of the local Q-network."""
    try:
        # Ensure the model is on CPU before saving for broader compatibility
        qnetwork_local.cpu()
        buffer = io.BytesIO()
        torch.save(qnetwork_local.state_dict(), buffer)
        buffer.seek(0)
        weights_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # Move model back to the original device if needed
        qnetwork_local.to(device)
        return {"model_weights_b64": weights_b64}
    except Exception as e:
        # Move model back to the original device in case of error
        qnetwork_local.to(device)
        raise HTTPException(status_code=500, detail=f"Error getting model weights: {e}")
