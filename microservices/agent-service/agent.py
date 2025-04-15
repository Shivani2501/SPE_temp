import numpy as np
import random
import torch
import requests
from requests.adapters import HTTPAdapter, Retry
import base64
import io
import time # Import time for potential backoff

DQN_SERVICE_URL = "http://dqn-service:8000"

# --- Configure Retries ---
retry_strategy = Retry(
    total=3,  # Total number of retries
    backoff_factor=0.5, # Wait 0.5s, 1s, 2s between retries
    status_forcelist=[500, 502, 503, 504], # Retry on server errors
    allowed_methods=["POST"] # Retry POST requests
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http_session = requests.Session()
http_session.mount("http://", adapter)
http_session.mount("https://", adapter)
# --- End Retry Configuration ---

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
        try:
            response = http_session.post( # Use the session with retries
                f"{DQN_SERVICE_URL}/predict",
                json={"state_b64": base64.b64encode(buffer.getvalue()).decode()},
                timeout=10 # Add a timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            resp_json = response.json()
            if "action_values_b64" not in resp_json:
                raise ValueError(f"Invalid response: {resp_json}")
            buffer = io.BytesIO(base64.b64decode(resp_json["action_values_b64"]))
            return torch.load(buffer)
        except requests.exceptions.RequestException as e:
            print(f"Error getting Q-values after retries: {e}")
            # Decide how to handle persistent failure: re-raise, return default, etc.
            # For now, re-raising to make the failure obvious
            raise e

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_values = self.get_q_values(state)
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
