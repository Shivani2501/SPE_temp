# import gymnasium as gym 
# import random
# import torch 
# import numpy as np
# from collections import deque, namedtuple
# import matplotlib.pyplot as plt
# import base64, io
# import requests

# AGENT_SERVICE_URL = "http://agent-service:8001" # Use service name for Docker Compose network

# env = gym.make('LunarLander-v2') # Changed from v3 to v2

# def encode_state(state):
#     buffer = io.BytesIO()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tensor = torch.from_numpy(state).float().to(device)
#     torch.save(tensor, buffer)
#     return base64.b64encode(buffer.getvalue()).decode()

# def dqn(n_episodes=20, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
#     scores = []
#     experiences = []
#     scores_window = deque(maxlen=100)
#     eps = eps_start
    
#     for i_episode in range(1, n_episodes+1):
#         state, _ = env.reset()
#         state_encoded = encode_state(state)
#         score = 0
        
#         for t in range(max_t):
#             # Get action from Agent service
#             action_resp = requests.post(
#                 f"{AGENT_SERVICE_URL}/act",
#                 json={"state_b64": state_encoded, "eps": eps}
#             )
#             action = action_resp.json()["action"]
            
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             next_state_encoded = encode_state(next_state)
            
#             # Collect experience for batch submission
#             experiences.append({
#                 "state_b64": state_encoded,
#                 "action": action,
#                 "reward": reward,
#                 "next_state_b64": next_state_encoded,
#                 "done": terminated or truncated
#             })
            
#             # Submit batch every 10 steps
#             if len(experiences) >= 10 or terminated or truncated:
#                 requests.post(
#                     f"{AGENT_SERVICE_URL}/batch_step",
#                     json={"experiences": experiences}
#                 )
#                 experiences = []
            
#             state = next_state
#             state_encoded = next_state_encoded
#             score += reward
#             if terminated or truncated:
#                 break 
                
#         scores_window.append(score)
#         scores.append(score)
#         eps = max(eps_end, eps_decay*eps)
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
#         if i_episode % 100 == 0:
#             print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#         if np.mean(scores_window)>=200.0:
#             print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
#             break
            
#     return scores

# if __name__ == "__main__":
#     scores = dqn()
    
#     # Save plot to file instead of showing interactively
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(np.arange(len(scores)), scores)
#     plt.ylabel('Score')
#     plt.xlabel('Episode #')
#     plt.savefig('training_scores.png')
#     plt.close()

#     # --- Get and Save Model Weights ---
#     DQN_SERVICE_URL = "http://dqn-service:8000" # Use service name for Docker Compose network
#     print("\nTraining finished. Retrieving model weights from DQN service...")
#     try:
#         response = requests.get(f"{DQN_SERVICE_URL}/get_model_weights")
#         response.raise_for_status() # Raise an exception for bad status codes

#         weights_b64 = response.json()["model_weights_b64"]
#         weights_bytes = base64.b64decode(weights_b64)

#         # Load the state dict from bytes
#         buffer = io.BytesIO(weights_bytes)
#         model_state_dict = torch.load(buffer, map_location=torch.device('cpu')) # Load to CPU

#         # Save the state dict to a file
#         torch.save(model_state_dict, 'lunar_lander_dqn.pth')
#         print("Model weights saved successfully to lunar_lander_dqn.pth")

#     except requests.exceptions.RequestException as e:
#         print(f"Error retrieving model weights: {e}")
#     except Exception as e:
#         print(f"An error occurred while saving weights: {e}")

import gymnasium as gym
import requests
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configurable constants
AGENT_SERVICE_URL = "http://agent-service:8001"
DQN_SERVICE_URL = "http://dqn-service:8000" # Corrected port
ENV_NAME = 'LunarLander-v2'

BATCH_SIZE = 10
N_EPISODES = 1000
MAX_T = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_SCORE = 200.0
WINDOW_SIZE = 100

# Create environment
env = gym.make(ENV_NAME)

# Encode state as JSON-friendly list
def encode_state(state):
    return state.tolist()

# Retry logic wrapper for network requests
def safe_post(url, json_payload, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            response = requests.post(url, json=json_payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"POST request failed ({attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to POST to {url} after {retries} attempts.")

# DQN training loop
def dqn():
    scores = []
    experiences = []
    scores_window = deque(maxlen=WINDOW_SIZE)
    eps = EPS_START

    for i_episode in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        state_encoded = encode_state(state)
        score = 0

        for t in range(MAX_T):
            # Get action from agent-service
            action_response = safe_post(
                f"{AGENT_SERVICE_URL}/act",
                json={"state": state_encoded, "eps": eps}
            )

            if "action" not in action_response:
                raise ValueError("Missing 'action' in response")

            action = action_response["action"]

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_encoded = encode_state(next_state)

            # Collect experience
            experiences.append({
                "state": state_encoded,
                "action": action,
                "reward": reward,
                "next_state": next_state_encoded,
                "done": terminated or truncated
            })

            # Submit batch
            if len(experiences) >= BATCH_SIZE or terminated or truncated:
                safe_post(
                    f"{AGENT_SERVICE_URL}/batch_step",
                    json={"experiences": experiences}
                )
                experiences = []

            state = next_state
            state_encoded = next_state_encoded
            score += reward

            if terminated or truncated:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(EPS_END, EPS_DECAY * eps)

        avg_score = np.mean(scores_window)
        logging.info(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}")

        if avg_score >= TARGET_SCORE:
            logging.info(f"\nEnvironment solved in {i_episode - WINDOW_SIZE} episodes!\tAverage Score: {avg_score:.2f}")
            break

    return scores

# Save plot of scores
def save_score_plot(scores, filename="training_scores.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Training Progress')
    plt.savefig(filename)
    plt.close()
    logging.info(f"Score plot saved as {filename}")

# Download and save trained model weights
def retrieve_and_save_model(filename="lunar_lander_dqn.pth"):
    logging.info("Retrieving model weights from DQN service...")
    try:
        response = requests.get(f"{DQN_SERVICE_URL}/get_model_weights")
        response.raise_for_status()

        weights_json = response.json()
        if "model_weights_b64" not in weights_json:
            raise ValueError("Missing 'model_weights_b64' key in response")

        weights_b64 = weights_json["model_weights_b64"]
        weights_bytes = base64.b64decode(weights_b64)

        buffer = io.BytesIO(weights_bytes)
        model_state_dict = torch.load(buffer, map_location=torch.device('cpu'))

        torch.save(model_state_dict, filename)
        logging.info(f"Model weights saved successfully to {filename}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error retrieving model weights: {e}")
    except Exception as e:
        logging.error(f"An error occurred while saving weights: {e}")

# Main execution
if __name__ == "__main__":
    import base64, io
    scores = dqn()
    save_score_plot(scores)
    retrieve_and_save_model()
