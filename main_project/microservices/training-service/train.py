import gymnasium as gym 
import random
import torch 
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import base64, io
import requests

AGENT_SERVICE_URL = "http://agent-service:8001" # Use service name for Docker Compose network

env = gym.make('LunarLander-v2') # Changed from v3 to v2

def encode_state(state):
    buffer = io.BytesIO()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(state).float().to(device)
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode()

def dqn(n_episodes=20, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    experiences = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        state_encoded = encode_state(state)
        score = 0
        
        for t in range(max_t):
            # Get action from Agent service
            action_resp = requests.post(
                f"{AGENT_SERVICE_URL}/act",
                json={"state_b64": state_encoded, "eps": eps}
            )
            action = action_resp.json()["action"]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_encoded = encode_state(next_state)
            
            # Collect experience for batch submission
            experiences.append({
                "state_b64": state_encoded,
                "action": action,
                "reward": reward,
                "next_state_b64": next_state_encoded,
                "done": terminated or truncated
            })
            
            # Submit batch every 10 steps
            if len(experiences) >= 10 or terminated or truncated:
                requests.post(
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
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
            
    return scores

if __name__ == "__main__":
    scores = dqn()
    
    # Save plot to file instead of showing interactively
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('training_scores.png')
    plt.close()

    # --- Get and Save Model Weights ---
    DQN_SERVICE_URL = "http://dqn-service:8002" # Use service name for Docker Compose network
    print("\nTraining finished. Retrieving model weights from DQN service...")
    try:
        response = requests.get(f"{DQN_SERVICE_URL}/get_model_weights")
        response.raise_for_status() # Raise an exception for bad status codes

        weights_b64 = response.json()["model_weights_b64"]
        weights_bytes = base64.b64decode(weights_b64)

        # Load the state dict from bytes
        buffer = io.BytesIO(weights_bytes)
        model_state_dict = torch.load(buffer, map_location=torch.device('cpu')) # Load to CPU

        # Save the state dict to a file
        torch.save(model_state_dict, 'lunar_lander_dqn.pth')
        print("Model weights saved successfully to lunar_lander_dqn.pth")

    except requests.exceptions.RequestException as e:
        print(f"Error retrieving model weights: {e}")
    except Exception as e:
        print(f"An error occurred while saving weights: {e}")
