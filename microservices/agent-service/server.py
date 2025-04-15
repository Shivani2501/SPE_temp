from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import torch
import base64
import io
import requests
from agent import Agent, DQN_SERVICE_URL # Import DQN_SERVICE_URL

app = FastAPI()

# Initialize agent
agent = Agent(state_size=8, action_size=4, seed=0)

class ActRequest(BaseModel):
    state_b64: str
    eps: float = 0.0

class Experience(BaseModel):
    state_b64: str
    action: int
    reward: float
    next_state_b64: str
    done: bool

class BatchStepRequest(BaseModel):
    experiences: List[Experience]

@app.post("/act")
async def act(request: ActRequest):
    state_bytes = base64.b64decode(request.state_b64)
    # Load state tensor, mapping to CPU if it was saved on GPU
    state = torch.load(io.BytesIO(state_bytes), map_location=torch.device('cpu'))
    
    # Get action from agent
    action = agent.act(state.numpy(), request.eps)
    # Convert numpy int to standard Python int for JSON serialization
    return {"action": int(action)}

@app.post("/batch_step")
async def batch_step(request: BatchStepRequest):
    """Receives a batch of experiences and forwards them to the DQN service for learning."""
    try:
        # Forward the experiences to the DQN service's /learn endpoint
        learn_url = f"{DQN_SERVICE_URL}/learn"
        response = requests.post(learn_url, json={"experiences": [exp.dict() for exp in request.experiences]})
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle potential connection errors or bad responses from DQN service
        raise HTTPException(status_code=503, detail=f"Error communicating with DQN service: {e}")
    except Exception as e:
        # Handle other potential errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
