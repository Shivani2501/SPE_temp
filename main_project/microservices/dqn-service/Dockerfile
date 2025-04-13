# DQN Service Dockerfile - Optimized
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install Python dependencies with CPU-only PyTorch
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY dqn.py .
COPY server.py .

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose API port
EXPOSE 8000

# Run the server with optimizations
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
