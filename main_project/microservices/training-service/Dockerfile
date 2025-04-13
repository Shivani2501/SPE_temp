# Training Service Dockerfile - Optimized with CPU-only PyTorch
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with CPU-only PyTorch
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY train.py .

# Install runtime dependencies (Adding swig for Box2D runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Run the training with optimizations
CMD ["python", "-OO", "train.py"]
