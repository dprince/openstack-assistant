# Copy openstack-k8s-mcp binary from source image
FROM quay.io/dprince/openstack-k8s-mcp:main AS mcp-binary

# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the openstack-k8s-mcp binary from the mcp-binary stage
COPY --from=mcp-binary /openstack-k8s-mcp /usr/local/bin/openstack-k8s-mcp

# Install system dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the application in editable mode
RUN pip install --no-cache-dir -e .

# Copy and set permissions for the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD []
