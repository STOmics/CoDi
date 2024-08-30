# Use an official Ubuntu base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary dependencies including HDF5 and pkg-config
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    pkg-config \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the CoDi repository
RUN git clone https://github.com/stomics/codi.git /opt/codi

# Set working directory
WORKDIR /opt/codi

# Install Python dependencies
RUN pip3 install --no-cache-dir --force-reinstall h5py==3.8.0
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the Dockerfile to /opt
COPY Dockerfile /opt/Dockerfile

# Ensure that the entry point is the Python interpreter
ENTRYPOINT ["/bin/bash"]

# Default command will be to show the help message of CoDi
# CMD ["python /opt/CoDi/CoDi.py", "--help"]

