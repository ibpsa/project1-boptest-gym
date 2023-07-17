# Start with a base Ubuntu image
FROM ubuntu:20.04
# This is the Focal Fossa ubuntu version
MAINTAINER Javier Arroyo <javier.arroyo@kuleuven.be>

# Avoid warnings while installing ubuntu
# debconf: unable to initialize frontend: Dialog
# debconf: (TERM is not set, so the dialog frontend is not usable.)
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install packages
RUN apt update && \
    apt install -y \
    git \
    python3.9 \
    python3-pip  && \
    rm -rf /var/lib/apt/lists/*

# Add developer user
RUN useradd -ms /bin/bash developer

# Activate developer
USER developer

# Set environment variables
ENV HOME /home/developer

# Install dependencies
RUN pip install gymnasium==0.28.1 stable-baselines3==2.0.0 numpy==1.24.4 pandas==2.0.3

# Expose port for interacting with BOPTEST
EXPOSE 5000


