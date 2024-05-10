# Start with a base Ubuntu image
FROM --platform=linux/x86_64 ubuntu:20.04
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

# Install dependencies
# Gym dependency and shimmy can be removed after retraining missing agents:
# A2C_simple, A2C_A, A2C_B
RUN pip install \
    gymnasium==0.28.1 \
    stable-baselines3==2.0.0 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    requests \
    scipy \
    nbconvert==7.7.3 \
    gym \
    shimmy \
    ipykernel==6.25.1 \
    PyYAML \
    pytest

# Add developer user with permissions in the working directory
RUN useradd -ms /bin/bash developer
ENV HOME /home/developer
RUN mkdir -m 1777 ${HOME}/boptestgym
ENV PYTHONPATH ${HOME}/boptestgym

# Activate developer
USER developer

