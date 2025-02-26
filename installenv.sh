#!/bin/bash

# Create a directory named .env and change into it
mkdir -p .env
cd .env

# Create a Python virtual environment named pytorch
python3 -m venv pytorch

# Activate the virtual environment
source pytorch/bin/activate

# Install the requirements from requirements.txt
pip install -r ../requirements.txt