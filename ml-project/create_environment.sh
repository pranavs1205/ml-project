#!/bin/bash

set -e

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Anaconda not found. Installing..."
    curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-MacOSX-x86_64.sh
    bash Anaconda3-2024.02-1-MacOSX-x86_64.sh -b
    export PATH="$HOME/anaconda3/bin:$PATH"
fi

# Create conda environment
conda create -y -n ml-project python=3.10

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml-project

# Install dependencies
pip install -r dev_requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=ml-project

echo "Environment setup complete."