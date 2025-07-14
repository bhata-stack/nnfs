#!/bin/bash
# To set up the conda environment, run `source env.sh`
# This will create the environment at ./_env and activate it

# Check that the script was sourced
(return 0 2> /dev/null) && sourced=1 || sourced=0

# Error if not sourced
if [[ $sourced == 0 ]]; then
    echo "Error: env.sh must be sourced (run 'source env.sh')"
    exit 1
fi

# Check if conda command is available
if ! $(conda --help &> /dev/null); then
    echo "Error: conda command not found"
    exit 1
fi

# Deactivate any active environments
echo "Deactivating conda environment (if active)"
conda deactivate || exit 1

# Create environment from file
echo "Creating conda environment at ./_env"
conda env create -p ./_env -f environment.yml || exit 1

# Activate the new environment
echo "Activating conda environment"
conda activate ./_env || exit 1

# Install the package
echo "Installing nnfs"
pip install -e . || exit 1
