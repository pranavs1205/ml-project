#!/bin/bash

# create_environment.sh
# Script to set up the complete development environment for ml-project

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${CYAN}[SUCCESS]${NC} $1"
}

# Environment configuration
ENV_NAME="ml-project"
ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh"
ANACONDA_INSTALLER="Anaconda3-installer.sh"

print_status "Starting ml-project environment setup..."

# Step 1: Check if Anaconda is installed
print_step "1. Checking for Anaconda installation..."

check_anaconda() {
    # Check if conda command is available
    if command -v conda &> /dev/null; then
        print_success "Anaconda/Miniconda is already installed and available in PATH"
        return 0
    fi
    
    # Check common Anaconda installation directories
    if [ -d "$HOME/anaconda3" ] && [ -f "$HOME/anaconda3/bin/conda" ]; then
        print_success "Anaconda found in $HOME/anaconda3"
        export PATH="$HOME/anaconda3/bin:$PATH"
        return 0
    fi
    
    if [ -d "$HOME/miniconda3" ] && [ -f "$HOME/miniconda3/bin/conda" ]; then
        print_success "Miniconda found in $HOME/miniconda3"
        export PATH="$HOME/miniconda3/bin:$PATH"
        return 0
    fi
    
    if [ -d "/opt/anaconda3" ] && [ -f "/opt/anaconda3/bin/conda" ]; then
        print_success "Anaconda found in /opt/anaconda3"
        export PATH="/opt/anaconda3/bin:$PATH"
        return 0
    fi
    
    return 1
}

install_anaconda() {
    print_step "Installing Anaconda..."
    
    # Download Anaconda installer
    print_status "Downloading Anaconda installer..."
    if ! wget -O "$ANACONDA_INSTALLER" "$ANACONDA_URL"; then
        print_error "Failed to download Anaconda installer"
        print_status "Trying alternative download method with curl..."
        if ! curl -o "$ANACONDA_INSTALLER" "$ANACONDA_URL"; then
            print_error "Failed to download Anaconda installer with both wget and curl"
            exit 1
        fi
    fi
    
    # Make installer executable
    chmod +x "$ANACONDA_INSTALLER"
    
    # Install Anaconda silently
    print_status "Installing Anaconda (this may take a few minutes)..."
    if ! bash "$ANACONDA_INSTALLER" -b -p "$HOME/anaconda3"; then
        print_error "Anaconda installation failed"
        exit 1
    fi
    
    # Add Anaconda to PATH
    export PATH="$HOME/anaconda3/bin:$PATH"
    
    # Initialize conda for bash
    "$HOME/anaconda3/bin/conda" init bash
    
    # Source bashrc to update current session
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi
    
    # Clean up installer
    rm -f "$ANACONDA_INSTALLER"
    
    print_success "Anaconda installed successfully!"
}

# Check if Anaconda is installed, if not install it
if ! check_anaconda; then
    print_warning "Anaconda not found. Installing Anaconda..."
    install_anaconda
else
    print_success "Anaconda is available"
fi

# Verify conda is working
if ! command -v conda &> /dev/null; then
    print_error "Conda command not found even after installation attempt"
    print_status "Please restart your terminal and run this script again"
    exit 1
fi

# Step 2: Create Conda Environment
print_step "2. Creating Conda environment: $ENV_NAME"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME\s"; then
    print_warning "Environment '$ENV_NAME' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        print_status "Using existing environment..."
    fi
fi

# Create new environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME\s"; then
    print_status "Creating new conda environment with Python 3.9..."
    if ! conda create -n "$ENV_NAME" python=3.9 -y; then
        print_error "Failed to create conda environment"
        exit 1
    fi
    print_success "Conda environment '$ENV_NAME' created successfully!"
else
    print_success "Using existing conda environment '$ENV_NAME'"
fi

# Step 3: Activate Environment
print_step "3. Activating conda environment: $ENV_NAME"

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# Activate the environment
if ! conda activate "$ENV_NAME"; then
    print_error "Failed to activate conda environment '$ENV_NAME'"
    exit 1
fi

print_success "Conda environment '$ENV_NAME' activated!"

# Step 4: Install Dependencies
print_step "4. Installing dependencies from dev_requirements.txt"

# Check if dev_requirements.txt exists
if [ ! -f "dev_requirements.txt" ]; then
    print_error "dev_requirements.txt not found in current directory"
    exit 1
fi

print_status "Installing packages from dev_requirements.txt..."
if ! pip install -r dev_requirements.txt; then
    print_error "Failed to install some packages from dev_requirements.txt"
    print_warning "Continuing with kernel setup..."
else
    print_success "All packages installed successfully!"
fi

# Step 5: Jupyter Kernel Setup
print_step "5. Setting up Jupyter kernel for ml-project environment"

# Install ipykernel if not already installed
print_status "Installing ipykernel..."
pip install ipykernel

# Create Jupyter kernel for this environment
print_status "Creating Jupyter kernel for '$ENV_NAME' environment..."
if ! python -m ipykernel install --user --name="$ENV_NAME" --display-name="Python (ml-project)"; then
    print_error "Failed to create Jupyter kernel"
    exit 1
fi

print_success "Jupyter kernel 'Python (ml-project)' created successfully!"

# Step 6: Verification
print_step "6. Verifying installation..."

print_status "Checking installed packages..."
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

print_status "Checking key packages..."
python -c "
import sys
packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'black', 'pylint']
failed = []

for package in packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        print(f'✗ {package}')
        failed.append(package)

if failed:
    print(f'\nFailed to import: {failed}')
    sys.exit(1)
else:
    print('\nAll key packages imported successfully!')
"

if [ $? -eq 0 ]; then
    print_success "Package verification completed successfully!"
else
    print_warning "Some packages failed verification, but setup continues..."
fi

# Step 7: Final Instructions
echo
print_success "=== ML Project Environment Setup Complete! ==="
echo
print_status "Summary of what was completed:"
echo "✓ Anaconda installation (if needed)"
echo "✓ Conda environment '$ENV_NAME' created"
echo "✓ Environment activated"
echo "✓ Dependencies installed from dev_requirements.txt"
echo "✓ Jupyter kernel 'Python (ml-project)' created"
echo
print_status "Next steps:"
echo "1. Your environment is currently active: $CONDA_DEFAULT_ENV"
echo "2. To activate this environment in future sessions:"
echo "   conda activate $ENV_NAME"
echo "3. To start Jupyter Notebook/Lab:"
echo "   jupyter notebook  # or jupyter lab"
echo "4. In Jupyter, select kernel: 'Python (ml-project)'"
echo "5. To deactivate the environment:"
echo "   conda deactivate"
echo
print_status "Development environment is ready! 🚀"

# Optional: Display environment info
echo
print_status "Environment information:"
conda info --envs
echo
print_status "Available Jupyter kernels:"
jupyter kernelspec list
