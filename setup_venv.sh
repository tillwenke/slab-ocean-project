#!/usr/bin/env bash
# Create the .venv Python environment for WRF model evaluation.
# Creates the venv inside this repo folder.
set -euo pipefail

module purge
source /etc/profile
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 eth_proxy

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
rm -rf .venv

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m ensurepip --upgrade
python -m pip install --upgrade pip wheel
python -m pip install setuptools==58.4.0

# wrf-python requires numpy 1.x at build and run time.
python -m pip install "numpy==1.26.4"
python -m pip install --no-build-isolation --no-cache-dir wrf-python

cat > constraints.txt << 'EOF'
numpy==1.26.4
EOF

python -m pip install -c constraints.txt \
    xarray "netcdf4==1.6.5" scipy tqdm dask distributed \
    jupyter ipykernel markdown click cmasher cartopy

python -m pip uninstall -y basemap basemap-data-hires 2>/dev/null || true

echo y | jupyter kernelspec uninstall our-venv 2>/dev/null || true
python -m ipykernel install --user --name=our-venv --display-name "Our Venv"

deactivate
module purge
