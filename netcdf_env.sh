module purge
source /etc/profile
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 eth_proxy

source $HOME/wcm/venv-wcm/bin/activate

python -m pip uninstall -y netCDF4 cftime
python -m pip install --no-cache-dir "netCDF4==1.7.2"

python - << 'EOF'
import sys, wrf, netCDF4
print("python:", sys.version)
print("wrf-python:", wrf.__version__)
print("netCDF4:", netCDF4.__version__)
EOF
