#!/bin/bash
#SBATCH -p qTRD
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -t 1:00:00
#SBATCH -e batch_errors/error%A.err 
#SBATCH -o batch_logs/out%A.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe
# a small delay at the start often helps
sleep 10s 

# print some message to the log
source /data/users4/rgirijala1/msproject/.venv/bin/activate

# in your sbatch script, before launching Python
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python consolidated_script.py

# it can be helpful for debugging to get the node name
echo $HOSTNAME
# a delay at the end is also good practice
sleep 10s
