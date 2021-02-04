#!/bin/sh
#SBATCH --job-name=ohsumed # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j%x.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=q1m_2h-1G

python3 main.py --configfile ohsumed
python3 main.py --configfile 20ng
