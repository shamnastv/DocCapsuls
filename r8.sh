#!/bin/sh
#SBATCH --job-name=R8 # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j%x.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl2_48h-1G

#python3 main.py --configfile R8
python3 main.py --configfile R8_new

#python3 main.py --configfile R8 --batch_size 8 --num_gcn_channels 4
python3 main.py --configfile R8_new --batch_size 8 --num_gcn_channels 4
