#!/bin/sh
#SBATCH --job-name=R52 # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j%x.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl2_48h-1G

#python3 main.py --configfile R52
python3 main.py --configfile R52_new

#python3 main.py --configfile R52 --batch_size 8 --num_gcn_channels 4
python3 main.py --configfile R52_new --batch_size 8 --num_gcn_channels 4
