#! /bin/bash

#SBATCH --account=p_masi_gpu
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=70:00:00
#SBATCH --output=/scratch/huoy1/projects/DeepLearning/Deep_5000_Brain/accre_log/train_log_$1_$2_$3.txt

setpkgs -a anaconda3
source activate python27
setpkgs -a cuda8.0

pieceval=$1_$2_$3
python train.py --piece=$pieceval --epoch=6
