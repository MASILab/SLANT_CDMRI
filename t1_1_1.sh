#! /bin/bash 
#SBATCH --account=p_masi_gpu 
#SBATCH --partition=pascal 
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=12G 
#SBATCH --time=70:00:00 
#SBATCH --output=/scratch/huoy1/projects/DeepLearning/Deep_5000_Brain/accre_log/train_log_1_1_1.txt 

setpkgs -a anaconda3 
source activate python27 
setpkgs -a cuda8.0 

python train.py --train_data_dir /share5/huoy1/CDMRI_Challenge_2018/TeamA/AtoB/Data/Training --test_data_dir /share5/huoy1/CDMRI_Challenge_2018/TeamA/AtoB/Data/Training --working_dir /share5/huoy1/CDMRI_Challenge_2018/working_dir/Experiment1
