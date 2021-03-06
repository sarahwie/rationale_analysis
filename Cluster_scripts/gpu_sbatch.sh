#!/bin/bash

#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=40Gb                               
#SBATCH --time=8:00:00                            
#SBATCH --output=/scratch/jain.sar/Rationale_Analysis/runs/%x.%j.out               
#SBATCH --error=/scratch/jain.sar/Rationale_Analysis/runs/%x.%j.err            
#SBATCH --mail-type=ALL    
#SBATCH --mail-user=successar@gmail.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1

source /home/jain.sar/Fresh/Rationale_Analysis/discovery_cluster_essentials.sh

$1 $2