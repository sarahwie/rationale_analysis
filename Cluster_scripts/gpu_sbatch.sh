#!/bin/bash

#SBATCH --nodes=1                                 
#SBATCH --tasks-per-node=1                        
#SBATCH --cpus-per-task=1                         
#SBATCH --mem=40Gb                               
#SBATCH --time=12:00:00                            
#SBATCH --output=%x.%j.out               
#SBATCH --error=%x.%j.err            
#SBATCH --mail-type=ALL    
#SBATCH --mail-user=successar@gmail.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1

source /home/jain.sar/Rationale_Analysis/Rationale_Analysis/discovery_cluster_essentials.sh

bash $1