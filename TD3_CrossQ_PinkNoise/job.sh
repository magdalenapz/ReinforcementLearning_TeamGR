#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=arnesjob

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# ML Cloud: 2080-galvani
# TCML: day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=1-00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"


#SBATCH --error=outputs/job-%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=outputs/job-%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=arne.guenther@student.uni-tuebingen.de
# your mail address


####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

#singularity exec --nv ~/TCML-CUDA12_4_TF2_17_PT_2_4.simg bash benchmark.sh $1
#singularity exec --nv ~/TCML-CUDA12_4_TF2_17_PT_2_4.simg bash selfplay.sh $1
singularity exec --nv ~/TCML-CUDA12_4_TF2_17_PT_2_4.simg python -u main.py --env Hockey-SelfPlay --policy CrossQ --heat 256 --policy_buffer heat=256 --pink_noise --eval_freq 2000000 #--max_timesteps 300000
done

echo DONE!