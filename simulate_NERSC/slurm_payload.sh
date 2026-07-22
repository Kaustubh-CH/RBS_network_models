#!/bin/bash
#SBATCH --job-name=24n24hr
#SBATCH -A m2043
#SBATCH -t 24:00:00
#SBATCH -N 18
#SBATCH --mail-user=kchakravarthula@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/latest_job_init_output.txt
#---- SBATCH --error=./NERSC/output/latest_job_init_error.txt
#SBATCH --image=adammwea/netpyneshifter:v5

#### SRUN PAYLOAD METHOD USING GNU PARALLEL MODULE AT NERSC - HIGHLY PARALLELIZED AND SCALABLE
# Run this script with the following commands (modulate nodes and time as needed):
# Q0S JOB: 
#   sbatch slurm_payload.sh
# INTERACTIVE NODE:
#   salloc --nodes=2 -C cpu -q interactive -t 04:00:00 --exclusive --image=adammwea/netpyneshifter:v5
# bash slurm_payload.sh

### OPTIONS
Batch_Run_Label=$SLURM_JOB_NAME #Label for the batch run. This will be 'interactive in the case of an interactive node runs'
[ -z "$Batch_Run_Label" ] && Batch_Run_Label="login" # If running on login node, Batch_Run_Label will be empty. Set it to 'login' in this case.
Duration_Seconds=180 #Simulation Duration in seconds
nodes=$SLURM_NNODES #hpc, number of nodes as requested by salloc
#nodes=1 #override as needed. If running on laptop or server, number of nodes is 1. #NOTE: if running on a laptop or server, you will probably need to run in docker container with srun and slurm installed.
max_gens=3000 #number of generations to run, generally a high number to allow for convergence
#max_gens=1 #override as needed

gen_pop=512 #population per generation

### PARALLELIZATION PARAMS
OMP_threads_per_process=1 #recomended 1 cpu per process NERSC, https://docs.nersc.gov/development/languages/python/parallel-python/#numpy-and-nested-threading
export OMP_NUM_THREADS=$OMP_threads_per_process

##load modules and activate conda env
echo "loading modules and setting enviornment variables..."
module load conda
conda activate preshifterv2
# module load parallel

### RUN
##init batchRun folder
echo "Batch_Run_Label:" $Batch_Run_Label
#run_path=$(python3 NERSC/init_batch_run.py --label ${Batch_Run_Label})
python NERSC/init_batch_run.py --label ${Batch_Run_Label}
#get run_path from NERSC/temp/run_path.txt
run_path=$(cat NERSC/temp/run_path.txt)
echo "batchRun folder path created:" $run_path

##copy this bashScript to the run_path
full_sh_path=$(realpath $0)
echo "bashScript Path: ${full_sh_path}"
script_name=$(basename $0) #get the name of this bashScript file
full_sh_path=${full_sh_path} 
cp ${full_sh_path} ${run_path}/${script_name} #save copy of this bashScript in the run_path
echo "bashScript copied to run_path"

## Report allocation specs
# SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo "SLURM Environment"
    echo "Nodes: $SLURM_JOB_NUM_NODES"
    echo "Physical cores per node: $(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')"
    echo "Logical CPUs per node: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
    echo "SLURM CPUs per node: $SLURM_CPUS_ON_NODE"
    echo "Sockets per node: $(lscpu | grep 'Socket(s):' | awk '{print $2}')"
    echo "cores per socket: $(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')"
# Non-SLURM environment
else
    echo "Non-SLURM Environment"
    echo "Nodes: $(hostname | wc -l)"
    echo "Physical cores per node: $(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')"
    echo "Logical CPUs per node: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
    echo "Sockets per node: $(lscpu | grep 'Socket(s):' | awk '{print $2}')"
    echo "cores per socket: $(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')"
fi

## Report processes allocation based on environmnet
echo "Allocating 1 simulation per socket for optimal memory usage (i.e. 2 per node)..."
njobs=$((nodes*2)) #generally 2 jobs per node, 1 per socket
#njobs=2 #override as needed
echo "Allocating $njobs parallel simulations across $nodes nodes at a time..."
# Determine the number of physical cores per node
cores_per_node=$(lscpu | grep '^Core(s) per socket:' | awk '{print $4}')
# Calculate the number of tasks as 75% of the physical cores per node
# This allows for some fault tolerance and lets packages like numpy use multiple cores if needed
ntasks=$(echo "$cores_per_node*0.75" | bc)
ntasks=${ntasks%.*} # Remove the decimal part, if any
echo "Allocating $ntasks parallel cell populations per simulation in each socket..."
echo "Initializing $gen_pop simulation(s) cfgs per generation, $Duration_Seconds seconds each..."

## test 5 - this solution works!:
# - trying loop method
# - putting $ in python command makes it run in the background...sick

##init run
echo "Initializing paralellized simulations commands..."
cd NERSC
echo '' > commands.txt #clear commands.txt
(
python -u batchRun_payload.py \
    --run_path ${run_path} \
    --duration ${Duration_Seconds} \
    --label ${Batch_Run_Label} \
    --pop_size ${gen_pop} \
    --max_generations ${max_gens} \
    --hof \
    --mpi_type 'mpi_direct'\
    --tasks ${ntasks} \
    & export PID=$! #run batchRun_payload.py in the background
) >> "$run_path/${Batch_Run_Label}_output.run" 2>"$run_path/${Batch_Run_Label}_error.err"
gen_count=$(ls -l ${run_path} | grep -c -P '^d.*gen_\d+$') # Check for folders named gen_# in run_path, get the number of the latest generation
echo "gen_count:" $gen_count
current_gen=$((gen_count-1)) #latest generation is the current generation (if negative, set to 0 inside wait_for_current_gen.sh)
current_gen=$((current_gen<0?0:current_gen)) #if current_gen is -1 then it will be set to 0 in wait_for_current_gen.sh
echo "Waiting for collection of commands for generation:" $current_gen
bash wait_for_commands.sh $gen_pop &>> "$run_path/${Batch_Run_Label}_wait_output.run" #wait for batchRun_*.py to finish sending the next srun commands to commands.txt
echo "Commands collected in commands.txt"

##Loop through generations
gen_count=0
while (($gen_count < $max_gens)); do
    # Run commands in parallel and clear commands.txt
    #echo "Running commands in parallel"
    echo "Delivering Payload..."
    parallel --jobs ${njobs} echo :::: commands.txt
    echo "Running parallel simulations..."
    parallel --jobs ${njobs} :::: commands.txt
    echo '' > commands.txt #clear commands.txt
    echo "Parallel commands finished"    
    # re-init
    bash wait_for_commands.sh $gen_pop &>> "$run_path/${Batch_Run_Label}_wait_output.run" #wait for batchRun_*.py to finish sending the next srun commands to commands.txt
    echo "Commands added to commands.txt"
    gen_count=$((gen_count+1))    
done

#test 4:
# - trying ::::
# - takes a few minutes, but this runs the first python command at least.
# - yea it get's stuck after the first command. probably waiting for the first command to finish....which it never will.
#echo "python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#parallel --jobs ${njobs} echo :::: commands.txt
#parallel --jobs ${njobs} :::: commands.txt

#test3:
# - now not even the first command is running.
# - setting 12 minute time limit.
# - Nope this doesnt work. Nothing runs.
#echo "srun -N 1 -n 1 -c 1 python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#srun parallel --jobs ${njobs} < commands.txt

#test2:
# - batchRun_payload.py runs. Not sure if parallel sruns are working.
# - I'm giving it 12 minutes to show evidence of running in parallel.
# - doesnt seem to start after running batchRun_payload.py. Probably because I didnt srun so it only does 1 job at a time.
#echo "srun -N 1 -n 1 -c 1 python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#parallel --jobs ${njobs} < commands.txt

#test 1: /usr/bin/bash: argument_srun: command not found
#echo "srun -N 1 -n 1 -c 1 python batchRun_payload.py -rp ${container_run_path} -d ${Duration_Seconds} -l ${Batch_Run_Label}" > commands.txt
#parallel --jobs ${njobs} argument_{} :::: commands.txt

# python -m pdb batchRun_payload.py  --run_path '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/NERSC/output/test_run' --duration 10 --label 'test_run' --pop_size 10 --max_generations 10 --hof --mpi_type 'mpi_direct' --tasks 2





