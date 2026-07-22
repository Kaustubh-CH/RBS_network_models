#!/bin/bash
#SBATCH --job-name=plot_and_report
#SBATCH -A m2043
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --mail-user=kchakravarthul@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH --output =./NERSC/output/plot_report_output.txt

module load conda
conda activate preshifterv2
#job_path=NERSC/output/240529_Run1_8nodes
#job_path=NERSC/output/240530_Run1_8nodes
#job_path=NERSC/output/240604_Run1_24n24hr
job_path=NERSC/output/240808_Run1_testing
job_path=NERSC/output/251017_Run1_24n24hr

#normal
# echo "Running plot commands..."
#python NERSC/plot_sims.py --job_dirs NERSC/output/240529_Run1_8nodes --start_gen 1 --verbose

#parallel
#final_gen=14
#njobs=$(($SLURM_NNODES*2))
module load parallel
echo "Running commands in parallel..."
#seq 0 $final_gen | parallel python NERSC/plot_sims.py --job_dirs $job_path --gen {} #parallelize over generations
parallel_cands=10
num_gens=20
njobs=1

#debugging
# parallel_cands=1
# num_gens=1
# njobs=1
for ((par_gen=0; par_gen<=1; par_gen++))
do
    python -m pdb NERSC/plot_sims.py --job_dirs $job_path --gen $par_gen --cand 0  #run for candidate 0 to initialize files
    # seq 0 $parallel_cands | parallel -j $njobs python -m pdb NERSC/plot_sims.py --job_dirs $job_path --gen $par_gen --cand {} #parallelize over candidates, -j jobs at a time
done

#generate pdf reports
#python NERSC/generate_pdf_reports.py

#After selecting HOF additions, run the following to generate the final report
#python NERSC/plot_sims.py --HOF
#python NERSC/generate_pdf_reports.py --HOF

### ARCHIVE
# Create commands.txt file
# echo "Creating plot_commands.txt file..."
# rm -f plot_commands.txt
# for i in $(seq 0 $final_gen); do
#     echo "
#         srun -N 1 -n 1 --sockets-per-node 1 --cpu_bind=cores \ 
#         module load conda && \
#         conda activate preshifter && \
#         python NERSC/plot_sims.py --job_dirs $job_path --gen $i" >> plot_commands.txt
# done

# Run commands in parallel
#echo "Running commands in parallel..."
#parallel --jobs ${njobs} :::: plot_commands.txt

#regular in series plotting
#python NERSC/plot_sims.py --job_dirs NE#RSC/output/240529_Run1_8nodes --start_gen 1 --verbose
# python NERSC/plot_sims.py --job_dirs NERSC/output/240530_Run1_8nodes 
#