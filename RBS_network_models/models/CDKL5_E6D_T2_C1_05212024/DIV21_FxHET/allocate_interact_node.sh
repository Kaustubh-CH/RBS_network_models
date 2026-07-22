#bin/bash

# more modular version

# Set the number of nodes
NODES=1
#NODES=4 # Modify as needed

# Get total logical CPUs per node (includes hyperthreads)
CPUS_PER_NODE=$(lscpu | awk '/^CPU\(s\):/{print $2}')

# Get number of sockets per node
SOCKETS_PER_NODE=$(lscpu | awk '/^Socket\(s\):/{print $2}')

# Get cores per socket
CORES_PER_SOCKET=$(lscpu | awk '/^Core\(s\) per socket:/{print $4}')

# Get total cores per node
CORES_PER_NODE=$(( SOCKETS_PER_NODE * CORES_PER_SOCKET ))

# Get logical CPUs per core (hyperthreading ratio)
THREADS_PER_CORE=$(( CPUS_PER_NODE / CORES_PER_NODE ))

# Define the number of logical CPUs per task
CPUS_PER_TASK=1  # Modify as needed, this seem to be optimal for simulations distributed across multiple MPI tasks

# Calculate number of tasks per node (ensuring each task fits within a socket)
TASKS_PER_NODE=$(( CPUS_PER_NODE / CPUS_PER_TASK ))

# Echo calculated values
echo "Nodes: $NODES"
echo "Total logical CPUs per node: $CPUS_PER_NODE"
echo "Sockets per node: $SOCKETS_PER_NODE"
echo "Cores per socket: $CORES_PER_SOCKET"
echo "Logical CPUs per core (Threads per core): $THREADS_PER_CORE"
echo "CPUs per task: $CPUS_PER_TASK"
echo "Number of tasks per node: $TASKS_PER_NODE"
echo "Total tasks: $(( NODES * TASKS_PER_NODE ))"

# Run the salloc command
salloc -A m2043 -q interactive -C cpu -t 04:00:00 \
  --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE --cpus-per-task=$CPUS_PER_TASK \
  --image=adammwea/netsims_docker:v1 \
  --threads-per-core=$THREADS_PER_CORE --hint=socket


# 2025-06-28 10:33:07 - proof of concept, this works for running the interactive job allocation and immediately executing commands within the allocated resources
# NOTE: 
#   -- ignore the srun errors, they dont affect anything -- interactive nodes are just not meant to be run like this.
#   -- A nice benefit of this approach is that it only runs for however long needed to run the commands, may save on hours used provided run time < 4 hours
# salloc \
#         -A m2043 -q interactive -C cpu -t 04:00:00 \
#         --nodes="$NODES" \
#         --ntasks-per-node="$TASKS_PER_NODE" \
#         --cpus-per-task="$CPUS_PER_TASK" \
#         --image=adammwea/netsims_docker:v1 \
#         --threads-per-core="$THREADS_PER_CORE" \
#         --hint=socket \
# <<EOF
# echo "salloc granted on: \$SLURM_JOB_NODELIST"
# echo "  Nodes:          $NODES"
# echo "  CPUs per node:  $CPUS_PER_NODE"
# echo "  Sockets/node:   $SOCKETS_PER_NODE"
# echo "  Cores/socket:   $CORES_PER_SOCKET"
# echo "  Threads/core:   $THREADS_PER_CORE"
# echo "  CPUs per task:  $CPUS_PER_TASK"
# echo "  Tasks per node: $TASKS_PER_NODE"
# echo "  Total tasks:    $(( NODES * TASKS_PER_NODE ))"

# echo "All done at $(date)"
# EOF