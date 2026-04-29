#!/bin/bash

# Array of schemas to run
SCHEMAS=("schema_v1" "schema_v2" "schema_v3" "schema_v4" "schema_v5")

for SCHEMA in "${SCHEMAS[@]}"; do
    echo "Submitting batch job for $SCHEMA"
    sbatch --export=ALL,SCHEMA_NAME=$SCHEMA --job-name="Evol_$SCHEMA" /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/config_and_run_multi_schema.sh
done
