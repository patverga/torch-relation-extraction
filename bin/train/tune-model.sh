#!/usr/bin/env bash

#!/bin/sh

config=$1
additional_args=${@:2}

source $config
export SAVE_MODEL=""
export MAX_EPOCHS=25
export EVAL_FREQ=5

OUT_LOG=$LOG_ROOT/hyperparams/
mkdir -p $OUT_LOG
echo "Writing to "$OUT_LOG

source ${TH_RELEX_ROOT}/bin/train/gen-run-cmd.sh
RUN_CMD="$RUN_CMD $additional_args"

# run on all available gpus
#gpus=`nvidia-smi -L | wc -l`
gpuids=( `eval $TH_RELEX_ROOT/bin/get-free-gpus.sh | sed '1d'` )
num_gpus=${#gpuids[@]}

# grid search over these
lrs="0.0001 0.001 0.01"
dropouts="0.0" # 0.1 0.25"
clipgrads="1 10" # 0.1 0.25"
l2s="0 1e-8 1e-6"
epsilons="1e-8 1e-4"
dims="300"
batchsizes="256 512 1024"

# array to hold all the commands we'll distribute
declare -a commands

# first make all the commands we want
for dim in $dims
do
   for lr in $lrs
   do
       for l2 in $l2s
       do
           for batchsize in $batchsizes;
           do
               for clipgrad in $clipgrads;
               do
                   for dropout in $dropouts;
                   do
                       for epsilon in $epsilons;
                       do
                           CMD="$RUN_CMD \
                                -colDim $dim \
                                -rowDim $dim \
                                -learningRate $lr \
                                -l2Reg $l2 \
                                -epsilon $epsilon \
                                -batchSize $batchsize \
                                -dropout $dropout \
                                -clipGrads $clipgrad \
                                -gpuid XX \
                                &> $OUT_LOG/train-$lr-$dim-$dropout-$clipgrad-$l2-$epsilon-$batchsize.log"
                           commands+=("$CMD")
                           echo "Adding job lr=$lr dim=$dim dropout=$dropout l2=$l2 batchsize=$batchsize epsilon=$epsilon"
                       done
                    done
                done
           done
       done
   done
done

# now distribute them to the gpus
#
# currently this is only correct if the number of jobs is a 
# multiple of the number of gpus (true as long as you have hyperparams
# ranging over 2, 3 and 4 values)!
num_jobs=${#commands[@]}
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for gpuid in ${gpuids[@]}; do
    for (( i=0; i<$jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]/XX/$gpuid}"
        echo "Starting job $jobid on gpu $gpuid"
        eval ${comm}
    done &
    j=$((j + 1))
done
