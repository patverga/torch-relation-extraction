#!/usr/bin/env bash

config=$1
additional_args=${@:2}

source $config
#export SAVE_MODEL=""
export MAX_EPOCHS=-1
export EVAL_FREQ=1

OUT_LOG=$LOG_ROOT/hyperparams/
mkdir -p $OUT_LOG
echo "Writing to "$OUT_LOG

source ${TH_RELEX_ROOT}/bin/train/gen-run-cmd.sh

# run on all available gpus
#gpus=`nvidia-smi -L | wc -l`
#gpuids=( `eval $TH_RELEX_ROOT/bin/get-free-gpus.sh | sed '1d'` )
gpuids=( 0 2 1 3 )
num_gpus=${#gpuids[@]}

# grid search over these
lrs=".01 .001"
dropouts="0.0"
hiddenDropouts="0.0"
wordDropouts="0.0"
clipgrads="0" # 1 10"
l2s="1e-8 0"
epsilons="1e-8"
dims="25"
tokenDims="200"
batchsizes="512 1024 2048"
negsamples="2 200"

# array to hold all the commands we'll distribute
declare -a commands

# first make all the commands we want
for dim in $dims
do
   for lr in $lrs
   do
       for tokenDim in $tokenDims
       do
           for l2 in $l2s
           do
               for batchsize in $batchsizes;
               do
                   for negsample in $negsamples;
                   do
                       for clipgrad in $clipgrads;
                       do
                           for dropout in $dropouts;
                           do
                               for hiddendropout in $hiddenDropouts;
                               do
                                   for worddropout in $wordDropouts;
                                       do
                                       for epsilon in $epsilons;
                                       do
                                           JOB_LOG="$OUT_LOG/train-$lr-$tokenDim-$dim-$dropout-$hiddendropout-$worddropout-$clipgrad-$l2-$epsilon-$batchsize-${negsample}.log"
            #                               if [ ! -f "$JOB_LOG" ]; then
                                               RESULT_DIR="$OUT_LOG/tac/$lr-$tokenDim-$dim-$dropout-$hiddendropout-$worddropout-$clipgrad-$l2-$epsilon-$batchsize-${negsample}"
                                               echo $JOB_LOG
                                               CMD="$RUN_CMD \
                                                    -colDim $dim \
                                                    -rowDim $dim \
                                                    -learningRate $lr \
                                                    -l2Reg $l2 \
                                                    -tokenDim $tokenDim \
                                                    -epsilon $epsilon \
                                                    -batchSize $batchsize \
                                                    -dropout $dropout \
                                                    -hiddenDropout $hiddendropout \
                                                    -wordDropout $worddropout \
                                                    -clipGrads $clipgrad \
                                                    -resultDir $RESULT_DIR \
                                                    -negSamples $negsample \
                                                    -gpuid XX $additional_args \
                                                    &> $JOB_LOG"
                                               commands+=("$CMD")
            #                               fi
                                       done
                                   done
                               done
                           done
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
        if [ "$gpuid" -ge 0 ]; then
            comm="CUDA_VISIBLE_DEVICES=$gpuid $comm -gpuid 0"
        fi
        eval ${comm}
    done &
    j=$((j + 1))
done
