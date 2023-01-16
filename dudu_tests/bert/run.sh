#!/bin/bash

NNODES=1
NPROC_PER_NODE=8
HOST_NODE_ADDR=dudu-set-0.dudu-jax.pelegdav.svc.cluster.local:29600
#let BATCH=16
#let BATCH=64
#let BATCH=256
BATCHES=( 16 )
#BATCHES=( 16 64 256 )
BUDGET=0
#BUDGET=1000
#ENCODERS=2
ENCODERS=12
SIMULATIONS=20


#cmd="torchrun \
#	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#	--rdzv_id=1230 --rdzv_backend=c10d \
#	--rdzv_endpoint=$HOST_NODE_ADDR \
#	custom_strategy_bert.py"
for ((GPU=1 ; GPU<=$NPROC_PER_NODE; GPU+=1))
do
  for BATCH in "${BATCHES[@]}"
  do
    STRATEGY="new_results/${ENCODERS}_layer/gpus${GPU}_batch_${BATCH}_${ENCODERS}_layer_strategy.json"
    VANILLA_STRATEGY="new_results/gpus${GPU}_batch_${BATCH}_${ENCODERS}_layer_vanilla_strategy5.json"
    VANILLA_OUTPUT="new_results/gpus${GPU}_batch_${BATCH}_${ENCODERS}_layer_vanilla_output6.txt"
    OUTPUT="new_results/gpus${GPU}_batch_${BATCH}_${ENCODERS}_layer_strategy_output6.txt"
    DDP_OUTPUT="new_results/gpus${GPU}_batch_${BATCH}_${ENCODERS}_layer_ddp_output.txt"

#     echo "torchrun \
#    --nnodes=$NNODES --nproc_per_node=$GPU \
#    --rdzv_id=1230 --rdzv_backend=c10d \
#    --rdzv_endpoint=$HOST_NODE_ADDR \
#    bert_finalized.py --batch_size ${BATCH} --budget 0 --import_strategy ${VANILLA_STRATEGY} --export_strategy ${VANILLA_STRATEGY} --print_params --encoder_layers ${ENCODERS} --n_simulations ${SIMULATIONS} \
#     2>&1 | tee -a ${OUTPUT}"
#
#    torchrun \
#      --nnodes=$NNODES --nproc_per_node=$GPU \
#      --rdzv_id=1230 --rdzv_backend=c10d \
#      bert_finalized.py --batch_size ${BATCH} --budget 0 --print_params --export_strategy ${VANILLA_STRATEGY} --encoder_layers ${ENCODERS} --n_simulations ${SIMULATIONS} \
#       2>&1 | tee -a ${OUTPUT}
#
    echo "torchrun \
      --nnodes=$NNODES --nproc_per_node=$GPU \
      --rdzv_id=1230 --rdzv_backend=c10d \
      --rdzv_endpoint=$HOST_NODE_ADDR \
      bert_finalized.py --batch_size ${BATCH} --budget ${BUDGET} --import_strategy ${STRATEGY} --export_strategy ${STRATEGY} --print_params --encoder_layers ${ENCODERS} --n_simulations ${SIMULATIONS} \
       2>&1 | tee -a ${OUTPUT}"

    torchrun \
      --nnodes=$NNODES --nproc_per_node=$GPU \
      --rdzv_id=1230 --rdzv_backend=c10d \
      bert_finalized.py --batch_size ${BATCH} --budget ${BUDGET} --import_strategy ${STRATEGY} --export_strategy ${STRATEGY} --print_params --encoder_layers ${ENCODERS} --n_simulations ${SIMULATIONS} \
       2>&1 | tee -a ${OUTPUT}

#     echo "torchrun \
#      --nnodes=$NNODES --nproc_per_node=$GPU \
#      --rdzv_id=1230 --rdzv_backend=c10d \
#      --rdzv_endpoint=$HOST_NODE_ADDR \
#      bert_finalized_ddp.py --batch_size ${BATCH} --budget ${BUDGET} --import_strategy ${STRATEGY} --export_strategy ${STRATEGY} --print_params --encoder_layers ${ENCODERS} --n_simulations ${SIMULATIONS} \
#       2>&1 | tee -a ${DDP_OUTPUT}"
#
#    torchrun \
#      --nnodes=$NNODES --nproc_per_node=$GPU \
#      --rdzv_id=1230 --rdzv_backend=c10d \
#      bert_finalized_ddp.py --batch_size ${BATCH} --budget ${BUDGET} --import_strategy ${VANILLA_STRATEGY} --export_strategy ${VANILLA_STRATEGY} --print_params --encoder_layers ${ENCODERS} --n_simulations ${SIMULATIONS} \
#       2>&1 | tee -a ${DDP_OUTPUT}


    # bert_finalized.py --batch_size ${BATCH} --budget ${BUDGET} --export_strategy ${STRATEGY} --print_params --encoder_layers ${ENCODERS} --n_simulations ${SIMULATIONS} \
    #	 2>&1 | tee -a ${OUTPUT}

  done
done
#	 bert_finalized.py --batch_size ${BATCH} --import_strategy ${STRATEGY} --budget ${BUDGET} --export_strategy ${EXPORT_STRATEGY} --print_params --encoder_layers ${ENCODERS} --n_simulations ${SIMULATIONS} \
#	 2>&1 | tee -a output_new.txt
#	bert_with_strategy.py --import_strategy ${STRATEGY} --budget ${BUDGET} --export_strategy ${EXPORT_STRATEGY} --print_params 2>&1 | tee -a output_new.txt
##	custom_strategy_bert.py --strategy strategies/bert_512_emb_12_layers_32_batch_4GPUs_no_parallel.json  --print_params"
#	custom_strategy_bert.py --print_params --import_strategy ${STRATEGY} --export_strategy ${EXPORT_STRATEGY}"
#torchrun \
#	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#	--rdzv_id=1230 --rdzv_backend=c10d \
#	--rdzv_endpoint=$HOST_NODE_ADDR \
#	multi_node_train.py

#LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
