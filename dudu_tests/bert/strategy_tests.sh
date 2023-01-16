#!/bin/bash

NNODES=1
NPROC_PER_NODE=4
HOST_NODE_ADDR=dudu-set-0.dudu-jax.pelegdav.svc.cluster.local:29600
let BATCH=32
BUDGET=1
STRATEGY="strategies/strategy.json"
EXPORT_STRATEGY="strategies/exported_strat.json"
#cmd="torchrun \
#	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#	--rdzv_id=1230 --rdzv_backend=c10d \
#	--rdzv_endpoint=$HOST_NODE_ADDR \
#	custom_strategy_bert.py"

cmd="torchrun \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1230 --rdzv_backend=c10d \
	--rdzv_endpoint=$HOST_NODE_ADDR \
	strategy_tests.py"
#	custom_strategy_bert.py --strategy strategies/bert_512_emb_12_layers_32_batch_4GPUs_no_parallel.json"
#	custom_strategy_bert.py --print_params --import_strategy ${STRATEGY} --export_strategy ${EXPORT_STRATEGY}"

LOGLEVEL=WARN NCCL_DEBUG=WARN exec $cmd
