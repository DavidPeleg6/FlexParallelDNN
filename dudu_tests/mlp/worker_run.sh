#!/bin/bash

NNODES=1
NPROC_PER_NODE=4
#let BATCH=16
#let BATCH=256
let BATCH=4096
BUDGET=0
#BUDGET=100
#LAYERS=4
LAYERS=32
EPOCHS=10
SIMULATIONS=20
#STRATEGY="results_19_1/batch_${BATCH}/${LAYERS}_layers/test_strategy2.json"
STRATEGY="results_19_1/batch_${BATCH}/${LAYERS}_layers/test_strategy.json"
#STRATEGY="results_19_1/batch_${BATCH}/${LAYERS}_layers/strategy.json"
#STRATEGY="results_19_1/batch_${BATCH}/${LAYERS}_layers/vanilla_strategy.json"
OUTPUT="results_19_1/batch_${BATCH}/${LAYERS}_layers/output.txt"
#STRATEGY="strategies/vanilla_strategy.json"
#STRATEGY="strategies/row_paral_strategy.json"
#STRATEGY="strategies/column_paral_strategy.json"
#HOST_NODE_ADDR=dudu-set-0.dudu-jax.pelegdav.svc.cluster.local:29600
HOST_NODE_ADDR=dudu-set-1.dudu-jax.pelegdav.svc.cluster.local:29600
#OPTIMIZER='EGRL'
OPTIMIZER='MCMC'
export JOBTIMEOUT_SECONDS=100
export SERVICE_URL='http://nat3-service.mmarder.aipg-rancher-amr.intel.com'
export CODEID='dudupe'


echo "torchrun \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1236 --rdzv_backend=c10d \
	--rdzv_endpoint=$HOST_NODE_ADDR \
	mnist_mlp.py --batch_size ${BATCH}"

torchrun \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1236 --rdzv_backend=c10d \
	--rdzv_endpoint=$HOST_NODE_ADDR \
	mnist_mlp.py --batch_size ${BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --import_strategy ${STRATEGY} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} | tee -a ${OUTPUT}
#	mnist_mlp.py --batch_size ${BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} | tee -a ${OUTPUT}
#	ddp_mlp.py --batch_size ${BATCH} --layers ${LAYERS} | tee -a ${OUTPUT}

#cmd="torchrun \
#	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#	--rdzv_id=1236 --rdzv_backend=c10d \
#	--rdzv_endpoint=$HOST_NODE_ADDR \
#	hybrid_paral_mlp.py --batch_size ${BATCH} --budget ${BUDGET} --print_params"
##	hybrid_paral_mlp.py --batch_size ${BATCH} --budget ${BUDGET}"

#LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
