#!/bin/bash

NNODES=1
NPROC_PER_NODE=1
let BATCH=256
#let BATCH=256
#let BATCH=4096
BUDGET=0
#BUDGET=1000
#LAYERS=4
LAYERS=8
EPOCHS=4
SIMULATIONS=20
#STRATEGY="results_19_1/batch_${BATCH}/${LAYERS}_layers/test_strategy.json"
#STRATEGY="results_19_1/batch_${BATCH}/${LAYERS}_layers/test_strategy2.json"
#STRATEGY="results_19_1/batch_${BATCH}/${LAYERS}_layers/strategy.json"
#TEST_FOLDER1="results_19_1"
#TEST_FOLDER2="results_19_1/batch_${BATCH}"
#TEST_FOLDER3="results_19_1/batch_${BATCH}/${LAYERS}_layers"
#STRATEGY="results_19_1/batch_${BATCH}/${LAYERS}_layers/vanilla_strategy.json"
#OUTPUT="results_19_1/weak_strong_scaling_graphs/${LAYERS}_layers_output.txt"
TEST_FOLDER1="results_19_2"
TEST_FOLDER2="results_19_2/batch_${BATCH}"
TEST_FOLDER3="results_19_2/batch_${BATCH}/${LAYERS}_layers"

#STRATEGY="strategies/vanilla_strategy.json"
#STRATEGY="strategies/row_paral_strategy.json"
#STRATEGY="strategies/column_paral_strategy.json"
#HOST_NODE_ADDR=dudu-set-0.dudu-jax.pelegdav.svc.cluster.local:29600
#HOST_NODE_ADDR=dudu-set-1.dudu-jax.pelegdav.svc.cluster.local:29600
#OPTIMIZER='EGRL'
OPTIMIZER='MCMC'
export JOBTIMEOUT_SECONDS=100
export SERVICE_URL='http://nat3-service.mmarder.aipg-rancher-amr.intel.com'
export CODEID='dudupe'

for ((GPU=1 ; GPU<=$NPROC_PER_NODE; GPU+=1))
do
  echo "simulating ${GPU} gpus"
  if ! [[ -d ${TEST_FOLDER1} ]]; then
    `mkdir ${TEST_FOLDER1}`
  fi
  if ! [[ -d ${TEST_FOLDER2} ]]; then
    `mkdir ${TEST_FOLDER2}`
  fi
  if ! [[ -d ${TEST_FOLDER3} ]]; then
    `mkdir ${TEST_FOLDER3}`
  fi
  VANILLA_STRATEGY="results_19_2/batch_${BATCH}/${LAYERS}_layers/GPU${GPU}_vanilla_strategy.json"
  STRATEGY="results_19_2/batch_${BATCH}/${LAYERS}_layers/GPU${GPU}_strategy.json"
  OUTPUT1="results_19_2/batch_${BATCH}/${LAYERS}_layers/custom_dp_output.txt"
  OUTPUT2="results_19_2/batch_${BATCH}/${LAYERS}_layers/ddp_output.txt"
  OUTPUT3="results_19_2/batch_${BATCH}/${LAYERS}_layers/mcmc_output.txt"
  # todo uncomment to support different batch sizes
  #	for (( BATCH=64; BATCH<=$BATCHES; BATCH*=2))
  #	do
  #		TEMP_BATCH=$BATCH
  #		echo "simulating batch size ${BATCH}. simulating ${GPU} gpus"
  #		EXPORT_FOLDER=${HIGH_FOLDER}/batch_${BATCH}
  #		if ! [ -d $EXPORT_FOLDER ]; then
  #			`mkdir $EXPORT_FOLDER`
  #		fi
  #		# these lines are added in order to support batch binning
        # todo uncomment this for weak scaling
        echo "Weak scaling" | tee -a ${OUTPUT}
        N_BATCH=$((BATCH))
#        # todo uncomment this for strong scaling
#        echo "Strong scaling" | tee -a ${OUTPUT}
#        N_BATCH=$((BATCH / GPU))
  ##     todo uncomment this only if youre using mcmc
  #  		if [[ $(( $N_BATCH % $GPU )) != 0 ]]; then
  #  			while (( $N_BATCH % $GPU != 0 ))
  #  			do
  #  				N_BATCH=$((N_BATCH-1))
  #  				echo "batch is indivisible by gpu amount. current batch:" $N_BATCH " current gpu:" $GPU
  #  			done
  #  		fi
#    echo "torchrun \
#      --nnodes=$NNODES --nproc_per_node=$GPU \
#      --rdzv_id=1236 --rdzv_backend=c10d \
#      --rdzv_endpoint=$HOST_NODE_ADDR \
#      mnist_mlp.py --batch_size ${N_BATCH} --layers ${LAYERS}"
#
#    torchrun \
#      --nnodes=$NNODES --nproc_per_node=$GPU \
#      --rdzv_id=1236 --rdzv_backend=c10d \
#      mnist_mlp.py --batch_size ${N_BATCH} --budget 0 --print_params --export_strategy ${VANILLA_STRATEGY} --layers ${LAYERS} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} 2>&1 | tee -a ${OUTPUT1}
#  #  	mnist_mlp.py --batch_size ${N_BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --import_strategy ${STRATEGY} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} 2>&1 | tee -a ${OUTPUT}
#  #    ddp_mlp.py --batch_size ${N_BATCH} --layers ${LAYERS} 2>&1 | tee -a ${OUTPUT}

  echo "torchrun \
      --nnodes=$NNODES --nproc_per_node=$GPU \
      --rdzv_id=1236 --rdzv_backend=c10d \
      --rdzv_endpoint=$HOST_NODE_ADDR \
      mnist_mlp.py --batch_size ${N_BATCH} --layers ${LAYERS}"

    torchrun \
      --nnodes=$NNODES --nproc_per_node=$GPU \
      --rdzv_id=1236 --rdzv_backend=c10d \
      ddp_mlp.py --batch_size ${N_BATCH} --layers ${LAYERS} 2>&1 | tee -a ${OUTPUT2}
      #      mnist_mlp.py --batch_size ${N_BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} 2>&1 | tee -a ${OUTPUT}
#    	mnist_mlp.py --batch_size ${N_BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --import_strategy ${STRATEGY} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} 2>&1 | tee -a ${OUTPUT}

#  echo "torchrun \
#      --nnodes=$NNODES --nproc_per_node=$GPU \
#      --rdzv_id=1236 --rdzv_backend=c10d \
#      --rdzv_endpoint=$HOST_NODE_ADDR \
#      mnist_mlp.py --batch_size ${N_BATCH} --layers ${LAYERS}"
#
#    torchrun \
#      --nnodes=$NNODES --nproc_per_node=$GPU \
#      --rdzv_id=1236 --rdzv_backend=c10d \
#      mnist_mlp.py --batch_size ${N_BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --import_strategy ${STRATEGY} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} 2>&1 | tee -a ${OUTPUT3}
##      mnist_mlp.py --batch_size ${N_BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} 2>&1 | tee -a ${OUTPUT3}
#  #    ddp_mlp.py --batch_size ${N_BATCH} --layers ${LAYERS} 2>&1 | tee -a ${OUTPUT}
done
# todo for multinode, use this to define a communication server
#torchrun \
#	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#	--rdzv_id=1236 --rdzv_backend=c10d \
#	--rdzv_endpoint=$HOST_NODE_ADDR \
#	ddp_mlp.py --batch_size ${BATCH} --layers ${LAYERS} | tee -a ${OUTPUT}
##	mnist_mlp.py --batch_size ${BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --import_strategy ${STRATEGY} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} | tee -a ${OUTPUT}
##	mnist_mlp.py --batch_size ${BATCH} --budget ${BUDGET} --print_params --layers ${LAYERS} --export_strategy ${STRATEGY} --epochs ${EPOCHS} --n_simulations ${SIMULATIONS} --optimization_alg ${OPTIMIZER} | tee -a ${OUTPUT}


#LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
