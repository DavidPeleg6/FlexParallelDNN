#!/bin/bash

NNODES=1
NPROC_PER_NODE=4
let BATCH=64
BUDGET=0
EPOCHS=1
STRATEGY="strategy.json"
OUTPUT_FILE="testing_output.txt"
HOST_NODE_ADDR=pytorch-set-0.pytorch.pelegdav.svc.cluster.local:29600
#HOST_NODE_ADDR=pytorch-set-1.pytorch.pelegdav.svc.cluster.local:29600

# --------------------------- VANILLA DATA PARALLEL
echo 'testing runtime and ips in case that vanilla data parallel is called' >> ${OUTPUT_FILE}
echo '{"fc1": [false, false], "fc2": [false, false], "fc3": [false, false], "fc4": [false, false], "fc5": [false, false], "fc6": [false, false], "fc7": [false, false]}' >> ${OUTPUT_FILE}
echo '{"fc1": [false, false], "fc2": [false, false], "fc3": [false, false], "fc4": [false, false], "fc5": [false, false], "fc6": [false, false], "fc7": [false, false]}' > ${STRATEGY}
#torchrun \
#    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#    --rdzv_id=1236 --rdzv_backend=c10d \
#    --rdzv_endpoint=$HOST_NODE_ADDR \
#    mnist_mlp.py --batch_size ${BATCH} --strategy ${STRATEGY} --budget ${BUDGET} --print_params --epochs ${EPOCHS} | tee output.txt
torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1236 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR \
    mnist_mlp.py --batch_size ${BATCH} --strategy ${STRATEGY} --budget ${BUDGET} --profile --print_params --epochs ${EPOCHS} | tee output.txt
cat output.txt | grep "Best ips" | tee -a ${OUTPUT_FILE}
cat output.txt | grep "Accuracy" | tee -a ${OUTPUT_FILE}

# --------------------------- GATHERS AND SPLITS ON EVERY LAYER
echo 'testing runtime and ips in case that a split and gather is called on every layer. this case means no data parallel' >> ${OUTPUT_FILE}
echo '{"fc1": [true, true], "fc2": [true, true], "fc3": [true, true], "fc4": [true, true], "fc5": [true, true], "fc6": [true, true], "fc7": [true, true]}' >> ${OUTPUT_FILE}
echo '{"fc1": [true, true], "fc2": [true, true], "fc3": [true, true], "fc4": [true, true], "fc5": [true, true], "fc6": [true, true], "fc7": [true, true]}' > ${STRATEGY}
torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1236 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR \
    mnist_mlp.py --batch_size ${BATCH} --strategy ${STRATEGY} --budget ${BUDGET} --profile --print_params --epochs ${EPOCHS} | tee output.txt
cat output.txt | grep "Best ips" | tee -a ${OUTPUT_FILE}
cat output.txt | grep "Accuracy" | tee -a ${OUTPUT_FILE}

## --------------------------- GATHER ON FIRST LAYER, SPLIT ON LAST
#echo 'testing runtime and ips in case that a gather is called on first layer and split is called on last. this case means no data parallel' >> ${OUTPUT_FILE}
#echo '{"fc1": [true, false], "fc2": [false, false], "fc3": [false, false], "fc4": [false, false], "fc5": [false, false], "fc6": [false, false], "fc7": [false, true]}' >> ${OUTPUT_FILE}
#echo '{"fc1": [true, false], "fc2": [false, false], "fc3": [false, false], "fc4": [false, false], "fc5": [false, false], "fc6": [false, false], "fc7": [false, true]}' > ${STRATEGY}
#torchrun \
#    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#    --rdzv_id=1236 --rdzv_backend=c10d \
#    --rdzv_endpoint=$HOST_NODE_ADDR \
#    mnist_mlp.py --batch_size ${BATCH} --strategy ${STRATEGY} --budget ${BUDGET} --epochs ${EPOCHS} | tee output.txt
#cat output.txt | grep "Best ips" | tee -a ${OUTPUT_FILE}
#cat output.txt | grep "Accuracy" | tee -a ${OUTPUT_FILE}
#
## -------------------------- GATHER INPUT
#echo 'testing runtime and ips in case that a gather is called on an already gathered input.' >> ${OUTPUT_FILE}
#echo '{"fc1": [true, false], "fc2": [true, false], "fc3": [true, false], "fc4": [true, false], "fc5": [true, false], "fc6": [true, false], "fc7": [true, false]}' >> ${OUTPUT_FILE}
#echo '{"fc1": [true, false], "fc2": [true, false], "fc3": [true, false], "fc4": [true, false], "fc5": [true, false], "fc6": [true, false], "fc7": [true, false]}' > ${STRATEGY}
#torchrun \
#    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#    --rdzv_id=1236 --rdzv_backend=c10d \
#    --rdzv_endpoint=$HOST_NODE_ADDR \
#    mnist_mlp.py --batch_size ${BATCH} --strategy ${STRATEGY} --budget ${BUDGET} --epochs ${EPOCHS} | tee output.txt
#cat output.txt | grep "Best ips" | tee -a ${OUTPUT_FILE}
#cat output.txt | grep "Accuracy" | tee -a ${OUTPUT_FILE}
#
## -------------------------- SPLIT INPUT
#echo 'testing runtime and ips in case that a split is called on an already split input' >> ${OUTPUT_FILE}
#echo '{"fc1": [false, true], "fc2": [false, true], "fc3": [false, true], "fc4": [false, true], "fc5": [false, true], "fc6": [false, true], "fc7": [false, true]}' >> ${OUTPUT_FILE}
#echo '{"fc1": [false, true], "fc2": [false, true], "fc3": [false, true], "fc4": [false, true], "fc5": [false, true], "fc6": [false, true], "fc7": [false, true]}' > ${STRATEGY}
#torchrun \
#    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
#    --rdzv_id=1236 --rdzv_backend=c10d \
#    --rdzv_endpoint=$HOST_NODE_ADDR \
#    mnist_mlp.py --batch_size ${BATCH} --strategy ${STRATEGY} --budget ${BUDGET} --epochs ${EPOCHS} | tee output.txt
#cat output.txt | grep "Best ips" | tee -a ${OUTPUT_FILE}
#cat output.txt | grep "Accuracy" | tee -a ${OUTPUT_FILE}

LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
