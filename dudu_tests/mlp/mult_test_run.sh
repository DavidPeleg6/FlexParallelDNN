#!/bin/bash

NNODES=1
NPROC_PER_NODE=4
BATCH=64
ITERATIONS=20
HOST_NODE_ADDR=pytorch-set-0.pytorch.pelegdav.svc.cluster.local:29600
#HOST_NODE_ADDR=pytorch-set-1.pytorch.pelegdav.svc.cluster.local:29600

for ((ITER=1 ; ITER<=$ITERATIONS; ITER+=1))
do
  echo "torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1236 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR \
    mnist_mlp.py --batch_size ${BATCH} --strategy strategy.json"

  torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1236 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR \
    mnist_mlp.py --batch_size ${BATCH} --strategy strategy.json | tee output.txt
  cat output.txt | grep "AVG. THROUGHPUT" | tee -a strategy_out.txt
done

for ((ITER=1 ; ITER<=$ITERATIONS; ITER+=1))
do
  echo "torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1236 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR \
    mnist_mlp.py --batch_size ${BATCH}"

  torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1236 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR \
    mnist_mlp.py --batch_size ${BATCH} | tee output.txt
  cat output.txt | grep "AVG. THROUGHPUT" | tee -a no_strategy_out.txt
done

LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
