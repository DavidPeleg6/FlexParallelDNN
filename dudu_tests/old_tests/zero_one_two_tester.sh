#!/bin/bash
echo "g: the amount of gpu's"
echo "e: the amount of epochs"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -g|--gpus)
      GPUS="$2"
      shift # past argument
      shift # past value
      ;;
	-e|--epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done
echo "total gpus: " $GPUS
if [[ $EPOCHS -eq 0 ]]; then
	echo "using default epochs"
	EPOCHS=2
fi
echo "total epochs:" $EPOCHS
cd $FF_HOME
HIGH_FOLDER=$FF_HOME/zero1-2-test-${GPUS}_gpus
if ! [[ -d ${HIGH_FOLDER} ]]; then
	`mkdir ${HIGH_FOLDER}`
fi
nvidia-smi &> $HIGH_FOLDER/gpu_info.txt

for ((GPU=1 ; GPU<=$GPUS; GPU+=1))
do

  echo "python zero_one_two.py --world_size ${GPU} --epochs ${EPOCHS}"
  python zero_one_two.py --world_size $GPU --epochs $EPOCHS 2>&1 | tee output.txt
  cat output.txt | grep THROUGHPUT >> ${HIGH_FOLDER}/all_runtimes_${GPU}gpu.txt
  echo "done training"

done