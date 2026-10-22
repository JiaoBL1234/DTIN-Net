#!/bin/bash

# dataset names = [market1501, dukemtmc, veri776, vehicleid]

source activate fastreid_

#SOLVER.MODE = "General"



#CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/Partial/bagtricks_R50.yml OUTPUT_DIR ./logs/CVT/ TEST.EVAL_PERIOD 1 SOLVER.CHECKPOINT_PERIOD 30 SOLVER.IMS_PER_BATCH "(64)" SOLVER.MAX_EPOCH 60 SOLVER.STEPS [20,40] INPUT.REA.ENABLED False

CUDA_VISIBLE_DEVICES=4,5 python3 tools/train_net.py --config-file ./configs/Generalization/bagtricks_R50.yml --num-gpus 2 \
OUTPUT_DIR ./logs/new_Batch_V1_bs128/ TEST.EVAL_PERIOD 1 SOLVER.CHECKPOINT_PERIOD 30 SOLVER.IMS_PER_BATCH "(128)" \
INPUT.REA.ENABLED True MODEL.BACKBONE.WITH_IN True DATALOADER.SPECIFIC_WAY True DATALOADER.SAME_RUNNING True
# CUDA_VISIBLE_DEVICES=5,7 python3 tools/train_net.py --num-gpus 2 --config-file ./configs/Generalization/bagtricks_R50.yml --eval-only MODEL.WEIGHTS ./logs/new_Batch_V1_bs64/model_best59.pth  MODEL.BACKBONE.WITH_IN True
#CUDA_VISIBLE_DEVICES=6,7 python3 tools/train_net.py --num-gpus 2 --config-file ./configs/Generalization/bagtricks_R50.yml --vis-only MODEL.WEIGHTS ./logs/Generalization/model_best.pth TEST.TSNE_SHOW True SOLVER.VISUALIZATION MODEL.BACKBONE.WITH_IN False





