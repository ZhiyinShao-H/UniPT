# Train New Models

## Pretraining
```bash
export MASTER_ADDR=$DIST_0_IP  "0.0.0.0"
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=<BS_FITS_YOUR_GPU>

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=64

python3 run.py with data_root="/root/paddlejob/workspace/env_run/output/shaozhiyin/person_pretrain/arrow_data" num_gpus=8 num_nodes=1 task_mlm_itm_PP whole_word_masking=True step200k per_gpu_batchsize=4 

python3 run.py with data_root="/root/paddlejob/workspace/env_run/output/shaozhiyin/person_pretrain/arrow_data_v2" num_gpus=[0,1,2,3,4,5,6,7] num_nodes=1 task_mlm_itm_PP whole_word_masking=True step200k per_gpu_batchsize=64 load_path="/root/paddlejob/workspace/env_run/output/shaozhiyin/data/vilt_200k_mlm_itm.ckpt"

python3 run.py with data_root="/root/paddlejob/workspace/env_run/output/shaozhiyin/person_pretrain/arrow_data_v2" num_gpus=8 num_nodes=1 task_mlm_itm_PP whole_word_masking=True step200k per_gpu_batchsize=64 

python3 run.py with data_root="/root/paddlejob/workspace/env_run/output/shaozhiyin/data/home/zhiyin/coco_1" num_gpus=8 num_nodes=1 task_mlm_itm_coco whole_word_masking=True step200k per_gpu_batchsize=64 load_path="/root/paddlejob/workspace/env_run/output/shaozhiyin/data/vilt_200k_mlm_itm.ckpt"

python3 run.py with data_root="/root/paddlejob/workspace/env_run/output/shaozhiyin/data/coco_2" num_gpus=8 num_nodes=1 task_mlm_itm_coco whole_word_masking=True step200k per_gpu_batchsize=64 load_path="/root/paddlejob/workspace/env_run/output/shaozhiyin/data/vilt_200k_mlm_itm.ckpt"

python3 run.py with data_root="/root/paddlejob/workspace/env_run/output/shaozhiyin/data/coco_2" num_gpus=2 num_nodes=1 task_mlm_itm_coco whole_word_masking=True step200k per_gpu_batchsize=16

python3 run.py with data_root="/root/paddlejob/workspace/env_run/output/shaozhiyin/data/LUPerson_arrow" num_gpus=8 num_nodes=1 task_mlm_cl_LPP whole_word_masking=True step200k per_gpu_batchsize=16

nohup python3 run.py with data_root="/root/paddlejob/workspace/env_run/output/shaozhiyin/person_pretrain/arrow_data_v2" num_gpus=8 num_nodes=1 task_mlm_itm_PP whole_word_masking=True step200k per_gpu_batchsize=64 > nohup/task_mlm_itm_PP_from_.out 2>&1 &
```

## Finetune on VQAv2
```bash
export MASTER_ADDR=$DIST_0_IP        "0.0.0.0"
export MASTER_PORT=$DIST_0_PORT        "8001"
export NODE_RANK=$DIST_RANK               0
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_trainval_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path="<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa_randaug per_gpu_batchsize=64 load_path="weights/vilt_200k_mlm_itm.ckpt"
```

## Finetune on NLVR2
```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_nlvr2_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path="<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_nlvr2_randaug per_gpu_batchsize=32 load_path="weights/vilt_200k_mlm_itm.ckpt"
```

## Finetune on COCO IR/TR
```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_coco_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path="<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_coco_randaug per_gpu_batchsize=4 load_path="weights/vilt_200k_mlm_itm.ckpt"
```

## Finetune on F30K IR/TR
```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_f30k_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path="<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_irtr_f30k_randaug per_gpu_batchsize=4 load_path="weights/vilt_200k_mlm_itm.ckpt"
```
