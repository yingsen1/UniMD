workdir=../
cd ${workdir}
echo "workdir"
echo $(pwd)
echo "python3:"
echo $(which python)

config=./configs/charades/charades_i3d_convnext_bs2_70e_stop20e.yaml
output=random

CUDA_VISIBLE_DEVICES=2 python3 train_sync.py \
--config ${config} \
-p 50 \
--output ${output}  \
--data_type "all"
#--skip-val-epoch 5