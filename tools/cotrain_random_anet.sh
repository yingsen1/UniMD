workdir=../
cd ${workdir}
echo "workdir"
echo $(pwd)
echo "python3:"
echo $(which python)

config=./configs/anet/anet_mae_convnext_queryavg_bs8_15e.yaml
output=random

CUDA_VISIBLE_DEVICES=1 python3 train_random.py \
--config ${config} \
-p 50 \
--output ${output}  \
--data_type "all"
#--skip-val-epoch 5