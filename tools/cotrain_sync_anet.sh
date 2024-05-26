workdir=../
cd ${workdir}
echo "workdir"
echo $(pwd)
echo "python3:"
echo $(which python)

config=./configs/anet/anet_mae_convnext_queryavg_bs8_15e.yaml
output=synchronized

CUDA_VISIBLE_DEVICES=1 python3 train_sync.py \
--config ${config} \
-p 50 \
--output ${output}  \
--data_type "all"
#--skip-val-epoch 5