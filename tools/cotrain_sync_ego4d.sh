workdir=../
cd ${workdir}
echo "workdir"
echo $(pwd)
echo "python3:"
echo $(which python)

config=./configs/ego4d/ego4d_vmaeVerb_convnext_35e_stop15e.yaml
output=random

CUDA_VISIBLE_DEVICES=3 python3 train_sync.py \
--config ${config} \
-p 50 \
--output ${output}  \
--data_type "all"
#--skip-val-epoch 5