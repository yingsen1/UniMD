workdir=../
cd ${workdir}
echo "workdir"
echo $(pwd)
echo "python3:"
echo $(which python)

config=./configs/ego4d/ego4d_vmaeVerb_convnext_35e_stop15e.yaml
output=tad

CUDA_VISIBLE_DEVICES=3 python3 train_random.py \
--config ${config} \
-p 50 \
--output ${output}  \
--data_type "tad"
#--skip-val-epoch 5