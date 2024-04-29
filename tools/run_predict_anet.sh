workdir=../
cd ${workdir}
echo "workdir"
echo $(pwd)
echo "python3:"
echo $(which python)

config=configs/anet/anet_mae_convnext_queryavg_bs8_15e.yaml
ckpt=checkpoint/individual/anet/anet_tad_individual.pth.tar

CUDA_VISIBLE_DEVICES=0 python3 eval.py --config ${config} \
  --ckpt ${ckpt} \
  --data_type "tad" # tad, mr, all