workdir=../
cd ${workdir}
echo "workdir"
echo $(pwd)
echo "python3:"
echo $(which python)

config=configs/ego4d/ego4d_vmaeVerb_convnext_35e_stop15e.yaml
ckpt=checkpoint/individual/ego4d/ego4d_mq_individual.pth.tar  # path to checkpoint

CUDA_VISIBLE_DEVICES=0 python3 eval.py --config ${config} \
  --ckpt ${ckpt} \
  --data_type "tad" # tad, mr, all
