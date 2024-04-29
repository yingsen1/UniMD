workdir=../
cd ${workdir}
echo "workdir"
echo $(pwd)
echo "python3:"
echo $(which python)

config=configs/charades/charades_i3dClip_convnext_bs4_70e_stop35e.yaml
ckpt=checkpoint/individual/charades/i3dclip_tad/charades_i3dClip_individual.pth.tar

CUDA_VISIBLE_DEVICES=0 python3 eval.py --config ${config} \
  --ckpt ${ckpt} \
  --data_type "tad" # tad, mr, all