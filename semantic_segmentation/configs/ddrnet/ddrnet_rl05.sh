python test.py --cityscapes-dir /esat/tiger/tverelst/dataset/cityscapes \
--network ddrnet_39 --model-checkpoint pretrained/ddrnet_23.pth --half \
--block-policy rl_semseg --block-target 0.5 --block-policy-verbose \
--clip-length 20 --num-clips-warmup 500 --num-clips-eval -1