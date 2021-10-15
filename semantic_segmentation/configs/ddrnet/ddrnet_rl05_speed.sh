python -O test.py --cityscapes-dir /esat/tiger/tverelst/dataset/cityscapes \
--network ddrnet_39 --model-checkpoint pretrained/ddrnet_23.pth \
--block-policy rl_semseg --block-target 0.5 --block-size 128 --block-policy-verbose \
--clip-length 20 --num-clips-warmup 20 --num-clips-eval 5 \
--half --fast --single-clip-loop