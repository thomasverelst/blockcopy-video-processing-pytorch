python -O test.py --cityscapes-dir /esat/tiger/tverelst/dataset/cityscapes \
--network ddrnet_39 --model-checkpoint pretrained/ddrnet_23.pth \
--block-policy static \
--clip-length 20 --num-clips-warmup 10 --num-clips-eval 5 \
--fast --single-clip-loop --timings 10