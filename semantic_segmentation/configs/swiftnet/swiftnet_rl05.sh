python test_swiftnet.py --cityscapes-dir /esat/tiger/tverelst/dataset/cityscapes \
--model-backbone resnet50 --model-checkpoint pretrained/swiftnet_rn50.pth --half \
--block-policy rl_semseg --block-target 0.5 --block-size 128 --block-train-interval 3 --block-policy-verbose \
--clip-length 20 --num-clips-warmup 500 --num-clips-eval -1