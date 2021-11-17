python test_swiftnet.py --cityscapes-dir /esat/tiger/tverelst/dataset/cityscapes \
--model-backbone resnet50 --model-checkpoint pretrained/swiftnet_rn50.pth --half \
--block-policy stati --block-size 128 \
--clip-length 20 --num-clips-warmup 500 --num-clips-eval -1