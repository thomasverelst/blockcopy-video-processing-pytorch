python test.py --cityscapes-dir /esat/tiger/tverelst/dataset/cityscapes \
--network swiftnet_resnet50 --model-checkpoint pretrained/swiftnet_rn50.pth --half \
--block-policy random --block-size 128 \
--clip-length 20 --num-clips-warmup 500 --num-clips-eval -1