python test_swiftnet.py --cityscapes-dir /esat/tiger/tverelst/dataset/cityscapes \
--model-backbone resnet18 --model-checkpoint pretrained/swiftnet_rn18.pth --half \
--block-policy stati --block-size 128 \
--clip-length 20 --num-clips-warmup 500 --num-clips-eval -1