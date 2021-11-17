python -O test_swiftnet.py --cityscapes-dir /esat/tiger/tverelst/dataset/cityscapes \
--model-backbone resnet50 --model-checkpoint pretrained/swiftnet_rn50.pth --batch-size 2 \
--block-policy static --block-size 128 \
--clip-length 20 --num-clips-warmup 50 --num-clips-eval 20 \
--half --fast --single-clip-loop