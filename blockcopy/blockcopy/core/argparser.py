def add_argparser_arguments(parser):
    parser.add_argument("--block-policy", type=str, default='rl_semseg', help='policy name')
    parser.add_argument("--block-num-classes", type=int, default=19, help='number of output classes')
    parser.add_argument("--block-optim-lr", type=float, default=0.0001, help='policy learning rate')
    parser.add_argument("--block-optim-wd", type=float, default=0.001, help='policy weight decay')
    parser.add_argument("--block-optim-momentum", type=float, default=0, help='policy optimizer momentum')
    parser.add_argument("--block-net", type=str, default='resnet8', help='backbone for policy net')
    parser.add_argument("--block-target", type=float, default=0.50, help='target execution percentage')
    parser.add_argument("--block-complexity-weight", type=float, default=5, help='weight gamma, setting importance of complexity reward')
    parser.add_argument("--block-size", type=int, default=128, help='size of blocks in px')
    parser.add_argument("--block-train-interval", type=int, default=4, help='optimize the policy every N frames')
    parser.add_argument("--block-cost-momentum", type=float, default=0.9, help='cost momentum')
    parser.add_argument("--block-policy-verbose", action="store_true", help="print debug info for policy training")
    return parser