import os
from Train import parser, train, test


args = parser.parse_args()
args.pt = 0.6#0.8#0.9#1.0#
args.ps = 0.4#0.5#0.7#0.6#
args.num_joint = [17, 1, 7, 3]
args.num_rnn = [1, 1, 1]
args.model = 'SemGCODERNN'#'GCODERNN'#'ODERNN'#'DGCGRU'#'SemGCGRU'#'GCGRU'#'STGCN'#'HDGODE'#'HGCODE'#'HNODE'#'DGCODERNN'#
args.dataset = 'h36m_single'
args.batch_size = 60
#args.early_stop = 100
args.learning_rate = 1e-2
train(args)
args.pretrained = True
args.learning_rate = 1e-3
train(args)
args.learning_rate = 1e-4
train(args)
args.batch_size = 1
test(args)