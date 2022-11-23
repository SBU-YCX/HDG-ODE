import os
from Train import parser, train, test


args = parser.parse_args()
args.num_rnn = [1, 1, 1]
args.model = 'SemGCGRU'#'GCGRU'#'STGCN'#'HDGODE'#'HGCODE'#'HNODE'#'DGCODERNN'#'SemGCODERNN'#'GCODERNN'#'ODERNN'#'DGCGRU'#
args.dataset = 'mupots_multi'#'mupots_single'#
args.learning_rate = 1e-2
train(args)
args.pretrained = True
args.learning_rate = 1e-3
train(args)
args.learning_rate = 1e-4
train(args)
args.batch_size = 1
test(args)