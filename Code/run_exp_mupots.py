import os
from Train import parser, train, test, plot


args = parser.parse_args()
args.model = 'HDGODE'
args.dataset = 'mupots_multi'
args.max_epoch = 100
args.early_stop = 20

args.learning_rate = 1e-2
train(args)
args.pretrained = True
args.learning_rate = 1e-3
train(args)
args.learning_rate = 1e-4
train(args)
args.batch_size = 1
test(args)
