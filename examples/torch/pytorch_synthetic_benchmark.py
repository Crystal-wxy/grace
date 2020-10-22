from __future__ import print_function
import time
import torch.nn as nn

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models, datasets, transforms
import horovod.torch as hvd
import timeit
import numpy as np

import horovod.torch as hvd
from grace_dl.torch.communicator.allgather import Allgather
from grace_dl.torch.communicator.allreduce import Allreduce
from grace_dl.torch.communicator.broadcast import Broadcast
from grace_dl.torch.compressor.topk import TopKCompressor
from grace_dl.torch.memory.residual import ResidualMemory

from grace_dl.torch.compressor.none import NoneCompressor
from grace_dl.torch.memory.none import NoneMemory

from grace_dl.torch.compressor.powersgd  import PowerSGDCompressor
from grace_dl.torch.memory.powersgd import PowerSGDMemory

from grace_dl.torch.compressor.efsignsgd import EFSignSGDCompressor
from grace_dl.torch.memory.efsignsgd import EFSignSGDMemory

from grace_dl.torch.compressor.qsgd import QSGDCompressor
from grace_dl.torch.compressor.onebit import OneBitCompressor
from grace_dl.torch.compressor.natural import NaturalCompressor
from grace_dl.torch.compressor.fp16 import FP16Compressor
from grace_dl.torch.compressor.randomk import RandomKCompressor
from grace_dl.torch.compressor.signsgd import SignSGDCompressor
from grace_dl.torch.compressor.signum import SignumCompressor
from grace_dl.torch.compressor.terngrad import TernGradCompressor
from grace_dl.torch.compressor.threshold import ThresholdCompressor

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')                    

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

hvd.init()
#torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    # torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()

# By default, Adasum doesn't need scaling up learning rate.
lr_scaler = hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

# optimizer = optim.SGD(model.parameters(), lr=0.01 * lr_scaler)

# Horovod: (optional) compression algorithm.
# compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
# optimizer = hvd.DistributedOptimizer(optimizer,
#                                      named_parameters=model.named_parameters(),
#                                      compression=compression,
#                                      op=hvd.Adasum if args.use_adasum else hvd.Average)

# new add - Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                      momentum=args.momentum)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# GRACE: compression algorithm.
#grc = Allgather(TopKCompressor(0.01), ResidualMemory(), hvd.size())
#grc = Allgather(RandomKCompressor(0.01), ResidualMemory(), hvd.size())
#grc = Allgather(ThresholdCompressor(0.01), ResidualMemory(), hvd.size())
#grc = Allgather(FP16Compressor(), ResidualMemory(), hvd.size())
#grc = Allgather(TernGradCompressor(), ResidualMemory(), hvd.size())
grc = Allgather(SignumCompressor(0.9), ResidualMemory(), hvd.size())
#grc = Allgather(SignSGDCompressor(), ResidualMemory(), hvd.size())
#compressor = OneBitCompressor()
#memory = EFSignSGDMemory(lr=0.1)
#memory = ResidualMemory()
#grc = Broadcast(compressor, memory, hvd.size())
#grc = Broadcast(EFSignSGDCompressor(lr=0.1), EFSignSGDMemory(lr=0.1), hvd.size())
#compressor = NaturalCompressor()
#memory = ResidualMemory()
#grc = Allreduce(compressor, memory)

#grc = Allgather(NoneCompressor(), NoneMemory(), hvd.size())

#grc = Allgather(PowerSGDCompressor(), NoneMemory(), hvd.size())

#compressor = PowerSGDCompressor()
#memory = PowerSGDMemory(q_memory=compressor.q_memory, compress_rank=1)
#grc = Broadcast(compressor, memory, hvd.size())
#grc = Broadcast(PowerSGDCompressor(), memory, hvd.size())

#grc = Allgather(EFSignSGDCompressor(0.3), EFSignSGDMemory(lr), hvd.size())
#grc = Allgather(NaturalCompressor(), NoneMemory())
#grc = Allgather(QSGDCompressor(15), NoneMemory(), hvd.size())

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer, grc, named_parameters=model.named_parameters())

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()

def benchmark_step():
    torch.cuda.synchronize()
    forward_time = time.time()
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    backward_time = time.time()
#    print("computation time: ", round(backward_time-forward_time, 3))
 #   print("communication time: ", round(time.time()-backward_time, 3))

def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time1 = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time1
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
