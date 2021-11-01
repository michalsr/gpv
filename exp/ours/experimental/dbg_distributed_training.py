import argparse

import torch
import torchvision
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import distributed as dist
import torch.nn.functional as F

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--local_rank', default=None, type=int)

  args = parser.parse_args()

  if args.local_rank is not None:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

  data = torchvision.datasets.MNIST("dbg-distributed-data", train=True, transform=ToTensor(), download=True)
  data_loader = DataLoader(
    data, batch_size=32, shuffle=True, num_workers=4)

  model = nn.Sequential(
    nn.Conv2d(
      in_channels=1,
      out_channels=16,
      kernel_size=5,
      stride=1,
      padding=2,
    ),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(784, 10)
  )
  if args.local_rank is not None:
    device = args.local_rank
  else:
    device = "cuda"

  model.to(device)

  if args.local_rank is not None:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

  opt = SGD(model.parameters(), 0.01)

  print("Starting train")
  for x, y in data_loader:
    out = model(x.to(device))
    loss = F.cross_entropy(out, y.to(device))
    loss.backward()
    opt.step()
    opt.zero_grad()

  print("Done")


if __name__ == '__main__':
  main()