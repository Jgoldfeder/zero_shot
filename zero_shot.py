import argparse
import cifar_classes
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import models.cifar as models
import torch, os,time
import torch.nn as nn
import torch.nn.parallel
from collections import defaultdict
from utils import  AverageMeter, accuracy
import numpy as np

# given a dataset, return all indices in the dataset where the label is the same as class_name
def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices  

# given a dataset and model, run the data through the model and return all the outputs as a python list
def get_labels(dataset,model):
    labels = []
    for x,y in dataset:
        label = model(x.unsqueeze(0))
        labels.append(label.squeeze(0))
    return labels

# given a dataloader and model, test its accuracy 
def test(testloader, model, use_cuda=True):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
     
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                model.module.labels.cuda()
            # convert targets to dense representation
            dense_targets = model.module.labels.get_dense(targets)
           

            # compute output
            outputs = model(inputs)

            # get category labels
            categories = model.module.labels.get_categorical(outputs)
           
            if use_cuda:
                categories = categories.cuda()
            # measure accuracy and record loss
            prec1, prec5 = accuracy(categories.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

          
                 
        return (top1.avg,top5.avg)  
  
  
  
  
  
  

parser = argparse.ArgumentParser(description='Continual Learning Visualization Tool')
parser.add_argument('--model', default="", type=str, metavar='PATH', help='path to a model trained on CIFAR-100')
parser.add_argument('--subset', type=str, default='', help='path to subset file')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help= 'model architechture'  )                      
parser.add_argument('--depth', type=int, default=32, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--dense-labels', default=True, action='store_true',help='Use dense labels.')
                                      
args = parser.parse_args()

# read the subset file, and convert the names of classes to their CIFAR-100 indices
subset_idx = []
with open(args.subset) as f:
    for line in f:
        if line.strip() == "":
            continue
        subset_idx.append(cifar_classes.coarse_label.index(line.strip()))

#people = ['baby', 'boy', 'girl', 'man', 'woman']
#for p in people:

# get every class index that we did not train on, ie the negation of the subset we trained on
not_subset_idx = []
for p in cifar_classes.coarse_label:
    idx = cifar_classes.coarse_label.index(p)
    if idx in subset_idx:
        continue
    print(p)
    not_subset_idx.append(idx)
    

# load cifar-100 with no augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataloader = datasets.CIFAR100
num_classes = 100

trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
indices = []
for idx in subset_idx:
    indices += get_indices(trainset,idx)
trainloader = data.DataLoader(data.Subset(trainset,indices), batch_size=100, shuffle=True, num_workers=0)
testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
indices = []
for idx in subset_idx:
    indices += get_indices(testset,idx)
testloader = data.DataLoader(data.Subset(testset,indices), batch_size=100, shuffle=False, num_workers=0)  
  
trainset_by_class = {}
for idx in not_subset_idx:
    indices = get_indices(trainset,idx)
    subset = data.Subset(trainset, indices)
    trainset_by_class[idx] = subset

testset_by_class = {}
for idx in not_subset_idx:
    indices = get_indices(testset,idx)
    subset = data.Subset(testset, indices)
    testset_by_class[idx] = subset
    
  
    
  

# create model
print("==> creating model '{}'".format(args.arch))
if args.arch.startswith('resnext'):
    model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
elif args.arch.startswith('densenet'):
    model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                dropRate=args.drop,
            )
elif args.arch.startswith('wrn'):
    model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
elif args.arch.endswith('resnet'):
    model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                block_name=args.block_name,
                categorical= not args.dense_labels
            )         
else:
    model = models.__dict__[args.arch](num_classes=num_classes)
model = torch.nn.DataParallel(model).cuda()
model.eval()

# Load model
print("loading model into memory")
assert os.path.isfile(args.model), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.model)
best_acc = checkpoint['best_acc']
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
if 'labels' in checkpoint and checkpoint['labels'] is not None:
    model.module.labels.set_labels(checkpoint['labels'])

labels = model.module.labels.dense_labels.clone()
results = []
for idx in not_subset_idx:  
    new_label_list = get_labels(testset_by_class[idx],model)
    prior_acc,_ = test(testloader,model)
    print("prior acc>",prior_acc)
    acc_list = []
    for label in new_label_list:
       new_labels = labels.clone()
       new_labels[idx] = label
       model.module.labels.set_labels(new_labels)

       d=data.DataLoader(trainset_by_class[idx], batch_size=100, shuffle=True, num_workers=0)
       acc,_ = test(d,model)
       acc_list.append(acc)
       
    results.append((np.average(acc_list),idx))
    print("new acc>",(np.average(acc_list)))

    
results.sort()
results.reverse()
for r in results:
    name = cifar_classes.coarse_label[r[1]]
    print(name.ljust(18, ' '), r[0])