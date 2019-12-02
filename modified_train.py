import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='CML Assignment')
parser.add_argument('--name', default='', type=str, help='name of the model')
parser.add_argument('--checkpoint', action='store_true', default=False, help='checkpoint name')
args = parser.parse_args()


# Data parameters
data_folder = './'  # folder with data files
keep_difficult = False  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters

checkpoint = args.checkpoint 
# "BEST_checkpoint_ssd300.pth.tar"  # path to model checkpoint, None if none
batch_size = 8  # batch size
start_epoch = 0  # start at this epoch
epochs = 1000  # number of epochs to run without early-stopping
epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
best_loss = 100.  # assume a high loss at first
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training or validation status every __ batches
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

val_dataset = PascalVOCDataset(data_folder,
                                   split='test',
                                   keep_difficult=keep_difficult)


val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)                                   



from detect import detect


def validate(val_loader, model, criterion):
    model.eval()  # eval mode disables dropout
    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(val_loader):
        	
            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            # Print status
            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
    
    # try:
    #     os.mkdir("verify/")
    # except OSError:
    #     None
    # img_path = '/scratch/pp1953/cml/ass/class_pics/IMG_0504.jpg'
    # original_image1 = Image.open(img_path, mode='r')
    # img_path = '/scratch/pp1953/cml/ass/class_pics/IMG_0505.jpg'
    # original_image2 = Image.open(img_path, mode='r')
    # original_image1 = original_image1.convert('RGB')
    # original_image2 = original_image2.convert('RGB')
    # det1_image= detect(original_image1, min_score=0.2, max_overlap=0.5, top_k=200, model=model).save("verify/" + args.name + "IMG_0504.jpg", "JPEG")
    # det2_image = detect(original_image2, min_score=0.2, max_overlap=0.5, top_k=200, model=model).save("verify/" + args.name + "IMG_0505.jpg", "JPEG")
    

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))
    return losses.avg



def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    start = time.time()
    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        # Update model
        optimizer.step()
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored




# model = checkpoint['model']
# state_dict = model.state_dict()
# torch.save({"state_dict": state_dict}, "pretrained_model_weights.pth")

model = SSD300(n_classes=n_classes)
biases = list()
not_biases = list()
for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)


# optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
#                             lr=lr, weight_decay=weight_decay)

# optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
#                             lr=lr, momentum=momentum, weight_decay=weight_decay)


if checkpoint:
    frozen = []
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if "pred_convs" not in param_name:
                frozen.append(param)
            else:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
    
    # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': frozen, 'lr': lr/10}, {'params': not_biases}],
    #                         lr=lr, momentum=momentum, weight_decay=weight_decay)

    optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': frozen, 'lr': lr/10}, {'params': not_biases}],
                            lr=lr, weight_decay=weight_decay)


	# checkpoint = torch.load(checkpoint,map_location=device)
	# start_epoch = checkpoint['epoch'] + 1
	# epochs_since_improvement = checkpoint['epochs_since_improvement']
	# best_loss = checkpoint['best_loss']
	# optimizer = checkpoint['optimizer']
    state_dict = {}
    checkpoint = torch.load("pretrained_model_weights.pth",map_location=device)
    for key in checkpoint['state_dict']:
            if "pred_convs" not in key :
                state_dict[key] = checkpoint['state_dict'][key]
    model.load_state_dict(state_dict,  strict=False)
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))


model.to(device)  
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

for epoch in range(start_epoch, epochs):
        # Paper describes decaying the learning rate at the 80000th, 100000th, 120000th 'iteration', i.e. model update or batch
        # The paper uses a batch size of 32, which means there were about 517 iterations in an epoch
        # Therefore, to find the epochs to decay at, you could do,
        # if epoch in {80000 // 517, 100000 // 517, 120000 // 517}:
        #     adjust_learning_rate(optimizer, 0.1)
        # In practice, I just decayed the learning rate when loss stopped improving for long periods,
        # and I would resume from the last best checkpoint with the new learning rate,
        # since there's no point in resuming at the most recent and significantly worse checkpoint.
        # So, when you're ready to decay the learning rate, just set checkpoint = 'BEST_checkpoint_ssd300.pth.tar' above
        # and have adjust_learning_rate(optimizer, 0.1) BEFORE this 'for' loop
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        # One epoch's validation
        val_loss = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion)
        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        # Save checkpoint
        save_checkpoint(epoch, model, best_loss, is_best, name=args.name)
