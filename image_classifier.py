# Imports here
import numpy as np
import time
import copy
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as pre_model
import signal

from contextlib import contextmanager

import requests


DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define data transforms for the training, validation, and testing sets
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(45),
                                                transforms.RandomResizedCrop(224),
                                                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize((256, 256)),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    'test': transforms.Compose([transforms.Resize((256, 256)),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# TODO: Load the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
                  'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
                  'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train_loader': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 45,
                                          shuffle = True),
               'valid_loader': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 35 ,
                                         shuffle = True),
               'test_loader': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 30, shuffle = True) 
}

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# TODO: Build and train your network
# Step 1 - Load a pre-trained model
with active_session():
    vgg = pre_model.vgg16(pretrained=True) 

# Step 2 - Define a new, untrained classifier. Here, I kept the input_size unchanged as the original model, and only changed the output
# size because we have 102 classes in this project.
input_size = 25088
hidden_sizes = [4096, 4096]
output_size = 102
my_classifier = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         Dropout(p = 0.5),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         Dropout(p = 0.5),
                         nn.Linear(hidden_sizes[1], output_size),
                         nn.Softmax(dim=1))

vgg.classifier = my_classifier  
#Move this model to GPU
vgg = vgg.cuda()


# Step 3 - Train the classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(vgg.classifier.parameters(), lr=0.001)
epochs = 9
# sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)


print_every = 60
steps = 0

# change to cuda
vgg.to('cuda')

for e in range(epochs):
    running_loss = 0
    for ii, (image, label) in enumerate(dataloaders['train_loader']):
        steps += 1

        image = image.to('cuda')
        label = label.to('cuda')
        # image, label = image.to('cuda'), label.to('cuda')

        optimizer.zero_grad()

        # Forward and backward pass

        output = vgg(image)
        output = torch.exp(output).data
        _, preds = torch.max(output, 1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))

            running_loss = 0

# Testing the accuracy using test data
correct = 0
total = 0
with torch.no_grad():
    for data in dataloaders['test_loader']:
        images, labels = data
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = vgg(images)
        outputs = torch.exp(outputs) 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))  
