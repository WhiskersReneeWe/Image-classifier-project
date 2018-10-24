



# I IGNORED OTHER LINES HERE. The following code is the training and validation part.
    with active_session():
    vgg = pre_model.vgg16(pretrained=True)

    for param in vgg.parameters():
    param.requires_grad = False

    # Step 2 - Define a new, untrained classifier

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
