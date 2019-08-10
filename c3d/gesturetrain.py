import torch.optim as optim
import torch.nn as nn
import torch
import C3D_model
import time
import copy


def train(net, dataloaders, dataset_sizes):
    since = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)
    best_acc = 0.0

    # net = C3D_model.C3D(net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.freeze()
    
    for epoch in range(3):
        print('Epoch {}/{}'.format(epoch, 3 - 1))
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            cnt = 0

            for inputs, labels in dataloaders[phase]:
                # print(cnt)
                cnt += 1
                inputs = inputs.to(device)
                #('Doing other things',)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())

        print("\n")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
