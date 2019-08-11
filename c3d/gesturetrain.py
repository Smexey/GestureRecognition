import torch.optim as optim
import torch.nn as nn
import torch
import C3D_model
import time
import copy

from tensorboardX import SummaryWriter



def train(net, dataloaders, dataset_sizes):
    since = time.time()

    writer = SummaryWriter(comment = "test1",log_dir = "c3d\\tblogs",filename_suffix = str(time.time()))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=10e-6,
                           eps=10e-12, weight_decay=0.1)

    best_acc = 0.0

    # net = C3D_model.C3D(net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.freeze()
    num_of_epochs = 30

    for epoch in range(num_of_epochs):
        omaseni = [0, 0, 0, 0, 0, 0]

        # if(epoch == 5):
            # net.unfreeze_all()


        print('Epoch {}/{}'.format(epoch, num_of_epochs))
        print('-' * 10)
        for phase in ['train', 'valid']:

            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for iteration,_ in enumerate(dataloaders[phase]):

                inputs, labels = _
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    chanceofpred, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    print("certainty: {} loss: {} hitpredict: {} label: {}".format(
                        *chanceofpred, loss.item(), *(preds == labels.data), *labels.data))

                    
                    

                    # backward + optimize only if in training phase
                    if(not (preds == labels.data)):
                        omaseni[preds] += 1
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

                writer.add_scalar('currloss'+ phase, running_loss/(iteration+1), iteration)
                writer.add_scalar('curracc'+ phase, running_corrects/(iteration+1), iteration)



            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            writer.add_scalar('epochloss'+ phase, epoch_loss, epoch)
            writer.add_scalar('epochacc'+ phase, epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
        print("OMASENI LABEL VEKTOR: ", omaseni)
        print("\n")
        import os
        torch.save(net.state_dict(), os.path.join(
            "checkpoints\\", "adam10e6eps12regul01"+str(epoch) + "_" + "{:.4f}".format(epoch_acc)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print("OMASENI LABEL VEKTOR: ", omaseni)
