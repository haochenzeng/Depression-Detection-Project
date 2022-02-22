import torch
from torch import nn
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
from Tools import *
from eval import *
from decoder import *

warnings.filterwarnings("ignore")

best_loss = 0.01

def validate(net, val_iter, loss_function, device):
    net.eval()
    val_loss =[]
    with torch.no_grad():
        for i, (X,y) in enumerate(val_iter):
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss_function(y_hat,y)
            val_loss.append(l.item())
    return np.mean(val_loss)


def train(net, train_iter, val_iter, test_iter, num_epochs, lr, device, visualize = False, use_pretrained_model = False):
    # weight initialize
    def init_net_weight(m):
        if type(m) ==nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if not use_pretrained_model:
        net.apply(init_net_weight)

    print('train on',device)
    net.to(device)

    use_gpu = torch.cuda.device_count() >= 1
    if use_gpu:
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=2.5e-4,momentum=0)
        optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
        loss = nn.CrossEntropyLoss().cuda()
    else:
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=2.5e-4,momentum=0)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        loss = nn.CrossEntropyLoss()

    timer, num_batches = Timer(),len(train_iter)
    train_accuracy, val_accuracy = [], []

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X,y) in enumerate(train_iter):
            # print(f'batch:{i}')
            timer.start()
            optimizer.zero_grad()
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0],accuracy(y_hat,y),X.shape[0])
            train_loss = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            train_accuracy.append(train_acc)
            timer.stop()
        val_loss = validate(net=net,val_iter=val_iter, loss_function=loss,device=device)
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save(net.state_dict(), './model.pt')

        val_acc = evaluate(net,val_iter)
        val_accuracy.append(val_acc)

        print(f'epoch{epoch+1},train_loss{train_loss:f},val_loss{val_loss},train_acc {train_acc:f},val_acc {val_acc}')

    torch.save(net.state_dict(),'checkpoints/CNN5_theta.pth')

    test_acc = evaluate(net,test_iter)
    print(f'train_loss = {train_loss:.3f},train_acc = {train_acc:.3f}')
    print(f'val_loss = {val_loss:.3f},val_acc = {val_acc:.3f}')
    print(f'test_acc = {test_acc:.3f}')
    print(f'{metric[2] * num_epochs /timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    print(f'total costs = {timer.sum():.1f}sec')

    # Visualization
    if visualize:
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.plot(range(num_epochs), val_accuracy)
        plt.title('test_accuracy')

        plt.figure(1)
        plt.subplot(1,2,2)
        plt.plot(range(num_epochs * num_batches), train_accuracy)
        plt.title('train_accuracy')
        plt.show()



