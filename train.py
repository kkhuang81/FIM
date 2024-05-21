import numpy as np
from model import MyModel
from utils import InitialWeight, LoadGraph, BCELoss, MAE
import time
import argparse
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import random
import math
import uuid
import glob, os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="HR",help='Dataset to use.')
parser.add_argument('--seed', type=int, default=25190, help='Random seed.')
parser.add_argument('--steps', type=int, default=2500, help='Number of steps to train.')
parser.add_argument('--patience', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dev', type=int, default=1, help='device id')
parser.add_argument('--batch', type=int, default=50, help='batch size')
parser.add_argument('--tau', type=float, default=1, help='infinitesimal value')
parser.add_argument('--train_per', type=float, default=0.8, help='percentage of training size')
parser.add_argument('--val_per', type=float, default=0.1, help='percentage of validation size')
parser.add_argument('--l1_lambda', type=float, default=0.001, help='l1 regularization coefficient')
parser.add_argument('--norm', type=float, default=2.0, help='gradient clip norm')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("--------------------------")
print(args)


def train(batch_x):
    model.train()
    time_epoch = 0 
    loss_train = 0
    t1 = time.time()        
    optimizer.zero_grad()
    output = batch_x[:,0,:]
    for slot in range(1,SlotNum):                        
        output = model(output, args.tau)                   
        loss_train += BCELoss(output, batch_x[:,slot,:])      
    loss_train += args.l1_lambda*torch.norm(model.weight.data, 1)
    loss_train.backward()     
    nn.utils.clip_grad_norm_(model.parameters(), args.norm) 
    optimizer.step()    
    act_fn(model.weight.data)    
    time_epoch += (time.time()-t1)
    return loss_train.item(), time_epoch

def validate():
    model.eval()
    micro_val = 0
    mae=0
    with torch.no_grad(): 
        output=ValiData[:,0,:]
        for slot in range(1,SlotNum):
            output = model(output, args.tau)
            micro_val += BCELoss(output, ValiData[:,slot,:])
            mae+=MAE(output, ValiData[:,slot,:])
        return micro_val, mae

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    micro_val = 0
    mae=0
    with torch.no_grad():
        output=TestData[:,0,:]
        for slot in range(1,SlotNum):
            output = model(output, args.tau)
            micro_val += BCELoss(output, TestData[:,slot,:]) 
            mae+=MAE(output, TestData[:,slot,:]) 
        return micro_val, mae



##Data Property
TimeHorizon = 20
SlotNum = int(TimeHorizon / args.tau + 1)


#Load Graph
TrainData, ValiData, TestData=LoadGraph(args.dataset, args.tau, args.train_per, args.val_per)
TrainSize=TrainData.shape[0]
nNodes=TrainData.shape[2]

#### initial to torch
TrainData = torch.FloatTensor(TrainData)
ValiData = torch.FloatTensor(ValiData)
TestData = torch.FloatTensor(TestData)


# Model and Optimizer
model=MyModel(nNodes=nNodes).cuda(args.dev)
optimizer = optim.Adam(model.parameters(), lr=args.lr) #weight_decay=args.weight_decay
act_fn = nn.ReLU(inplace=True)

#immigrate labels into GPU
TrainData = TrainData.cuda(args.dev)
ValiData = ValiData.cuda(args.dev)
TestData = TestData.cuda(args.dev)

torch_dataset = Data.TensorDataset(TrainData)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=args.batch,shuffle=False,num_workers=0)  # cannot shuffle

train_time = 0
train_time1 = 0
bad_counter = 0
best = 10000
best_epoch = 0

#for filename in glob.glob('*.pt'):
    #os.remove(filename)
checkpt_file = uuid.uuid4().hex+'.pt'

for epoch in range(args.steps):
    start=(epoch*args.batch)%TrainSize
    end=min(start+args.batch, TrainSize)
    loss_tra,train_ep = train(TrainData[start:end])    
    f1_val, mae_val = validate()
    train_time += train_ep 
    if epoch == 0:
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            '| val',
            'acc:{:.3f}'.format(f1_val),
            'mae:{:.3f}'.format(mae_val),
            '| cost:{:.3f}'.format(train_time))   
    if(epoch+1)%100 == 0:
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            '| val',
            'acc:{:.3f}'.format(f1_val),
            'mae:{:.3f}'.format(mae_val),
            '| cost:{:.3f}'.format(train_time))
    if f1_val < best:
        best = f1_val
        best_epoch = epoch
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

f1_test, mae_test = test()
print("Train cost: {:.4f}s".format(train_time), 'and {:.4f}'.format(train_time1))
print('Load {}th epoch'.format(best_epoch))
print("Test f1:{:.3f}".format(f1_test), "Test mae:{:.3f}".format(mae_test))
print("--------------------------")
