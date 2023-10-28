"""
Collect the models in one place, for easy use.
"""

from copy import deepcopy
import time
import logging
import os.path
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

#from SimulationUtils import genRandomCenters, CenterLoader

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True

def getDtype(bits):
    if bits==16:
        return torch.float16
    elif bits==32:
        return torch.float32
    else:
        raise ValueError("ERROR! Currently only support 16 and 32 bits")


def mergeDict(target, d1):
    for k,v in d1.items():
        target[k]=v

class Mao_Recon_CNN_Blocked(nn.Module):
    def __init__(self, shrink=1, batchSide=16, subGridSize=39):
        super(Mao_Recon_CNN_Blocked, self).__init__()

        self.batchSide=batchSide
        self.subGridSize=subGridSize
        self.padWidth=self.subGridSize//2

        '''From pytorch documentation:
        
        Class torch.nn.Conv3d(in_channels, out_channels, 
        kernel_size, stride=1, padding=0, dilation=1, groups=1, 
        bias=True, padding_mode='zeros', device=None, dtype=None)'''

        # Define the number of channels
        nchan0 = 1  # Initial channel
        nchan1 = 32//shrink
        nchan2 = 2*nchan1
        nchan3 = 2*nchan2

        self.conv1 = nn.Conv3d(nchan0, nchan1, 3, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv3d(nchan1, nchan1, 3, stride=1, padding=0, dilation=2)
        self.conv3 = nn.Conv3d(nchan1, nchan2, 3, stride=1, padding=0, dilation=2)
        self.conv4 = nn.Conv3d(nchan2, nchan2, 3, stride=1, padding=0, dilation=4)
        self.conv5 = nn.Conv3d(nchan2, nchan3, 3, stride=1, padding=0, dilation=4)
        self.conv6 = nn.Conv3d(nchan3, nchan3, 2, stride=1, padding=0, dilation=8)
        self.conv7 = nn.Conv3d(nchan3, nchan3, 1, stride=1, padding=0, dilation=8)
        self.fc1 = nn.Linear(nchan3, 1)


    def encode(self, x):
        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)

        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)

        # Layer 3
        x = self.conv3(x)
        x = F.relu(x)

        # Layer 4
        x = self.conv4(x)
        x = F.relu(x)

        # Layer 5
        x = self.conv5(x)
        x = F.relu(x)

        # Layer 6
        x = self.conv6(x)
        x = F.relu(x)

        # Layer 7
        x = self.conv7(x)
        x = F.relu(x)
        x = x[:,:,0:self.batchSide,0:self.batchSide,0:self.batchSide].flatten(2).squeeze()
        x = torch.transpose(x,0,1)

        # Mean and flatten, and then fully connected layer
        x = self.fc1(x)
        x = torch.flatten(x)
        return x

    def forward(self, x):
        return self.encode(x)


def trainOne(model, device, optimizer, inputSim, targetSim, deltaNg=128):

    batchSide=inputSim.batchSide
    Ng=inputSim.ngrid
    blockSide=batchSide+2*inputSim.padWidth

    # Define the pytorch tensors that we'll need.
    inputGrids = torch.empty((1,1,blockSide,blockSide,blockSide), dtype=torch.float32)
    targetVals = torch.empty((batchSide**3,), dtype=torch.float32)
    inputGrids_gpu=torch.empty_like(inputGrids, device=device)
    targetVals_gpu=torch.empty_like(targetVals, device=device)

    # Initialize
    model.train()
    loss_fn = nn.MSELoss(reduction='sum')

    # Loop over all the centers
    loss_total=0.0
    numBlocks=0
    di,dj,dk=np.random.randint(0,deltaNg,size=(3,))
    isim=np.random.randint(0,inputSim.nsims)
    for i0 in range(0,Ng,deltaNg):
        for j0 in range(0,Ng,deltaNg):
            for k0 in range(0,Ng,deltaNg):
                cen=(isim,i0+di,j0+dj,k0+dk)
                inputSim.getPaddedBlock(inputGrids, *cen)
                inputGrids_gpu.copy_(inputGrids, non_blocking=True)
                targetSim.getBlock(targetVals, *cen)
                targetVals_gpu.copy_(targetVals, non_blocking=True)

                optimizer.zero_grad()
                output = model(inputGrids_gpu)

                loss = loss_fn(output, targetVals_gpu)
                loss.backward()

                optimizer.step()

                loss_total += loss.item()
                numBlocks +=1 
                # End of loop

    loss_total /= (numBlocks*batchSide**3)
    return loss_total

@torch.no_grad()
def testOne(model, device, inputSim, targetSim, deltaNg=128):

    batchSide=inputSim.batchSide
    Ng=inputSim.ngrid
    blockSide=batchSide+2*inputSim.padWidth

    # Define the pytorch tensors that we'll need.
    inputGrids = torch.empty((1,1,blockSide,blockSide,blockSide), dtype=torch.float32)
    targetVals = torch.empty((batchSide**3,), dtype=torch.float32)
    inputGrids_gpu=torch.empty_like(inputGrids, device=device)
    targetVals_gpu=torch.empty_like(targetVals, device=device)

    # Initialize
    model.eval()
    loss_fn = nn.MSELoss(reduction='sum')

    # Loop over all the centers
    loss_total=0.0
    numBlocks=0
    di,dj,dk=np.random.randint(0,deltaNg,size=(3,))
    isim=np.random.randint(0,inputSim.nsims)
    for i0 in range(0,Ng,deltaNg):
        for j0 in range(0,Ng,deltaNg):
            for k0 in range(0,Ng,deltaNg):
                cen=(isim,i0+di,j0+dj,k0+dk)
                inputSim.getPaddedBlock(inputGrids, *cen)
                inputGrids_gpu.copy_(inputGrids, non_blocking=True)
                targetSim.getBlock(targetVals, *cen)
                targetVals_gpu.copy_(targetVals, non_blocking=True)

                output = model(inputGrids_gpu)

                loss = loss_fn(output, targetVals_gpu)

                loss_total += loss.item()
                numBlocks +=1 
                # End of loop

    loss_total /= (numBlocks*batchSide**3)
    return loss_total


# traindict and testdict are dictionaries 
# dict["input"] = inputSimulation
# dict["target"] = targetSimulation
# dict["centers"] = centers 
def runTraining(model, device, optimizer, traindict, testdict, \
    deltaNg=128, numEpochs=1, path=None, 
    config=None):

    # Check output directories etc
    if path is None:
        raise ValueError("ERROR! Must specify an output path")

    if config is None:
        raise ValueError("ERROR! config must be set")

    # Get some basic parameters
    Ngrid=traindict["input"].ngrid
    nsims=traindict["input"].nsims

    schedConfig={'factor':0.7, 'patience': 5, 'cooldown':2, 'threshold': 1.0e-3}
    mergeDict(schedConfig, config.get('scheduler',{}))
    scheduler=ReduceLROnPlateau(optimizer, mode='min', verbose=True, **schedConfig)
    minLearningRate=config.get("minLearningRate",5.0e-6)

    minValidationLoss=1.0e100
    bestModel=deepcopy(model.state_dict())
    
    # Define a logger for the training progress
    trainlog=logging.getLogger("training")

    earlyStop=False
    bestEpoch=0

    epochs=[]
    train_loss_arr=[]
    test_loss_arr=[]

    for iEpoch in range(numEpochs):
        tic = time.perf_counter()
        
        # Training
        trainloss=trainOne(model, device, optimizer, traindict["input"], \
            traindict["target"], deltaNg=deltaNg)
        train_loss_arr.append(trainloss)
 
        # Testing
        testloss=testOne(model, device, testdict["input"], \
            testdict["target"], deltaNg=deltaNg)
        test_loss_arr.append(testloss)

        epochs.append(iEpoch)
        if testloss < minValidationLoss:
            minValidationLoss=testloss
            bestEpoch=iEpoch
            bestModel=deepcopy(model.state_dict())


        #scheduler.step(testloss)
        scheduler.step(testloss)

        # Check for an early stop
        currentLR=float([group['lr'] for group in optimizer.param_groups][0])
        if currentLR < minLearningRate:
            earlyStop=True
            logging.debug("Early stop as learning rate drops belows threshold")
        if ((iEpoch-bestEpoch) > 30) & (iEpoch > 30): # changed from 20 to 30
            earlyStop=True
            logging.debug("Early stop as no improvement for a while")

        # Save intermediate steps
        if (iEpoch % 10)==0:
            checkptPath=os.path.join(path,f"checkpt-{iEpoch:05d}.pth")
            torch.save({
                'epoch': iEpoch,
                'model_state_dict': model.state_dict(),
                'best_state_dist':bestModel,
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': trainloss,
                'valloss':testloss,
                'minvalloss':minValidationLoss,
                'learningRate': currentLR
            }, checkptPath)

            savearrs=np.array([epochs,train_loss_arr,test_loss_arr]).T
            np.savetxt(path+"/losses_uptoepoch%i.txt"%(iEpoch), savearrs, fmt=['%.4f','%.4f','%.4f'],\
                header='%12s\t%12s\t%12s'%('epoch','train loss total','test loss total'),delimiter='\t')

        # Status
        toc = time.perf_counter()
        trainlog.info(f"{iEpoch:5d}: {trainloss:.4f} {testloss:.4f} {minValidationLoss:.4f}"+
            f" {currentLR*1e3:.3f} {(toc-tic)/60.0:.4f}") 

        # Check if stopping
        if earlyStop:
            break
    

    # Cleanup here
    checkptPath=os.path.join(path,"model.pth")
    checkpt={
        'epoch': iEpoch,
        'model_state_dict': model.state_dict(),
        'best_state_dict':bestModel,
        'optimizer_state_dict': optimizer.state_dict(),
        'trainloss': trainloss,
        'valloss':testloss,
        'minvalloss':minValidationLoss,
        'learningRate': currentLR
    }
    torch.save(checkpt, checkptPath)

    # End
    return checkpt
