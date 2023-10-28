"""
Time how long it takes to run an iteration of the model
"""

import time
import logging
import os
import os.path
import yaml
import sys

import numpy as np
import torch
import SimulationUtils_1024cubex8randoms as SU
import ModelBlock_32channel_savelosses as ModelBlock

if len(sys.argv) < 2:
    raise RuntimeError("Not enough parameters")
jobname=sys.argv[1]
with open(f"input/{jobname}.yml","r") as ff:
    config=yaml.safe_load(ff)

Ngrid=config['Ngrid']
batchSide=config['batchSide']
deltaNg=config.get('deltaNg',128)
subGridSize=config['subGridSize']
learningRate=config['learningRate']
numEpochs=config['numEpochs']
shrinkFactor=config['shrink']

trainSims=config['trainSims']
ntrain=len(trainSims)
testSims=config['testSims']
ntest=len(testSims)
ftype=config.get("iniSmoothing","sm3")

# Set up logging
outputPath=f"output/{jobname}"
if not os.path.isdir(outputPath):
    raise RuntimeError(f"Output path {outputPath} does not exist")
logfn=os.path.join(outputPath, "training.log")

## https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=logfn,
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on device={device}....")

seed=config.get("seed",3141)
np.random.seed(seed)
logging.info(f"Using a random seed {seed}....")

logging.info(f"Training with {ntrain} sims, validating with {ntest} sims")

# Initialize training and testing 
train={}
test={}

# Load the training data
tic = time.perf_counter()
train['input'] = SU.SimulationDataset(nsims=ntrain, ngrid=Ngrid, subgridSize=subGridSize, batchSide=batchSide)
train['target'] = SU.SimulationDataset(nsims=ntrain, ngrid=Ngrid, subgridSize=subGridSize, batchSide=batchSide)
logging.info(f"Loading initial conditions from smoothing directory {ftype}....")
for ii,isim in enumerate(trainSims):
    sim1=SU.loadRecon(isim) # Simulation 0 is the train
    train['input'].load(sim1,ii, normalize=True)
    sim1=SU.loadSmoothedIni(isim, ftype=ftype)
    train['target'].load(sim1,ii, normalize=True)
    logging.info(f"Loading train sim {isim} into slot {ii}")
toc = time.perf_counter()
logging.info(f"Time to load training data : {toc-tic:.4f}")

# Load the testing data
tic = time.perf_counter()
test['input'] = SU.SimulationDataset(nsims=ntest, ngrid=Ngrid, subgridSize=subGridSize, batchSide=batchSide)
test['target'] = SU.SimulationDataset(nsims=ntest, ngrid=Ngrid, subgridSize=subGridSize, batchSide=batchSide)
for ii,isim in enumerate(testSims):
    sim1=SU.loadRecon(isim) # Simulation 0 is the train
    test['input'].load(sim1,ii, normalize=True)
    sim1=SU.loadSmoothedIni(isim, ftype=ftype)
    test['target'].load(sim1,ii, normalize=True)
    logging.info(f"Loading test sim {isim} into slot {ii}")
toc = time.perf_counter()
logging.info(f"Time to load testing data : {toc-tic:.4f}")


# Define the model
model = ModelBlock.Mao_Recon_CNN_Blocked(shrink=shrinkFactor, batchSide=batchSide).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
numparam=get_n_params(model)
logging.info(f"number of parameters in the model = {numparam}")

# Load checkpoint if required
#restartPath=config.get("restart", None)
#if restartPath is not None:
#    restart=torch.load(restartPath)
#    model.load_state_dict(restart['model_state_dict'])
#    optimizer.load_state_dict(restart['optimizer_state_dict'])
#    for g in optimizer.param_groups:
#        g['lr']=learningRate
#    logging.info(f"Loading previously saved model: {restartPath}")
    

ModelBlock.runTraining(model, device, optimizer, train, test, deltaNg=deltaNg,
    config=config, path=outputPath, numEpochs=numEpochs)
