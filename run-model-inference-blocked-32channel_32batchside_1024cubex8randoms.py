"""
The goal of this code is to time cutting a grid if the grid is
stored wholly on the CPU or the GPU.
"""
import logging
logging.basicConfig(level=logging.DEBUG)

import torch
import time
import sys
import numpy as np
import yaml
import ModelBlock_32channel_savelosses as ModelBlock
import SimulationUtils_1024cubex8randoms as SU


if len(sys.argv) < 2:
    raise RuntimeError("Not enough parameters")
jobname=sys.argv[1]
with open(f"input/{jobname}.yml","r") as ff:
    config=yaml.safe_load(ff)

simNum=config["simNum"]
shrinkFactor=config["shrink"]
checkptPath=config["checkptPath"]
runOnGPU=config.get("runOnGPU",True)
Ng=config["Ngrid"]
subGridSize=config["subGridSize"]
batchSide=config.get("batchSide",128)
outpath=config["outPath"]
verbose=config.get("verbose",True)

padWidth=subGridSize//2
dtype=ModelBlock.getDtype(config.get("modelBits",32))

device = torch.device("cuda" if (torch.cuda.is_available() & runOnGPU) else "cpu")
logging.info(f"Running on device={device}....")


# Input and output
inputSim=SU.SimulationDataset(nsims=1, ngrid=Ng, subgridSize=subGridSize, batchSide=batchSide)
sim1=SU.loadRecon(simNum)
if sim1.shape[0]!=Ng:
    raise RuntimeError("Ng does not match input data")
inputSim.load(sim1,0,normalize=True)
logging.info("Data created...")

model = ModelBlock.Mao_Recon_CNN_Blocked(shrink=shrinkFactor, batchSide=batchSide).to(device)

checkpt=torch.load(checkptPath)
model.load_state_dict(checkpt["best_state_dict"]) # best_state
logging.info(f"Read model from {checkptPath}....")
model.eval()

# Define the pytorch tensors that we'll need.
blockSide=batchSide+2*inputSim.padWidth
inputGrids = torch.empty((1,1,blockSide,blockSide,blockSide), dtype=torch.float32)
inputGrids_gpu=torch.empty_like(inputGrids, device=device)
simOutput_device=torch.empty((Ng, Ng, Ng), dtype=dtype, device=device)
logging.info("Grids allocated....")

## Cutting
logging.info(f"Running with an x/y/z stride of delta={batchSide}")
tic=time.perf_counter()
with torch.no_grad():
    for i0 in range(0,Ng,batchSide):
        toc = time.perf_counter()
        print(f"Doing slab={i0}, elapsed time={toc-tic:.4f}")
        for j0 in range(0,Ng,batchSide):
            for k0 in range(0,Ng,batchSide):
                cen=(0,i0,j0,k0)
                inputSim.getPaddedBlock(inputGrids, *cen)
                inputGrids_gpu.copy_(inputGrids, non_blocking=True)

                output = model(inputGrids_gpu)
                output = output.reshape((batchSide,batchSide,batchSide))

                ilo=i0
                ihi=ilo+batchSide
                jlo=j0
                jhi=jlo+batchSide
                klo=k0
                khi=klo+batchSide

                simOutput_device[ilo:ihi,jlo:jhi,klo:khi] = output

                # End of loop

logging.info(f"Time to do the model evaluation = {toc-tic:.4f}")

simOutput=simOutput_device.cpu().numpy()
simOutput=np.asarray(simOutput, dtype=np.float32)
np.save(outpath, simOutput)
