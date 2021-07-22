# %% import python-library
# original lib
import common as com
import Model as HVAE

# torch
import torch
from torch.autograd import Variable
import numpy
import os
import pandas as pd
import random

# %% Hyperparameter
n_mels = 128
frames = 5
n_fft = 2048
hop_length = 1024
power = 2.0

# %%
def run_train_epoch(Batch_size = 128, Beta = 4, upto=None):
    torch.set_grad_enabled(True) # enable/disable grad for efficiency of forwarding test batches
    model.train()
    x = train_data
    N,D = x.size()
    B = Batch_size # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    for step in range(nsteps):
        
        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B]).float()
        
        # get the logits, potentially run the same batch a number of times, resampling each time
        recon_x, recon_sigma, mu, logvar = model(xb)
        
        # evaluate the binary cross entropy loss
        loss = HVAE.loss_function(Beta, recon_x, recon_sigma, xb, mu, logvar)
        lossf = loss.data.item()
        lossfs.append(lossf)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    print("train epoch average loss: %f" % (numpy.mean(lossfs)))
# ----------------------------------------------------------------------


# %% 
# Hyper parameter 
# BN : Decorder Architecture (True: seperate)
# embedding_dim: mu, sigma dimension (encoder)
# Beta: Reconstruct Error weight

BN = True
embedding_dim = 16
layers = [640, 640, 512, 256]
Beta = 2
Epoch = 200
Batch_size = 128
os.makedirs('./model', exist_ok=True)
Model_save_dir = 'model/Setting_12.pt'

# %%
OK_csv = pd.read_csv('/home/jongwook95.lee/Audio_anomal/Comp/Data/both_vib_noise.csv', header=None)
files = list(OK_csv.sample(n=4000, random_state=5024)[0])
train_data = com.list_to_vector_array_tdms(files, channel = 'timesignal_v1',
                                    msg="generate train_dataset",
                                    n_mels = n_mels,
                                    frames = frames,
                                    n_fft = n_fft,
                                    hop_length = hop_length,
                                    power = power)

train_data = torch.from_numpy(train_data).cuda()
# %%
train_data.shape


# %% 모델 생성
numpy.random.seed(102)
torch.manual_seed(102)
torch.cuda.manual_seed_all(102)

if BN == True:
    model = HVAE.VAE_BN(layers=layers, embedding_dim=embedding_dim)
else:
    model = HVAE.VAE_NoBN(layers=layers, embedding_dim=embedding_dim)
print("number of model parameters:",sum([numpy.prod(p.size()) for p in model.parameters()]))
model.cuda()

# %% set up the optimizer
opt = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)

# %%
for epoch in range(Epoch):
    print("epoch %d" % (epoch, ))
    scheduler.step(epoch)
    run_train_epoch(Batch_size = Batch_size, Beta = Beta)

# %% Model 저장
torch.save(model.state_dict(), Model_save_dir)




# %%
  
# %%
