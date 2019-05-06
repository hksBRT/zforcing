from model import ZForcing
import numpy as np
import torch
import time
from torch.autograd import Variable
import pdb
# from blizzard_data import Blizzard_tbptt
if __name__ == "__main__":
    seed = 1234
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    inp_dim = 200
    model = ZForcing(inp_dim=inp_dim, emb_dim=512, rnn_dim=1024, z_dim=256,
                     mlp_dim=512, out_dim=400, nlayers=1,
                     cond_ln=True)
    model.z_force = True
    bsz = 32
    hidden = model.init_hidden(bsz)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)

    kld_step = 0.00005
    aux_step = 0.00005
    kld_weight = 0.2
    aux_weight = 0.0
    nbatches = 10
    t = time.time()
    old_valid_loss = np.inf
    b_fwd_loss, b_bwd_loss, b_kld_loss, b_aux_loss, b_all_loss = \
        (0., 0., 0., 0., 0.)
    model.train()

    x = np.random.random((bsz,40,inp_dim))
    y = np.random.random((bsz,40,inp_dim))
    x_mask = np.ones((bsz,40))
    
    
    # fn = './data_0.npy' 
    
    # data = np.load(fn, allow_pickle=True)
    # file_name = 'data_0'
    # train_data = Blizzard_tbptt(name='train',
    #                             path='./',
    #                             frame_size=200,
    #                             file_name=file_name,
    #                             X_mean=0,
    #                             X_std=1)

    x = x.transpose(1, 0, 2)
    y = y.transpose(1, 0, 2)
    x_mask = x_mask.T

    opt.zero_grad()

    x = Variable(torch.from_numpy(x)).float()
    y = Variable(torch.from_numpy(y)).float()
    x_mask = Variable(torch.from_numpy(x_mask)).float()

    fwd_nll, bwd_nll, aux_nll, kld = model(x, y, x_mask, hidden)