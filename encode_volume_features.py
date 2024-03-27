import numpy as np
import torch
from torch import nn
from models import unet
# from nnunet.network_architecture.generic_UNet import Generic_UNet_predict
# from nnunet.network_architecture.initialization import InitWeights_He
import torchio as tio
import glob
from sys import argv


"""
Data prep: 3D CT volume -> encoded feature representations
"""


def get_pretrained_model(pth, dev):
    """
    pth: path to pretrained UNet encoder state_dict
    dev: where to load the pt weights (str)
    """ 
    dev = torch.device(dev)

    pt_weights = torch.load(pth, map_location=torch.device(dev))

    num_input_channels = 1
    base_num_features = 32
    num_classes = 2
    net_num_pool_op_kernel_sizes = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
    net_conv_kernel_sizes = [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],[3, 3, 3], [3, 3, 3]]

    network = unet.Encoder(num_input_channels, base_num_features, num_classes, len(net_num_pool_op_kernel_sizes),
               net_num_pool_op_kernel_sizes, net_conv_kernel_sizes)

    # put model on same device as the pt weights model
    network.to(dev)
    network.load_state_dict(pt_weights)

    return network


def masked_data_prep(pth, seg_net, dev):
    # """
    # pth: path to .npz file  (nnUnet preprocessed volume)
    # seg_net: path to pretrained UNet encoder
    # dev: device to run prep
    # """
    # dev = torch.device(dev)
    # load file (dat is modality, z, x, y) -> modality:1 == mask
    dat = np.load(pth)
    # extract volume data, extract mask data 
    # convert all masks values != 1 to 0
    # multiply vol * mask
    # create subject
    # crop 
    vol = dat['data'][0,:]
    mask = dat['data'][1,:]
    mask[mask != 1] = 0
    vol = vol*mask
    vol = torch.tensor(vol).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)
    sub = tio.Subject(ct = tio.ScalarImage(tensor=vol), mask = tio.LabelMap(tensor=mask))
    sub = tio.CropOrPad(mask_name = 'mask')(sub)

    # sample all patches, run through encoder to get outputs of interest
    # sampler will throw ValueError if cropped image has any dimensions smaller than the patch dimensions
    
    try:
        sampler = tio.GridSampler(subject=sub,patch_size=(28, 256, 256))
    except ValueError as ve:
        old_dim = sub.shape
        print(old_dim)
        if old_dim[1] <  28:
            new_x = 28
        else:
            new_x = old_dim[1]

        if old_dim[2] <  256:
            new_y = 256
        else:
            new_y = old_dim[2]

        if old_dim[3] <  256:
            new_z = 256
        else:
            new_z = old_dim[3]

        sub = tio.CropOrPad(target_shape = (new_x, new_y, new_z), mask_name = 'mask')(sub)
        sampler = tio.GridSampler(subject=sub, patch_size=(28, 256, 256))

    all_patches = torch.empty(0)
    seg_net.eval()
    for i, patch in enumerate(sampler):
        # need to run one patch at a time due to GPU memory limitations... 
    	with torch.no_grad():
            # Generic_UNet_predict returns: Predicted mask, skip connections (list), decoding block, decoder outputs (list)
            skips, db = seg_net(patch.ct.data.unsqueeze(0).to(dev))
            pred_in = torch.empty(0)
            for s in skips:
                p = torch.mean(s, axis = [2,3,4])
                pred_in = torch.cat((pred_in, p.detach().cpu()), axis =-1)
            # cat decoding block too if including...
            # p = torch.mean(db, axis = [2,3,4])
            # pred_in = torch.cat((pred_in, p.detach().cpu()), axis =-1)
            all_patches = torch.cat((all_patches, pred_in), dim=0)

    return all_patches

if __name__ == "__main__":

    # pretrained encoder weights (filename and full path)
    pt_pth = str(argv[1])
    # path to data dir (preprocessed nnUnet volumes)
    data_pth = str(argv[2])
    # path to write processed data
    out_pth = str(argv[3])
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_encoder = get_pretrained_model(pt_pth, dev)

    for n in glob.glob(data_pth+"*.npz"):
        dat = masked_data_prep(n, pt_encoder, dev)
        
        torch.save(dat, out_pth+n.split("/")[-1][:-4]+".pt")