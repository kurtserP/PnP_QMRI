import os
import torch
import numpy as np
from DNCNN.network_dncnn import DnCNN as net


def filter_dncnn(img, model,nb):
    n_channels = 1
    device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = os.path.join('DNCNN/models', model + '.pth')
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    #model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)

    img_min = np.min(img)
    img_max = np.max(img)

    img_scale = (img-img_min)/(img_max-img_min)
    img_L = np.expand_dims(img_scale, axis=2)
    img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().unsqueeze(0)
    img_L = img_L.to(device)
    img_E = model(img_L)

    filtered_img = img_E.cpu().detach().numpy()[0, 0, :, :]
    filtered_img = filtered_img*(img_max-img_min)+img_min
    return filtered_img

