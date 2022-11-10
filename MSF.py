import torch
import numpy as np
import random

def SPM(model, inputs):
    model.zero_grad() 
    inputs = inputs.detach().clone()
    inputs.requires_grad_()
    
    output = model(inputs)
    output.backward(torch.ones(output.shape).cuda())
    
    fag = torch.abs(inputs.grad)
    fag = torch.max(fag, dim=1, keepdim=True)[0]
    
    return fag

def MSF_fun(model, data):
    '''
    strong correlation: sc
    weak correlation: wc
    '''
    model.eval()
    mask = SPM(model, data)
    sc_mask = torch.ones_like(mask)
    wc_mask = torch.ones_like(mask)
    imgh = imgw = 256

    for i in range(len(mask)): 
        maxind = np.argsort(mask[i].cpu().numpy().flatten())[::-1]
        pointcnt = 0
        for pointind in maxind:
            pointx = pointind//imgw
            pointy = pointind % imgw

            if sc_mask[i][0][pointx][pointy] == 1:

                patch_half = 5    # patch_weight = patch_height = 2*patch_half
                sc_top = max(pointx-patch_half, 0)
                sc_bot = min(pointx+patch_half, imgh)
                sc_lef = max(pointy-patch_half, 0)
                sc_rig = min(pointy+patch_half, imgw)

                sc_mask[i][:, sc_top:sc_bot, sc_lef:sc_rig] = torch.zeros_like(sc_mask[i][:, sc_top:sc_bot, sc_lef:sc_rig])
                
                # obtain the wc mask
                wc_pointx = random.randint(patch_half, imgh-patch_half)
                wc_pointy = random.randint(patch_half, imgw-patch_half)
                if sc_mask[i][0][wc_pointx][wc_pointy] == 1:
                    wc_top = wc_pointx-patch_half
                    wc_bot = wc_pointx+patch_half
                    wc_lef = wc_pointy-patch_half
                    wc_rig = wc_pointy+patch_half
                    
                    wc_mask[i][:, wc_top:wc_bot, wc_lef:wc_rig] = torch.zeros_like(wc_mask[i][:, wc_top:wc_bot, wc_lef:wc_rig])

                pointcnt += 1
                if pointcnt >= 4:
                    break

    final_data = sc_mask * wc_mask * data + (2-sc_mask-wc_mask) * (torch.rand_like(data)*2-1.)

    return final_data

