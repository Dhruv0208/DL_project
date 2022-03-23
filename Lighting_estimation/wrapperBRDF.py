import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import models

# Return triplet of predictions, ground-truth and error
def wrapperBRDF(dataBatch, opt,
    encoder, albedoDecoder, normalDecoder, roughDecoder, depthDecoder ):

    # Load data from cpu to gpu
    albedo_cpu = dataBatch['albedo']
    albedoBatch = Variable(albedo_cpu ).cuda()

    normal_cpu = dataBatch['normal']
    normalBatch = Variable(normal_cpu ).cuda()

    rough_cpu = dataBatch['rough']
    roughBatch = Variable(rough_cpu ).cuda()

    depth_cpu = dataBatch['depth']
    depthBatch = Variable(depth_cpu ).cuda()

    segArea_cpu = dataBatch['segArea']
    segEnv_cpu = dataBatch['segEnv']
    segObj_cpu = dataBatch['segObj']
    seg_cpu = torch.cat( [segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
    segBatch = Variable( seg_cpu ).cuda()

    segBRDFBatch = segBatch[:, 2:3, :, :]
    segAllBatch = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]

    # Load the image from cpu to gpu
    im_cpu = (dataBatch['im'] )
    imBatch = Variable(im_cpu ).cuda()



    ########################################################
    inputBatch = imBatch
    # Initial Prediction
    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    albedoPred = 0.5 * (albedoDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)
    normalPred = normalDecoder(imBatch, x1, x2, x3, x4, x5, x6)
    roughPred = roughDecoder(imBatch, x1, x2, x3, x4, x5, x6)
    depthPred = 0.5 * (depthDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)

    albedoBatch = segBRDFBatch * albedoBatch
    albedoPred1 = models.LSregress(albedoPred.detach() * segBRDFBatch.expand_as(albedoPred),
            albedoBatch * segBRDFBatch.expand_as(albedoBatch), albedoPred )
    albedoPred1 = torch.clamp(albedoPred1, 0, 1)

    depthPred1 = models.LSregress(depthPred.detach() *  segAllBatch.expand_as(depthPred),
            depthBatch * segAllBatch.expand_as(depthBatch), depthPred)

    ## Compute Errors
    pixelObjNum = (torch.sum(segBRDFBatch ).cpu().data).item()
    pixelAllNum = (torch.sum(segAllBatch ).cpu().data).item()

    albedoErr = torch.sum( (albedoPred1 - albedoBatch )
            * (albedoPred1 - albedoBatch) * segBRDFBatch.expand_as(albedoBatch) / pixelObjNum / 3.0)
    normalErr = torch.sum( (normalPred - normalBatch)
        * (normalPred - normalBatch) * segAllBatch.expand_as(normalBatch) ) / pixelAllNum / 3.0
    roughErr = torch.sum( (roughPred - roughBatch)
        * (roughPred - roughBatch) * segBRDFBatch ) / pixelObjNum


    depthErr = torch.sum( (torch.log(depthPred1 + 1) - torch.log(depthBatch + 1) )
            * ( torch.log(depthPred1 + 1) - torch.log(depthBatch + 1) ) * segAllBatch.expand_as(depthBatch ) ) / pixelAllNum


    return [albedoPred, albedoErr], [normalPred, normalErr], \
        [roughPred, roughErr], [depthPred, depthErr]