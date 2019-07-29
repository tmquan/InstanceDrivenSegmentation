from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import random
import shutil
import logging
import argparse
from natsort import natsorted

import cv2
import skimage.io
import skimage.measure
import numpy as np 

# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision

# # Using tensorflow
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model

# An efficient dataflow loading for training and testing
import tensorpack.dataflow as df
from tensorpack import *
import tqdm
import Augmentor

#
# Global configuration
#
BATCH = 10
EPOCH = 100000
SHAPE = 512
NF = 64

#
# Create the data flow using tensorpack dataflow (independent from tf and pytorch)
#
# TODO
class RandomSeedData(df.DataFlow):
    def __init__(self, size=100, datadir='data/', verbose=True, istrain=True):
        super(RandomSeedData, self).__init__()
        self.size = size

        # Read the image
        self.imagedir = os.path.join(datadir, 'image')
        self.imagefiles = natsorted(glob.glob(self.imagedir + '/*.*'))
        self.images = []
        for imagefile in self.imagefiles:
            # print(imagefile)
            # image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
            # image = cv2.resize(image, (SHAPE, SHAPE))
            # self.images.append(image)
            image = skimage.io.imread(imagefile)
            self.images.append(image)
        self.images = np.array(np.squeeze(self.images))

        # Read the label data
        self.labeldir = os.path.join(datadir, 'label')
        self.labelfiles = natsorted(glob.glob(self.labeldir + '/*.*'))
        self.labels = []
        for labelfile in self.labelfiles:
            # print(labelfile)
            # label = cv2.imread(labelfile, cv2.IMREAD_GRAYSCALE)
            # label = cv2.resize(label, (SHAPE, SHAPE))
            # self.labels.append(label)
            label = skimage.io.imread(labelfile)
            self.labels.append(label)
        self.labels = np.array(np.squeeze(self.labels))

        print(self.images.shape)
        print(self.labels.shape)
        self.istrain=istrain

    def normal(self, x, size):
        return (int)(x * (size - 1) + 0.5)

    
    def __iter__(self):
        for _ in range(self.size):
            randidx = np.random.randint(self.size)

            # Produce the label
            # Note that we take the position = 255
            image = np.squeeze(self.images[randidx]).astype(np.uint16)
            label = np.squeeze(self.labels[randidx]).astype(np.uint16)
            # label = skimage.measure.label(label)
            
            # Random crop here
            if self.istrain:
                # p = Augmentor.Pipeline()
                # p.crop_by_size(probability=1, width=SHAPE, height=SHAPE, centre=False)
                # image = p._execute_with_array(image) 
                # label = p._execute_with_array(label) 
                ystart = np.random.randint(0, image.shape[0]-SHAPE)
                xstart = np.random.randint(0, image.shape[1]-SHAPE)
                image = image[ystart:ystart+SHAPE, xstart:xstart+SHAPE]
                label = label[ystart:ystart+SHAPE, xstart:xstart+SHAPE]

            image = np.squeeze(image).astype(np.uint16)
            label = np.squeeze(label).astype(np.uint16)
            label = skimage.measure.label(label)
            
            y0 = 0
            x0 = 0
            value = label[y0, x0]
            # while value == 0:
                # Produce the distance transform with random location
            y_x_loc = np.random.uniform(0, 1, size=self.size)
            y0, x0 = y_x_loc[0], y_x_loc[1]  
            x0 = self.normal(x0, SHAPE)
            y0 = self.normal(y0, SHAPE)
            value = label[y0, x0]

            field = 255*np.ones_like(image).astype(np.uint8)
            field[y0, x0] = 0
            field = cv2.distanceTransform(field, cv2.DIST_L2, 3)
            field = cv2.normalize(field, field, 0, 255.0, cv2.NORM_MINMAX)
            field = 255-field.astype(np.uint8)

            membr = np.zeros_like(image)
            
            if value>0:
                membr[label==value] = 255
            membr = membr.astype(np.uint8)

            # print(randidx, image.shape, field.shape, membr.shape)
            # cv2.imshow('debug', np.concatenate([image, 
            #                                     field, 
            #                                     membr], axis=1))
            # cv2.waitKey(10)
            
            # Expand the dimension
            # image = np.expand_dims(image, axis=0)
            # field = np.expand_dims(field, axis=0)
            # membr = np.expand_dims(membr, axis=0)

            yield [image.astype(np.float32) / 255.0, 
                   field.astype(np.float32) / 255.0,
                   membr.astype(np.float32) / 255.0]

#
# Create the model
#
# TODO
def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

class Conv_residual_conv(nn.Module):
    def __init__(self,in_dim,out_dim,act_fn):
        super(Conv_residual_conv,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim,self.out_dim,act_fn)
        self.conv_2 = conv_block_3(self.out_dim,self.out_dim,act_fn)
        self.conv_3 = conv_block(self.out_dim,self.out_dim,act_fn)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class FusionGenerator(nn.Module):

    def __init__(self,input_nc=2, output_nc=1, ngf=16):
        super(FusionGenerator,self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FusionNet------\n")

        # encoder

        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output

        self.out = nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Tanh()


        # initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self,input):

        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)/2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)/2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        out = self.out_2(out)
        #out = torch.clamp(out, min=-1, max=1)
        out = (out + 1.0) / 2.0
        return out

#
# Perform sample
#
# TODO

#
# Main
#
if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='the image directory')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--sample', action='store_true', help='run inference')
    args = parser.parse_args()
    
    # Choose the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    if args.sample:
        # TODO: Run the inference
        pass
    else:
        #
        # Train from scratch or load the pretrained network
        #
        # TODO: Load the pretrained model
        if args.load:
            pass
            
        # Initialize the program
        writer = SummaryWriter()
        use_cuda = torch.cuda.is_available()
        xpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        step = 0

        # TODO
        net = FusionGenerator()
        optimizer = optim.Adam(net.parameters(), lr=3e-6)
        criterion = nn.L1Loss()
    
        # Load the pretrained model if train from that
        if args.load: 
            # TODO
            pretrained_dict = torch.load(args.load) #("renderer.pkl")
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

        # Create a dataflow
        # TODO
        ds_train = RandomSeedData(size=100, datadir=args.data) #'data/SNEMI/db_train')
        augs = [
                # imgaug.ResizeShortestEdge(270),
                imgaug.RandomCrop(SHAPE), 
                imgaug.Flip(horiz=True), 
                imgaug.Flip(vert=True), 
                imgaug.Transpose()
                ]
        ds_train = AugmentImageComponents(ds_train, augs, (0, 1, 2))
        # ds_train = MapData(ds_train, lambda dp: [np.expand_dims(img, axis=0) for img in dp])
        ds_train = MapData(ds_train, lambda dp: [np.expand_dims(dp[0], axis=0), 
                                                 np.expand_dims(dp[1], axis=0), 
                                                 np.expand_dims(dp[2], axis=0), 
                                                 ])
        ds_train = df.BatchData(ds_train, batch_size=BATCH)
        ds_train = df.PrintData(ds_train)
        # ds_train = df.PrefetchDataZMQ(ds_train, nr_proc=4)

        # ds_valid = RandomSeedData(10)
        # ds_valid = df.BatchData(ds_valid, batch_size=1)
        # ds_valid = df.PrefetchDataZMQ(ds_valid, nr_proc=4)

        # train
        max_step = 10000000
        for epoch in range(EPOCH):
            for mb_train in ds_train.get_data():
                step = step+1
                if step > max_step:
                    exit()
                # print("Step: {}, Epoch {}".format(step, epoch))

                image = torch.tensor(mb_train[0]).float()
                field = torch.tensor(mb_train[1]).float()
                membr = torch.tensor(mb_train[2]).float()

                # if use_cuda:
                #     net = net.cuda()
                #     zvector = zvector.cuda()
                #     picture = picture.cuda()
                net = net.to(xpu)
                image = image.to(xpu)
                field = field.to(xpu)
                membr = membr.to(xpu)
                
                # TODO: Forward pass
                estim = net(torch.cat([image, field], 1))

                # Reset the optimizer
                optimizer.zero_grad()

                # TODO: Loss calculation
                # loss = criterion(estim, membr)
                def dice_loss(input, target):
                    smooth=.001
                    input=input.view(-1)
                    target=target.view(-1)
                    
                    return(1-2*(input*target).sum()/(input.sum()+target.sum()+smooth))
                loss = dice_loss(estim, membr)
                loss.backward()
                optimizer.step()
                
               
                # TODO: Log to tensorboard after n steps
                writer.add_scalar('train/loss', loss.item(), step)               
                # writer.add_image('train/estim', estim[0], step, dataformats='HW')
                # writer.add_image('train/picture', picture[0], step, dataformats='HW')
                writer.add_image('train/estim', torch.cat([image, field, membr, estim], 3)[0][0], step, dataformats='HW')
                
                # TODO: Lowering the learning rate after n steps
                if step < 200000:
                    lr = 1e-4
                elif step < 400000:
                    lr = 1e-5
                else:
                    lr = 1e-6
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

               
                # TODO: Valid set after n steps, need to implement as callback
                # if step % 100 == 0:
                #     net.eval()
                #     for minibatch_valid in ds_valid.get_data():
                #         zvector_valid = torch.tensor(minibatch_valid[0]).float()
                #         estim_valid = torch.tensor(minibatch_valid[1]).float()
                #         estim_valid = net(zvector_valid)
                #         loss = criterion(estim, estim)
                #         writer.add_scalar('valid/loss', loss.item(), step)
                #         writer.add_image('valid/estim', estim_valid[0], dataformats='HW')
                #         writer.add_image('valid/estim', estim_valid[0], dataformats='HW')

               
                # TODO: Log to console after n steps, need to implement as callback
                if True:
                    print('\rStep {} \tLoss: {:.4f}'.format(step, loss.item()), end="")   

                # TODO: Save the model after n steps, need to implement as callback
                if step % 10000 == 0:
                    print('\rStep {} \tLoss: {:.4f}'.format(step, loss.item()))
                    torch.save(net.cpu().state_dict(), "driver_snemi.pkl")
                    # net.cuda()
                    net = net.to(xpu)
