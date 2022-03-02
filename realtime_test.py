#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 04:15:34 2020

@author: adarsh
"""

import argparse
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import datasets
from multiscaleloss import compute_photometric_loss, estimate_corresponding_gt_flow, flow_error_dense, smooth_loss
from datetime import datetime

from tensorboardX import SummaryWriter
from util import flow2rgb, AverageMeter, save_checkpoint
import cv2
import torch
import os, os.path
import numpy as np
import h5py
import sys

from vis_utils import  flow_viz_np
from torch.utils.data import Dataset, DataLoader

from dvs.DAVIS346 import DAVIS346

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='Spike-FlowNet Real-time Testing',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--savedir', type=str, metavar='DATASET', default='results_dt4',
                    help='results save dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='spike_flownets',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers')


parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')

parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder')
parser.add_argument('--div-flow', default=1,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--render', dest='render', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--data-split', default=10, type=int,
                     help='Number of split frames')
parser.add_argument('--dt', default=1, type=int,
                     help='Number of split frames')
parser.add_argument('--packet-interval', default=2000, type=int,
                     help='Time in us to accumulate events for a frame')
parser.add_argument('--eval-fps', default=30, type=int,
                     help='Evaluate model at this FPS')
parser.add_argument('--min-activity', default=50, type=int,
                     help='Minimum activity to register frame')

args = parser.parse_args()

#Initializations
best_EPE = -1
n_iter = 0
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_resize = 256
event_interval = 0
spiking_ts = 1
sp_threshold = 0
spike_repeat = 4



testenv = 'indoor_flying1'

testdir = os.path.join(args.data, testenv)

testfile = testdir + '/' + testenv + '_data.hdf5'

gt_file = testdir + '/' + testenv + '_gt.hdf5'

args.testfile = testfile
args.gt_file = gt_file

if args.dt == 1:
    args.pretrained = 'pretrain/checkpoint_dt1.pth.tar'
elif args.dt == 4:
     args.pretrained = 'pretrain/checkpoint_dt4.pth.tar'
     
#Print args
print(' ' * 20 + 'Options')
for k, v in vars(args).items():
  print(' ' * 20 + k + ': ' + str(v))


class Test_loading(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.dt = 4
        self.xoff = 45
        self.yoff = 2
        self.split = 10

        d_set = h5py.File(testfile, 'r')
        # Training input data, label parse
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        if (index + 20 < self.length) and (index > 20):
            aa = np.zeros((256, 256, int(self.dt*self.split/2)))
            bb = np.zeros((256, 256, int(self.dt*self.split/2)))
            cc = np.zeros((256, 256, int(self.dt*self.split/2)))
            dd = np.zeros((256, 256, int(self.dt*self.split/2)))

            for k in range(int(self.dt/2)):
                im_on = np.load(testdir + '/count_data/' + str(int(index+k+1))+'.npy')
                im_off = np.load(testdir + '/count_data/' + str(int(index+self.dt/2+k+1))+'.npy')
                aa[:,:,self.split*k:self.split*(k+1)] = im_on[0,self.yoff:-self.yoff, self.xoff:-self.xoff,:].astype(float)
                bb[:,:,self.split*k:self.split*(k+1)] = im_on[1,self.yoff:-self.yoff, self.xoff:-self.xoff,:].astype(float)
                cc[:,:,self.split*k:self.split*(k+1)] = im_off[0,self.yoff:-self.yoff, self.xoff:-self.xoff,:].astype(float)
                dd[:,:,self.split*k:self.split*(k+1)] = im_off[1,self.yoff:-self.yoff, self.xoff:-self.xoff,:].astype(float)

            return aa, bb, cc, dd, self.image_raw_ts[index], self.image_raw_ts[index+self.dt]
        else:
            pp = np.zeros((image_resize,image_resize,int(self.split*self.dt/2)))
            return pp, pp, pp, pp, np.zeros((self.image_raw_ts[index].shape)), np.zeros((self.image_raw_ts[index].shape))

    def __len__(self):
        return self.length



def validate(test_loader, model, epoch, output_writers):
    global args, image_resize, sp_threshold, spike_repeat
    d_label = h5py.File(gt_file, 'r')
    gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
    print('label size', gt_temp.shape, gt_temp.shape[2], gt_temp.shape[3])
    gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
    d_label = None
    
    d_set = h5py.File(testfile, 'r')
    gray_image = d_set['davis']['left']['image_raw']

    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    batch_size_v = 4
    sp_threshold = 0.5

    AEE_sum = 0.
    AEE_sum_sum = 0.
    AEE_sum_gt = 0.
    AEE_sum_sum_gt = 0.
    percent_AEE_sum = 0.
    iters = 0.
    scale = 1

    for i, data in enumerate(test_loader, 0):
        former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off, st_time, ed_time = data

        if torch.sum(former_inputs_on + former_inputs_off) > 0:
            input_representation = torch.zeros(former_inputs_on.size(0), batch_size_v, image_resize, image_resize, former_inputs_on.size(3)).float()

            for b in range(batch_size_v):
                if b == 0:
                    input_representation[:, 0, :, :, :] = former_inputs_on
                elif b == 1:
                    input_representation[:, 1, :, :, :] = former_inputs_off
                elif b == 2:
                    input_representation[:, 2, :, :, :] = latter_inputs_on
                elif b == 3:
                    input_representation[:, 3, :, :, :] = latter_inputs_off

            # compute output
            input_representation = input_representation.to(args.device)
            output = model(input_representation.type(torch.cuda.FloatTensor), image_resize, sp_threshold)

            # pred_flow = output
            pred_flow = np.zeros((image_resize, image_resize, 2))
            output_temp = output.cpu()
            pred_flow[:, :, 0] = cv2.resize(np.array(output_temp[0, 0, :, :]), (image_resize, image_resize), interpolation=cv2.INTER_LINEAR)
            pred_flow[:, :, 1] = cv2.resize(np.array(output_temp[0, 1, :, :]), (image_resize, image_resize), interpolation=cv2.INTER_LINEAR)

            U_gt_all = np.array(gt_temp[:, 0, :, :])
            V_gt_all = np.array(gt_temp[:, 1, :, :])

            U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all, gt_ts_temp, np.array(st_time), np.array(ed_time))
            gt_flow = np.stack((U_gt, V_gt), axis=2)
            #   ----------- Visualization
            if epoch < 0:
                mask_temp = former_inputs_on + former_inputs_off + latter_inputs_on + latter_inputs_off
                mask_temp = torch.sum(torch.sum(mask_temp, 0), 2)
                mask_temp_np = np.squeeze(np.array(mask_temp)) > 0
                
                spike_image = mask_temp
                spike_image[spike_image>0] = 255
                if args.render:
                    cv2.imshow('Spike Image', np.array(spike_image, dtype=np.uint8))
                
                gray = cv2.resize(gray_image[i], (scale*image_resize,scale* image_resize), interpolation=cv2.INTER_LINEAR)
                if args.render:
                    cv2.imshow('Gray Image', cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

                out_temp = np.array(output_temp.cpu().detach())
                x_flow = cv2.resize(np.array(out_temp[0, 0, :, :]), (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                y_flow = cv2.resize(np.array(out_temp[0, 1, :, :]), (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                flow_rgb = flow_viz_np(x_flow, y_flow)
                if args.render:
                    cv2.imshow('Predicted Flow Output', cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))

                gt_flow_x = cv2.resize(gt_flow[:, :, 0], (scale * image_resize, scale * image_resize),interpolation=cv2.INTER_LINEAR)
                gt_flow_y = cv2.resize(gt_flow[:, :, 1], (scale * image_resize, scale * image_resize),interpolation=cv2.INTER_LINEAR)
                gt_flow_large = flow_viz_np(gt_flow_x, gt_flow_y)
                if args.render:
                    cv2.imshow('GT Flow', cv2.cvtColor(gt_flow_large, cv2.COLOR_BGR2RGB))
                
                masked_x_flow = cv2.resize(np.array(out_temp[0, 0, :, :] * mask_temp_np), (scale*image_resize,scale* image_resize), interpolation=cv2.INTER_LINEAR)
                masked_y_flow = cv2.resize(np.array(out_temp[0, 1, :, :] * mask_temp_np), (scale*image_resize, scale*image_resize), interpolation=cv2.INTER_LINEAR)
                flow_rgb_masked = flow_viz_np(masked_x_flow, masked_y_flow)
                if args.render:
                    cv2.imshow('Masked Predicted Flow', cv2.cvtColor(flow_rgb_masked, cv2.COLOR_BGR2RGB))
                
                gt_flow_cropped = gt_flow[2:-2, 45:-45]
                gt_flow_masked_x = cv2.resize(gt_flow_cropped[:, :, 0]*mask_temp_np, (scale*image_resize, scale*image_resize),interpolation=cv2.INTER_LINEAR)
                gt_flow_masked_y = cv2.resize(gt_flow_cropped[:, :, 1]*mask_temp_np, (scale*image_resize, scale*image_resize),interpolation=cv2.INTER_LINEAR)
                gt_masked_flow = flow_viz_np(gt_flow_masked_x, gt_flow_masked_y)
                if args.render:
                    cv2.imshow('GT Masked Flow', cv2.cvtColor(gt_masked_flow, cv2.COLOR_BGR2RGB))
                
                cv2.waitKey(1)

            image_size = pred_flow.shape
            full_size = gt_flow.shape
            xsize = full_size[1]
            ysize = full_size[0]
            xcrop = image_size[1]
            ycrop = image_size[0]
            xoff = (xsize - xcrop) // 2
            yoff = (ysize - ycrop) // 2

            gt_flow = gt_flow[yoff:-yoff, xoff:-xoff, :]

            AEE, percent_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = flow_error_dense(gt_flow, pred_flow, (torch.sum(torch.sum(torch.sum(input_representation, dim=0), dim=0), dim=2)).cpu())

            AEE_sum = AEE_sum + args.div_flow * AEE
            AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

            AEE_sum_gt = AEE_sum_gt + args.div_flow * AEE_gt
            AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

            percent_AEE_sum += percent_AEE

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i < len(output_writers):  # log first output of first batches
                # if epoch == 0:
                #     mean_values = torch.tensor([0.411,0.432,0.45], dtype=input_representation.dtype).view(3,1,1)
                output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

            iters += 1

    print('-------------------------------------------------------')
    print('Mean AEE: {:.2f}, sum AEE: {:.2f}, Mean AEE_gt: {:.2f}, sum AEE_gt: {:.2f}, mean %AEE: {:.2f}, # pts: {:.2f}'
                  .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters, percent_AEE_sum / iters, n_points))
    print('-------------------------------------------------------')
    gt_temp = None

    return AEE_sum / iters

def live_validate(input_representation, model, scale=1):

    sp_threshold = 0.5

    output = model(input_representation.unsqueeze(0).to(args.device), image_resize, sp_threshold)
    out_temp = output.cpu().detach()

    #   ----------- Visualization

    mask_temp = torch.sum(input_representation, 0)
    mask_temp = torch.sum(mask_temp, 2)
    mask_temp_np = (torch.squeeze(mask_temp) > 0).float()

    
    spike_image = mask_temp
    spike_image[spike_image>0] = 255
    # cv2.imshow('Spike Image', np.array(spike_image, dtype=np.uint8))
    
    # gray = cv2.resize(gray_image[i], (scale*image_resize,scale* image_resize), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('Gray Image', cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

    # out_temp = np.array(output_temp.cpu().detach())
    x_flow = cv2.resize(np.array(out_temp[0, 0, :, :]), (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
    y_flow = cv2.resize(np.array(out_temp[0, 1, :, :]), (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
    flow_rgb = flow_viz_np(x_flow, y_flow)
    # cv2.imshow('Predicted Flow Output', cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))

    masked_x_flow = cv2.resize(np.array(out_temp[0, 0, :, :] * mask_temp_np), (scale*image_resize,scale* image_resize), interpolation=cv2.INTER_LINEAR)
    masked_y_flow = cv2.resize(np.array(out_temp[0, 1, :, :] * mask_temp_np), (scale*image_resize, scale*image_resize), interpolation=cv2.INTER_LINEAR)
    flow_rgb_masked = flow_viz_np(masked_x_flow, masked_y_flow)
    # cv2.imshow('Masked Predicted Flow', cv2.cvtColor(flow_rgb_masked, cv2.COLOR_BGR2RGB))

    # cv2.waitKey(1)
    
    del input_representation
    
    return spike_image, flow_rgb, flow_rgb_masked


# Simple ISO 8601 timestamped logger
def log(s):
  print('\n[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def visualize(spike_image, flow_rgb, flow_rgb_masked):
    cv2.imshow('Spike Image', np.array(spike_image, dtype=np.uint8))
    cv2.imshow('Predicted Flow Output', cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))
    cv2.imshow('Masked Predicted Flow', cv2.cvtColor(flow_rgb_masked, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)
    


# create model
if args.pretrained:
    network_data = torch.load(args.pretrained)
    #args.arch = network_data['arch']
    print("=> using pre-trained model '{}'".format(args.arch))
else:
    network_data = None
    print("=> creating model '{}'".format(args.arch))

model = models.__dict__[args.arch](network_data).cuda()
model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True
model.eval()

spikecam = DAVIS346(shape=(image_resize, image_resize), data_split=args.data_split, dt=args.dt, packet_interval=args.packet_interval, min_activity=args.min_activity)

end = time.time()
eval_base = time.time()
start = time.time()

eval_fps = args.eval_fps

eval_time = 1.0/eval_fps
forward_time = eval_time
total_time = eval_time


torch.set_num_threads(8)
print('Number of Threads: ', torch.get_num_threads())

while True:
    try:
        for i in range(5*args.dt):
            spikecam.get_event_buffer()
            # spikecam.show_image()
        
        data_time = time.time() - end
        
        # if args.render:
        #     input_representation = spikecam.get_encoded_buffer()
        #     spike_img = torch.sum(torch.sum(input_representation, 0), 2)
        #     spike_img[spike_img>0] = 255
        #     cv2.imshow('Spike Image', np.array(spike_img, dtype=np.uint8))
        #     cv2.waitKey(1)
        
        #Forward pass
        if time.time() - start > eval_time:
            start = time.time()
            input_representation = spikecam.get_encoded_buffer()
            
            spike_image, flow_rgb, flow_rgb_masked = live_validate(input_representation, model, scale=1)
            
            if args.render:
                visualize(spike_image, flow_rgb, flow_rgb_masked)
            
            forward_time = time.time() - start
            total_time = time.time() - eval_base + data_time
            eval_base = time.time()
            
        
        end = time.time()
        printlog = '\033[K \r' + 'Data FPS: ' + str(1//data_time) + '\t  Forward-pass FPS: ' + str(1//forward_time) + '\t  Total FPS: ' + str(1//total_time)
        sys.stdout.write(printlog)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            spikecam.davis346.shutdown()
            break
        
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        spikecam.davis346.shutdown()
        break
        

cv2.destroyAllWindows()
spikecam.davis346.shutdown()
