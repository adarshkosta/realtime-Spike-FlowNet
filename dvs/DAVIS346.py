#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:10:28 2019

@author: akosta
"""

from __future__ import print_function

import cv2
import numpy as np
import torch

from pyaer import libcaer
from pyaer.davis import DAVIS

import torch
from torchvision import transforms, utils

class DAVIS346():
    def __init__(self, shape=(256,256), data_split=10, dt=1, packet_interval=1000, min_activity=100):
        self.davis346 = DAVIS(noise_filter=True)
        self.data_split = data_split
        self.dt = dt
        self.half_point = (self.data_split*self.dt)//2
        self.shape = shape
        self.channels = 4
        self.xoff = (self.davis346.dvs_size_X - self.shape[0])//2
        self.yoff = (self.davis346.dvs_size_Y - self.shape[1])//2
        self.packet_interval = packet_interval
        self.min_activity = min_activity
        
        self.former_events_on = torch.zeros([self.half_point, self.shape[1], self.shape[0]], dtype=torch.float32)
        self.former_events_off = torch.zeros([self.half_point, self.shape[1], self.shape[0]], dtype=torch.float32)
        self.latter_events_on = torch.zeros([self.half_point, self.shape[1], self.shape[0]], dtype=torch.float32)
        self.latter_events_off = torch.zeros([self.half_point, self.shape[1], self.shape[0]], dtype=torch.float32)
        
        self.spike_img_np = np.full((self.shape[1], self.shape[0]), 0, dtype=np.uint8)
        
        self.spike_img = np.full((2, self.shape[1], self.shape[0]), 0, dtype=np.uint8)
        # self.inp_buffer = np.full(((self.data_split*self.dt), 2, self.shape[1], self.shape[1]), 0, dtype=np.uint8)
        self.data_buffer = torch.zeros([self.channels, self.shape[1], self.shape[0], self.half_point], dtype=torch.float32)
#        torch.from_numpy(np.full((self.timesteps, self.shape[0], self.shape[1]), 0, dtype=np.uint8)).float()

        self.frames_ts = []

        self.gray_image = np.full((self.davis346.dvs_size_Y, self.davis346.dvs_size_X), 0, dtype=np.uint8)
        
        self.transform = transforms.ToTensor()
        
        self.start_stream()
        self.set_params()
        self.print_params()
        
    def print_params(self): 
        print ("Davis346 ID:", self.davis346.device_id)
        if self.davis346.device_is_master:
            print ("Davis346 is master.")
        else:
            print ("Davis346 is slave.")
        print ("Davis346 Serial Number:", self.davis346.device_serial_number)
        print ("Davis346 String:", self.davis346.device_string)
        print ("Davis346 USB bus Number:", self.davis346.device_usb_bus_number)
        print ("Davis346 USB device address:", self.davis346.device_usb_device_address)
        print ("Davis346 size X:", self.davis346.dvs_size_X)
        print ("Davis346 size Y:", self.davis346.dvs_size_Y)
        print ("Logic Version:", self.davis346.logic_version)
        print ("Background Activity Filter:", self.davis346.dvs_has_background_activity_filter)
        print ("Exposure:",
                   self.davis346.get_config(libcaer.DAVIS_CONFIG_APS,libcaer.DAVIS_CONFIG_APS_EXPOSURE))
        print ("Autoexposure:", 
               self.davis346.get_config(libcaer.DAVIS_CONFIG_APS, libcaer.DAVIS_CONFIG_APS_AUTOEXPOSURE))
        print ("Packet Container Interval: ", 
               self.davis346.get_config(libcaer.CAER_HOST_CONFIG_PACKETS, libcaer.CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL))

    def start_stream(self):
        self.davis346.start_data_stream()
        
    def set_params(self):  
        # setting bias after data stream started
        self.davis346.set_bias_from_json("dvs/davis346_config.json")
        
        #Disable modules
        self.davis346.set_config(libcaer.DAVIS_CONFIG_IMU, libcaer.DAVIS_CONFIG_IMU_RUN_ACCELEROMETER, False);
        self.davis346.set_config(libcaer.DAVIS_CONFIG_IMU, libcaer.DAVIS_CONFIG_IMU_RUN_GYROSCOPE, False);
        self.davis346.set_config(libcaer.DAVIS_CONFIG_IMU, libcaer.DAVIS_CONFIG_IMU_RUN_TEMPERATURE, False);
        
        #APS config
        self.davis346.set_config(libcaer.DAVIS_CONFIG_APS, libcaer.DAVIS_CONFIG_APS_GLOBAL_SHUTTER, False);
        self.davis346.set_config(libcaer.DAVIS_CONFIG_APS, libcaer.DAVIS_CONFIG_APS_AUTOEXPOSURE, False);
        self.davis346.set_config(libcaer.DAVIS_CONFIG_APS, libcaer.DAVIS_CONFIG_APS_EXPOSURE, 4200);
        
        #Output config
        self.davis346.set_config(libcaer.CAER_HOST_CONFIG_PACKETS, libcaer.CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL, self.packet_interval)
        self.davis346.set_config(libcaer.CAER_HOST_CONFIG_DATAEXCHANGE, libcaer.CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, True);
        
        
    def get_packet(self):
        data = self.davis346.get_event()
        return data
    
            
    def push_in_queue1(self):
#        img = torch.from_numpy(img).float()
        # img = self.transform(img)
        
#        self.img_buffer = torch.cat((img_buffer[1:], img.unsqueeze(0))) 

        self.inp_buffer = np.concatenate((self.inp_buffer[1:], np.expand_dims(self.spike_img, axis=0)), axis=0)
        
    def push_in_queue(self):
#        img = torch.from_numpy(img).float()
        img_on= self.transform(self.spike_img[0])
        img_off= self.transform(self.spike_img[1])
 
        
        self.former_events_on = torch.cat((self.former_events_on[1:], self.latter_events_on[0].unsqueeze(0)))
        self.former_events_off = torch.cat((self.former_events_off[1:], self.latter_events_off[0].unsqueeze(0)))
        
        self.latter_events_on = torch.cat((self.latter_events_on[1:], img_on))
        self.latter_events_off = torch.cat((self.latter_events_off[1:], img_off))
        
#        self.img_buffer = torch.cat((img_buffer[1:], img.unsqueeze(0))) 

        # self.inp_buffer = np.concatenate((self.inp_buffer[1:], np.expand_dims(self.spike_img, axis=0)), axis=0)
    
    def get_encoded_buffer1(self):
        buffer = self.inp_buffer.copy()
        
        buffer = buffer.transpose((1,2,3,0))
        
        buffer = torch.from_numpy(buffer/255.0).float()
        
        self.data_buffer[0] = buffer[0,:,:,0:self.half_point]
        self.data_buffer[1] = buffer[1,:,:,0:self.half_point]
        self.data_buffer[2] = buffer[0,:,:,self.half_point:]
        self.data_buffer[3] = buffer[1,:,:,self.half_point:]
        
        return self.data_buffer
    
    def get_encoded_buffer(self):
        self.data_buffer[0] = self.former_events_on.transpose(0,2).transpose(0,1)
        self.data_buffer[1] = self.former_events_off.transpose(0,2).transpose(0,1)
        self.data_buffer[2] = self.latter_events_on.transpose(0,2).transpose(0,1)
        self.data_buffer[3] = self.latter_events_off.transpose(0,2).transpose(0,1)
        
        return self.data_buffer
    
    def get_event_buffer(self):
        data = self.get_packet()
        if data is not None:
            (pol_events, num_pol_event,
             special_events, num_special_event, 
             frames_ts, frames, imu_events,
             num_imu_event) = data
            
            # if len(frames_ts) != 0:
            #     frame_fps = 1000000.0/(frames_ts - self.frames_ts)
            #     self.frames_ts = frames_ts
            #     # print('Frame FPS: ', frame_fps)
            
            # if frames.shape[0] != 0:
            #     self.gray_image = frames[0]
                
            # print(frames_ts)

            self.spike_img = np.full((2, self.shape[1], self.shape[0]), 0, dtype=np.uint8)
            
            spimg_p = np.full((self.davis346.dvs_size_Y, self.davis346.dvs_size_X), 0, dtype=np.uint8)
            spimg_np = np.full((self.davis346.dvs_size_Y, self.davis346.dvs_size_X), 0, dtype=np.uint8)
            
            if num_pol_event > self.min_activity:
                events_p = pol_events[pol_events[:,3] == 1]
                events_np = pol_events[pol_events[:,3] == 0]
                
                spimg_p[events_p[:,2], events_p[:,1]] = 255
                spimg_np[events_np[:,2], events_np[:,1]] = 255
                
                self.spike_img[0] = spimg_p[self.yoff:-self.yoff, self.xoff:-self.xoff]
                self.spike_img[1] = spimg_np[self.yoff:-self.yoff, self.xoff:-self.xoff]
                
            self.push_in_queue()
    
                     
            
    def show_image(self, scale=1):
        img_pnp = self.spike_img[0] + self.spike_img[1]
        img_pnp = cv2.resize(img_pnp, (scale*self.shape[1], scale*self.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('PNP Image', img_pnp)
        
        # cv2.imshow('SpikeImg0', self.inpC_buffer[0].numpy())
        # if len(self.frames_ts) != 0:
        #     img_gray = cv2.resize(self.gray_image, (346*scale, 260*scale))
        #     cv2.imshow('Grayscale Image', img_gray)
        
    def save_buffer(self):
        np.save('data_buffer.npy', np.array(self.data_buffer))
    
#%%
# shape = (256,256)
# data_split = 10
# dt= 1
# spikecam = DAVIS346(shape=shape, data_split=data_split, dt=dt)


    

# while True:
#     try:
#         spikecam.get_event_buffer()
#         spikecam.show_image()
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             spikecam.davis346.shutdown()
#             break
        
#     except KeyboardInterrupt:
#         cv2.destroyAllWindows()
#         spikecam.davis346.shutdown()
#         break
#%%