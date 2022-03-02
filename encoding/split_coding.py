import numpy as np
import os
import h5py
import argparse
from tqdm import trange
from multiprocessing import Pool as ThreadPool

def generate_fimage(input_event=0, gray=0, image_raw_event_inds=0):
    print(image_raw_event_inds.shape, image_raw_ts.shape)

    split_interval = gray_image.shape[0]
    data_split = 10 # N * (number of event frames from each groups)

    td_img_c = np.zeros((2, gray.shape[1], gray.shape[2], data_split), dtype=np.uint8)

    t_index = 0
    for i in trange(split_interval):
        if image_raw_event_inds[i-1] < 0:
            frame_data = input_event[0:image_raw_event_inds[i], :]
        else:
            frame_data = input_event[image_raw_event_inds[i-1]:image_raw_event_inds[i], :]

        if frame_data.size > 0:
            td_img_c.fill(0)

            for m in range(data_split):
                for vv in range(int(frame_data.shape[0]/data_split)):
                    v = int(frame_data.shape[0] / data_split)*m + vv
                    if frame_data[v, 3].item() == -1:
                        td_img_c[1, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1
                    elif frame_data[v, 3].item() == 1:
                        td_img_c[0, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1

        t_index = t_index + 1

        np.save(os.path.join(count_dir, str(i)), td_img_c)
        np.save(os.path.join(gray_dir, str(i)), gray[i,:,:])


def preprocess(raw_events, gray, raw_event_inds):
    frame_data, gray_data, indices = [], [], []
    
    for i in range(gray.shape[0]):
        if raw_event_inds[i-1] < 0:
            frame_data.append(raw_events[0:raw_event_inds[i], :])
        else:
            frame_data.append(raw_events[raw_event_inds[i-1]:raw_event_inds[i], :])

        gray_data.append(gray[i, :, :])
        indices.append(i)

    
    return frame_data, gray_data, indices

def mp_generate_fimage(data):
    frame_data, gray, idx = data
    
    # print('Gray shape:', gray.shape)
    # print('Frame shape:', frame_data.shape)
    # print('Idx:', idx)
    
    data_split = 10
    
    td_img_c = np.zeros((2, gray.shape[0], gray.shape[1], data_split), dtype=np.uint8)
    
    if frame_data.size > 0:
        td_img_c.fill(0)

        for m in range(data_split):
            for vv in range(int(frame_data.shape[0]/data_split)):
                v = int(frame_data.shape[0] / data_split)*m + vv
                if frame_data[v, 3].item() == -1:
                    td_img_c[1, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1
                elif frame_data[v, 3].item() == 1:
                    td_img_c[0, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1
                    
    np.save(os.path.join(count_dir, str(idx)), td_img_c)
    np.save(os.path.join(gray_dir, str(idx)), gray)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spike Encoding')
    parser.add_argument('--save-dir', type=str, default='../datasets', metavar='PARAMS', help='Main Directory to save all encoding results')
    parser.add_argument('--save-env', type=str, default='indoor_flying1', metavar='PARAMS', help='Sub-Directory name to save Environment specific encoding results')
    parser.add_argument('--mp', default=True, help='Use Multiprocessing')
    args = parser.parse_args()
    
    save_path = os.path.join(args.save_dir, args.save_env)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    count_dir = os.path.join(save_path, 'count_data')
    if not os.path.exists(count_dir):
        os.makedirs(count_dir)
      
    gray_dir = os.path.join(save_path, 'gray_data')
    if not os.path.exists(gray_dir):
        os.makedirs(gray_dir)
      
    args.data_path = save_path + '/' + args.save_env + '_data.hdf5'
    
    
    #Print args
    print(' ' * 20 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 20 + k + ': ' + str(v))
    
    d_set = h5py.File(args.data_path, 'r')
    
    raw_data = d_set['davis']['left']['events']
    image_raw_event_inds = d_set['davis']['left']['image_raw_event_inds']
    image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
    gray_image = d_set['davis']['left']['image_raw']
    d_set = None
    
    
    if args.mp:
        # With MP
        print('Preprocessing...')
        frame_data, gray_data, indices = preprocess(raw_data, gray_image, image_raw_event_inds)
        raw_data = None
        print('Done!')
        
        print('Saving encoded files using mutiprocessing...')
        pool = ThreadPool()
        pool.map(mp_generate_fimage, zip(frame_data, gray_data, indices))
        pool.close()
        pool.join()
    else:
        # Without MP
        print('Saving encoded files using single process...')
        generate_fimage(input_event=raw_data, gray=gray_image, image_raw_event_inds=image_raw_event_inds)
        raw_data = None

    print('Encoding complete!')
