import torch
from PIL import Image
import open_clip
import pandas as pd
import os
import time 
import argparse,pprint
import pyiqa


def main(config):
    print('----------------------------------------------------------------')
    print('Begin spatial_naturalness_quality_measure calculation')
    print('----------------------------------------------------------------')
    
    info_path = config.info_path
    data_path = config.data_path
    names = []
    model_dirs = pd.read_csv(info_path)['Image'].to_list()

    print('Begin Initializing.')
    start = time.time()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # create metric with default setting
    iqa_metric = pyiqa.create_metric(config.method_name, device=device)
    end = time.time()
    print('Initializing costs ' + str(end-start) + 's.')

    niqe = []
    for model_dir in model_dirs:
        model_dir = os.path.join(data_path, model_dir + '.obj')
        print(model_dir)
        start = time.time()
        imgs = sorted(os.listdir(model_dir))
        for i in range(len(imgs)):
            img_dir = os.path.join(model_dir, imgs[i])
            #print(img_dir)
            names.append(img_dir)
            niqe.append(iqa_metric(img_dir).cpu().numpy())
        end = time.time()
        print('Inference time costs ' + str(end-start) + 's.')
        
    final_data = {'img':names,'niqe':niqe}
    final_data = pd.DataFrame(final_data)
    final_data.to_csv(config.output_csv,index=None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default = 'H3D')
    parser.add_argument('--info_path', type=str, default='path_to_datainfo')
    parser.add_argument('--data_path', type=str, default='path_to_projections')
    parser.add_argument('--method_name', type=str, default='niqe')
    parser.add_argument('--output_csv', type=str, default='path_to_output_csv')
    

    config = parser.parse_args()
    pprint.pprint(config.__dict__)
    main(config)
