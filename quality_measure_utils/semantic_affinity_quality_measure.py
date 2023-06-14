import torch
from PIL import Image
import open_clip
import pandas as pd
import os
import time 
import argparse,pprint



def predict_prob(img_dir,texts, preprocess, model, tokenizer):
    

    image = preprocess(Image.open(img_dir)).unsqueeze(0)
    model  = model.cuda()
    text = tokenizer(texts).cuda()
    image = image.cuda()
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
    return text_probs[0]


def main(config):
    print('----------------------------------------------------------------')
    print('Begin semantic_affinity_quality_measure calculation')
    print('----------------------------------------------------------------')
    texts = config.texts
    total_prob = [[] for _ in range(len(texts))]
    info_path = config.info_path
    data_path = config.data_path
    names = []
    model_dirs = pd.read_csv(info_path)['Image'].to_list()

    print('Begin Initializing.')
    start = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(config.model_name, pretrained=config.pretrained)
    tokenizer = open_clip.get_tokenizer(config.model_name)
    end = time.time()
    print('Initializing costs ' + str(end-start) + 's.')


    for model_dir in model_dirs:
        model_dir = os.path.join(data_path, model_dir + '.obj')
        print(model_dir)
        start = time.time()
        imgs = sorted(os.listdir(model_dir))
        # using all the six projections
        for i in range(len(imgs)):
            img_dir = os.path.join(model_dir, imgs[i])
            #print(img_dir)
            names.append(img_dir)
            prob = predict_prob(img_dir = img_dir, texts=texts, preprocess = preprocess, model = model, tokenizer = tokenizer)
            # record prob
            for j in range(len(texts)):
                total_prob[j].append(prob[j])
        end = time.time()
        print('Inference time costs ' + str(end-start) + 's.')
        
    #saving probs
    final_data = {'img':names}
    for j in range(len(texts)):
        final_data['text'+str(j)] = total_prob[j]
    final_data = pd.DataFrame(final_data)
    final_data.to_csv(config.output_csv,index=None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default = 'H3D')
    parser.add_argument('--model_name', type=str, default = 'ViT-B-32')
    parser.add_argument('--pretrained', type=str, default = 'laion2b_s34b_b79k')
    parser.add_argument('--texts', type=list, default=[
                        "a high quality projection of 3d human model",
                        "a low quality projection of 3d human model",
                        "a good projection of 3d human model",
                        "a bad projection of 3d human model",
                        "a perfect projection of 3d human model",
                        "a distorted projection of 3d human model",
                        ])
    parser.add_argument('--info_path', type=str, default='path_to_dataifo')
    parser.add_argument('--data_path', type=str, default='path_to_projections')
    parser.add_argument('--output_csv', type=str, default='path_to_output_csv')
    


    config = parser.parse_args()
    pprint.pprint(config.__dict__)
    main(config)
