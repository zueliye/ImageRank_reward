import open_clip
import torch
from PIL import Image
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import pandas as pd
import numpy as np


def clip(filepath,imgpath):
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14')
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    df = pd.read_csv(filepath,encoding='utf-8')
    input_pt=[]
    label_pt=[]

    for i in range(len(df['image_uid'])):
        try:
            img = Image.open(os.path.join(imgpath,df['image_uid'][i]+'.png')).convert('RGB')
            img = preprocess_train(img).unsqueeze(0).to(device)
            prompt = tokenizer(str(df['prompt'][i]))
            label = np.array(df['label'][i]).astype(np.int64)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = clip_model.encode_image(img)
                text_features = clip_model.encode_text(prompt)
                image_features = image_features.squeeze()
                text_features = text_features.squeeze()
                text_features=[image_features,text_features]
                input = torch.cat(text_features, dim=0).unsqueeze(0)
            input_pt.append(input)
            label = torch.from_numpy(label)
            label_pt.append(label)
            print(i)
            torch.save(input_pt,'input.pt')
            torch.save(label_pt,'label.pt')

        except Exception as e:
            print(e)
            continue





















