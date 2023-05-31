import random
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import pandas as pd
import csv

df = pd.read_csv(r'./Download_data_parquet/train_rank.csv',encoding='utf-8')
df1 = pd.read_csv('pre_train_clean.csv',encoding='utf-8')
df2=pd.DataFrame(columns=['image_uid','perturbated_prompt','label'])
df2.to_csv('perturbated_train.csv',index=False)

def Get_perturbated_prompts():
    prompt_list = []
    for i in range(len(df['prompt'])):
        line = df['prompt'][i]
        if line not in prompt_list:
            prompt_list.append(line)
    print(prompt_list)

    N = len(df1['prompt'])
    output=[]
    count=0
    while (count<N):
        idx = random.randint(0,len(prompt_list)-1)
        text=prompt_list[idx]
        count += 1
        output.append(text)
    # print(len(output))
    return output


output=Get_perturbated_prompts()
with open('perturbated_train.csv', 'a+', encoding='utf-8', newline='') as df2:
    csv_writer = csv.writer(df2)
    for i in range(len(output)):
        csv_writer.writerow([df1['image_uid'][i],output[i],df1['label'][i]])
        print(df1['image_uid'][i], output[i], df1['label'][i])








