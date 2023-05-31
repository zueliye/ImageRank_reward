import pandas as pd
import shutil
import torch
from torch.utils.data import Dataset,DataLoader
df2=pd.read_csv('Download_data_parquet/train_rank.csv', encoding='utf-8')

import csv
# df=pd.DataFrame(columns=['image_uid','prompt','label'])
# df.to_csv('train_clean.csv',index=False)
# i=1
def train_clean(row):
    with open('train_clean.csv','a+',encoding='utf-8',newline='')as df:
        csv_writer = csv.writer(df)
        for j in range(len(df2['prompt'])):
            if df2['best_image_uid'][j]==df2[row][j]:
                csv_writer.writerow([df2[row][j],df2['prompt'][j],2])
            if df2['best_image_uid'][j]!=df2[row][j]:
                if df2['best_image_uid'][j] == 'none':
                    csv_writer.writerow([df2[row][j], df2['prompt'][j], 1])
                    print([df2[row][j], df2['prompt'][j], 0])
                else:
                    csv_writer.writerow([df2[row][j], df2['prompt'][j], 0])

# row='image_1_uid'
# train_clean(row)
# row='image_2_uid'
# train_clean(row)
# row='image_3_uid'
# train_clean(row)
# row='image_4_uid'
# train_clean(row)

#去除重复行
# df=pd.read_csv('train_clean.csv')
# df=df.drop_duplicates()
# df.to_csv('train_clean_clean.csv',index=False)
# print(df)

# df=pd.read_csv('train_clean_clean.csv')
#
# df_loc = df.sort_values(axis=0, by='label', ascending=False)
# print(df_loc)
# df_loc.to_csv('train_clean_clean.csv',index=False)
# df=pd.read_csv('pre_train_clean.csv')
# df=df.drop('1',axis=0)
# df=df[~df['label'].isin([1])]
# df.to_csv('pre_train_clean_del1.csv',index=False)
# print(df)
#
df=pd.read_csv('pre_train_clean_del1.csv')
print(df['label'].value_counts())


