# ImageRank_reward
Aligning Text-to-Image Models using Human Feedback
## 数据爬取
从https://huggingface.co/datasets/yuvalkirstain/PickaPic-images 下载image信息；

从https://huggingface.co/datasets/yuvalkirstain/PickaPic-rankings 下载rank信息；

（1）先配置git代理

git config --global https.proxy http://127.0.0.1:1080

（2）下载数据

git clone https://huggingface.co/datasets/yuvalkirstain/PickaPic-images.git

git clone https://huggingface.co/datasets/yuvalkirstain/PickaPic-rankings.git

下载的数据为parquet格式，转换为csv格式，具体代码在parquet_to_csv.py

image图片需要通过url下载
![image](https://github.com/zueliye/ImageRank_reward/assets/92658543/2ac26b25-1389-4528-b5f0-da2bdd594f52)

应用selenium模拟点击url下载，具体代码在url_to_img.py

首先需要查看chrome浏览器版本，下载chromedriver插件http://chromedriver.storage.googleapis.com/index.html

把chromedrive.exe文件复制到浏览器的安装目录下：C:\Program Files (x86)\Google\Chrome\Application

配置环境变量:此电脑→右击属性→高级系统设置→环境变量→用户变量→Path→编辑→新建，

C:\Program Files (x86)\Google\Chrome\Application\

## 数据预处理
处理为(image_uid,prompt,label)形式,具体代码在pre_data.py

If image_uid==best_image_uid:label=2;
If best_image_uid==None:label=1;
If image_uid!=best_image_uid:label=0;
![image](https://github.com/zueliye/ImageRank_reward/assets/92658543/3c0e3455-f5c4-4e1e-a356-8adf1b58b6fd)

101243

问题：图片rank数据中含有重复数据，导致数据预处理过程中，产生img-prompt对多个label的情况

### 数据筛选：

（1）去除重复行df.drop_duplicates()；82709条
（2）将数据按照label降序排列；df.sort_values(axis=0, by='label', ascending=False)
（3）去除重复行，保留第一次出现的行，即label值较大的行; 
df=df.drop_duplicates(subset=['image_uid','prompt'],keep='first') 76332条

计数：df['label'].value_counts()

0    37271；1    27734；2    11327

共计76332条。随机划分训练集、测试集8：2。

改为2分类问题

（1）2改为1，去掉1，0不变，df[~df['label'].isin([1])]

（2）2改为1，1改成0

### clip
img-prompt编码模型，具体代码见clip.py

问题：编码速度很慢

保存编码向量，后续优化可直接调用

Dataloader.py加载编码好的向量

## 奖励函数--reward Function
CNN.py , MLP.py, LSTM.py

### 数据增强--数据扰动
产生perturbated_prompts，见prompt-classification.py
## 模型训练及测试
train.py test.py
## 结果
10epoch，batchsize=64

MLP速度最快，提升数据量能有效提升分类结果；CNN速度最慢，在数据量较小时效果较好，但提升数据量效果提升不大；
| Model | Loss | data | Accuracy | r2-score|
| :---: | :---: | :---: | :---: | :---: |
| CNN | CrossEntropy | 76332| 85.16% |    |
| LSTM | CrossEntropy | 76332 | 85.16% |    |
| MLP | MSE | 76332 |  |0.33|
