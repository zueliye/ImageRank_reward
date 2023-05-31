from selenium import webdriver
from time import sleep

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(executable_path="C:/Program Files/Google/Chrome/Application/chromedriver", options=options)


driver.maximize_window()
sleep(3)

import pandas as pd
df = pd.read_csv('Download_data_parquet/train.csv')
l=50768
for i in range(50768,len(df['url'])):
    driver.get(df['url'][i])
    print(l)
    l+=1
    # sleep(1)
driver.close()