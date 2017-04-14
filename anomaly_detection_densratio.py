#coding:utf-8

import numpy as np
import pandas as pd
from densratio import densratio

#ファイルの読み込み
#train_filename = "./pcap_data/rhenium/1-8-06.csv"
#test_filename = "./pcap_data/rhenium/1-10-18.csv"
train_filename = "./pcap_data/inside/week1_wed.csv"
test_filename =  "./pcap_data/inside/week1_thur.csv"


#エントロピー計算
def entropy(data, window_s):
  
  def calc(hensu):
    hensu = hensu.value_counts()
    num = []
    for i in hensu:
      num.append(i/window_s * np.log2(i/window_s))
    goukei = sum(num) * -1
    return goukei

  entro = []
  for x in range(0, len(data), window_s):
    window = data.iloc[x:x+window_s, :]
    hensu = np.array(window.apply(lambda y: calc(y)))
    entro.append(hensu)
  
  return np.array(entro).T


#通信時間の計算
def time_c(time, window_s):
  time = pd.DataFrame(time)
  Time = []
  for x in range(0, len(time), window_s):
    window = np.array(time.iloc[x:x+window_s, :])
    Time.append((window[len(window) - 1] - window[0])/(len(window)))
  return np.array(Time)


#ペイロード長の計算
def length_c(byte, window_s):
  byte = pd.DataFrame(byte)
  byte_length = []
  for x in range(0, len(byte), window_s):
    window = np.array(byte.iloc[x:x+window_s, :])
    byte_length.append(window.mean())
  return np.array(byte_length)

train_data = pd.read_csv(train_filename, sep=',')
test_data = pd.read_csv(test_filename, sep=',')

entro_train = train_data.iloc[:, 3:]
entro_test = test_data.iloc[:, 3:]
#time_train = train_data.iloc[:, 1]
#time_test = test_data.iloc[:, 1]
#length_train = train_data.iloc[:, 2]
#length_test = test_data.iloc[:, 2]

#窓幅の設定
window_size = 30

train_entro = sum(entropy(entro_train, window_size))
test_entro = sum(entropy(entro_test, window_size))

#train_time = time_c(time_train, window_size)
#test_time = time_c(time_test, window_size)

#train_length = length_c(length_train, window_size)
#test_length = length_c(length_test, window_size)

#print("train_entro:{0}".format(train_entro))
#print("test_entro:{0}".format(test_entro))
def anomality(train, test):
  '''
  dens = densratio(train, test)
  print(dens.compute_density_ratio(train))
  result = -1 * np.log(dens.compute_density_ratio(train))
  print(result.shape)
  print(result)
  print(max(result))
  '''
  result = densratio(train, test)
  w_hat = result.compute_density_ratio(train)
  anom_score = -np.log(w_hat)
  anom_percentile = 5
  thresh = np.percentile(anom_score, 100-anom_percentile)
  w_hat_test = result.compute_density_ratio(test)
  test_anom_score = -np.log(w_hat_test)
  print(len(test_anom_score[test_anom_score > thresh])/len(test_anom_score))

print("####----entropy----####")
anomality(train_entro, test_entro)
print("####----time----####")
#anomality(train_time, test_time)
print("####----length----####")
#anomality(train_length, test_length)
'''
for (train, test) in zip(train_entro, test_entro):
  dens = densratio(train, test)
  result = -1 * np.log(dens.compute_density_ratio(train))
  print(result)
'''

