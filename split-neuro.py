import os
from sklearn.model_selection import train_test_split
import shutil

pos = os.listdir('../neuro-patches/patches/nft')
neg = os.listdir('../neuro-patches/patches/background')

pos_train, pos_test = train_test_split(pos, test_size=0.33)
print(pos_train)
neg_train, neg_test = train_test_split(neg, test_size=0.33)

for item in neg_train:
    shutil.move(os.path.join('/home/millerj/neuro-patches/patches/background', item), os.path.join('/home/millerj/neuro-patches/patches/train/background', item))
