import os
from sklearn.model_selection import train_test_split
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

root = "data/processed"
files = os.listdir(root)

train, test = train_test_split(files, test_size=0.05, random_state=123)
# print(train[0], test[:3])
for f in train:
    os.system(f'cp "data/processed/{f}" data/train/')
for f in test:
    os.system(f'cp "data/processed/{f}" data/test/')