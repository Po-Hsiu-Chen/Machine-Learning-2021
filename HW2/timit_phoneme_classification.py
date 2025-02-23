import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# 讀取數據集
data_root = './timit_11/'
print('Loading data ...')
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

# 顯示資料集的大小
print(f'Size of training data: {train.shape}')
print(f'Size of testing data: {test.shape}')

# 自定義 Dataset 類別
class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float() # 將資料轉為浮點數型態的張量
        if y is not None:
            y = y.astype(int)  # 確保標籤是整數型態
            self.label = torch.LongTensor(y)  # 將標籤轉為 LongTensor 類型
        else:
            self.label = None  # 若沒有標籤則為 None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx] # 回傳資料和對應的標籤
        else:
            return self.data[idx] # 若無標籤則只回傳資料

    def __len__(self):
        return len(self.data) # 回傳資料集大小

def plot_learning_curve(train_losses, valid_losses, n_epochs):
    plt.figure(figsize=(8, 6))
    plt.plot(range(n_epochs), train_losses, c='tab:red', label='Train Loss')
    plt.plot(range(n_epochs), valid_losses, c='tab:cyan', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_curve(train_accuracies, valid_accuracies, n_epochs):
    plt.figure(figsize=(8, 6))
    plt.plot(range(n_epochs), train_accuracies, c='tab:red', label='Train Accuracy')
    plt.plot(range(n_epochs), valid_accuracies, c='tab:cyan', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def sample(data, stride):
    base = data.copy()
    print("Original data shape:", data.shape)
    
    for step in range(1, stride+1):
        Rshift = np.roll(base, step, axis=0)
        data = np.concatenate((Rshift, data), axis=1)
        print(f"After right shift {step}: {data.shape}")
    
    for step in range(-1, -stride-1, -1):
        Lshift = np.roll(base, step, axis=0)
        data = np.concatenate((data, Lshift), axis=1)
        print(f"After left shift {step}: {data.shape}")

    data = np.reshape(data, (-1, 2*stride+1, 39))
    print("Final reshaped data:", data.shape)
    return data


stride = 2
# (,429)split to(11,39)
train = np.reshape(train, (-1,11,39))
test = np.reshape(test, (-1,11,39))
print(f'train reshape to(11,39): {train.shape}')
print(f'test reshape to(11,39): {test.shape}')

# pick only the 5th MFCC which is corresponding to label
train = train[:,5,:]
test = test[:,5,:]

# include nearby MFCC (To extend the frame length)
train = sample(train, stride)
test = sample(test, stride)

# flatten to (-1,39*n)
train = np.reshape(train, (-1,(2*stride+1)*39))
test = np.reshape(test, (-1,(2*stride+1)*39))
print(f'train reshape to(-1,39*n): {train.shape}')
print(f'test reshape to(-1,39*n): {test.shape}')

# 設定驗證資料集比例
VAL_RATIO = 0.2
split_idx = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y = train[:split_idx], train_label[:split_idx]
val_x, val_y = train[split_idx:], train_label[split_idx:]
print(f'Size of training set: {train_x.shape}')
print(f'Size of validation set: {val_x.shape}')


BATCH_SIZE = 64
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # 只對訓練資料進行隨機打亂
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False) # 驗證資料不進行打亂

# 釋放不再需要的變數，釋放記憶體
del train, train_label, train_x, train_y, val_x, val_y
gc.collect()


# 定義模型類別
class Classifier(nn.Module):
    def __init__(self):
        # super(Classifier, self).__init__()
        # self.layer1 = nn.Linear(429, 1024)
        # self.layer2 = nn.Linear(1024, 512)
        # self.layer3 = nn.Linear(512, 128)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.out = nn.Linear(128, 39)
        # self.drop = nn.Dropout(0.2)
        # self.act_fn = nn.ReLU()

        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear((2*stride+1)*39, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 39)
        )
 
    def forward(self, x):
        x = self.net(x)
        return x


# 檢查設備（GPU 或 CPU）
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定隨機種子，確保結果可重現
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0)
device = get_device()
print(f'DEVICE: {device}')

# 設定訓練參數
num_epoch = 20              # number of training epoch
learning_rate = 0.0001       # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)# 使用 Adam 優化器

# 記錄損失與準確率
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

# 訓練模型
best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        

        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)
        batch_loss.backward()
        optimizer.step()
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    if len(val_set) > 0:
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += batch_loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        valid_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc / len(train_set))
        valid_accuracies.append(val_acc / len(val_set))

        print(f'[{epoch+1}/{num_epoch}] Train Acc: {train_accuracies[-1]:.6f} Loss: {train_losses[-1]:.6f} | Val Acc: {valid_accuracies[-1]:.6f} Loss: {valid_losses[-1]:.6f}')
        
        # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
plot_learning_curve(train_losses, valid_losses, num_epoch)
plot_accuracy_curve(train_accuracies, valid_accuracies, num_epoch)

# 若沒有驗證集，則儲存最後一個訓練的模型
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')


# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

# 用模型進行預測
predict = []
model.eval() # 設定模型為評估模式
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)
# 將預測結果寫入 CSV 檔案
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))