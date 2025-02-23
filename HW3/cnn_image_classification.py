# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image  # 用於處理影像
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm # 進度條顯示工具
import time  # 計時用
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
# start_time = time.time()  # 記錄開始時間

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

# 訓練集 Data Augmentation
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),  # 調整圖片大小為 128x128
    transforms.RandomHorizontalFlip(),  # 以 50% 機率進行水平翻轉
    transforms.RandomRotation(20),  # 隨機旋轉 ±20 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 調整亮度與對比度
    transforms.ToTensor(),  # 轉換成 Tensor 格式
])

# 測試集、驗證集 (不用 Data Augmentation)
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 設定 batch size
batch_size = 128  # 較大的 batch size 可以使梯度更穩定，但需注意 GPU 記憶體限制

# the path where checkpoint saved
model_path = './model.ckpt'

# 定義影像讀取函數（避免使用 lambda，因為 lambda 無法被 pickle）
def pil_loader(path):
    return Image.open(path)


# 定義 CNN 分類器（Classifier）
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # input image size: [3, 128, 128]
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn_layers = nn.Sequential(
            # nn.Conv2d(3, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),

            # nn.Conv2d(64, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),

            # nn.Conv2d(128, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(4, 4, 0),
            
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64] 
           
            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
          
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )


        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            # nn.Linear(256 * 8 * 8, 256),  # 壓平後變成 (256 * 8 * 8) → 256 個神經元
            # nn.ReLU(),
            # nn.Linear(256, 256),          # 256 → 256
            # nn.ReLU(),
            # nn.Linear(256, 11)            # 256 → 11（有 11 個類別）

            nn.Linear(512*4*4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 11)

        )


    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        x = self.cnn_layers(x) # Extract features by convolutional layers.
        x = x.flatten(1) # 將convolutional layers輸出的feature map flatten 成一維
        x = self.fc_layers(x) # fully-connected layers 將 features 轉換為分類的 logits
        return x
class PseudoDataset(Dataset):
    def __init__(self, unlabeled_set, indices, pseudo_labels):
        self.data = Subset(unlabeled_set, indices)
        self.target = torch.LongTensor(pseudo_labels)[indices]

    def __getitem__(self, index):
        
        if index < 0 : #Handle negative indices
            index += len(self)
        if index >= len(self):
            raise IndexError("index %d is out of bounds for axis 0 with size %d"%(index, len(self)))
            
        x = self.data[index][0]
        y = self.target[index].item()
        return x, y

    def __len__(self):
        
        return len(self.data)

def get_pseudo_labels(dataset, model, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    print('get pseudo labels...')
    total_unlabeled = len(dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    masks = []
    pseudo_labels = []
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    # Iterate over the dataset by batches.
    for batch in tqdm(dataloader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits).cpu()

        # Filter the data and construct a new dataset.
        preds = torch.max(probs, 1)[1]
        mask = torch.max(probs, 1)[0] > threshold
        masks.append(mask)
        pseudo_labels.append(preds)

    masks = torch.cat(masks, dim=0).cpu().numpy()
    pseudo_labels = torch.cat(pseudo_labels, dim=0).cpu().numpy()
    indices = torch.arange(0, total_unlabeled)[masks]
    dataset = PseudoDataset(unlabeled_set, indices, pseudo_labels)
    print('using {0:.2f}% unlabeld data'.format(100 * len(dataset) / total_unlabeled))
    # # Turn off the eval mode.
    model.train()
    return dataset
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    start_time = time.time()  # 記錄開始時間
    # CUDA or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'DEVICE: {device}')

    # Construct datasets，DatasetFolder 會自動從資料夾中讀取圖片
    train_set = DatasetFolder("food-11/training/labeled", loader=pil_loader, extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder("food-11/validation", loader=pil_loader, extensions="jpg", transform=test_tfm)
    unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=pil_loader, extensions="jpg", transform=train_tfm)
    test_set = DatasetFolder("food-11/testing", loader=pil_loader, extensions="jpg", transform=test_tfm)
    
    # Construct data loaders.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Initialize a model
    model = Classifier().to(device)
    model.device = device

    # Initialize loss finction（cross-entropy）
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5) # you may fine-tune some hyperparameters such as learning rate on your own.

    # The number of training epochs
    n_epochs = 80

    # Whether to do semi-supervised learning.
    do_semi = False

    # 初始化 Loss 和 Accuracy 記錄列表
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    # 訓練迴圈
    best_acc = 0.0
    for epoch in range(n_epochs):
        # ---------- Semi-Supervised Learning ----------
        if do_semi:
            # Obtain pseudo-labels for unlabeled data using trained model.
            pseudo_set = get_pseudo_labels(unlabeled_set, model)
            
            # 只有當 pseudo_set 不為 None 時才執行半監督學習
            if pseudo_set is not None:
                # Construct a new dataset and a data loader for training.
                concat_dataset = ConcatDataset([train_set, pseudo_set])
                train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
            else:
                print(" No pseudo-labels generated, using only labeled data for training.")
        # ---------- Training ----------
        model.train()

        train_loss = []
        train_accs = []

        # 逐batch處理training set
        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(device)) # Forward the data
            loss = criterion(logits, labels.to(device)) # 計算 cross-entropy loss  # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            optimizer.zero_grad() # 清除先前累積的 gradient
            loss.backward() # Compute the gradients for parameters.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) # Clip the gradient norms for stable training. 
            optimizer.step() # Update 參數

            # 計算目前batch的accuracy
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # 紀錄  loss 和 accuracy
            train_loss.append(loss.item())
            train_accs.append(acc.item())

        # 計算目前 epoch 的平均loss與accuracy
        train_losses.append(sum(train_loss) / len(train_loss))
        train_accuracies.append(sum(train_accs) / len(train_accs))

        # 顯示訓練結果
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_losses[-1]:.5f}, acc = {train_accuracies[-1]:.5f}")

        # ---------- Validation ----------
        model.eval()

        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch

            # validation不用算gradient
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device)) # 計算 loss

            # 計算目前batch的accuracy
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # 紀錄  loss 和 accuracy
            valid_loss.append(loss.item())
            valid_accs.append(acc.item())

        # 計算 validation set 的平均 loss 與 accuracy
        valid_losses.append(sum(valid_loss) / len(valid_loss))
        valid_accuracies.append(sum(valid_accs) / len(valid_accs))

        # 顯示驗證結果
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_losses[-1]:.5f}, acc = {valid_accuracies[-1]:.5f}")

        if sum(valid_accs) > best_acc:
            best_acc = sum(valid_accs)
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc/len(valid_accs)))
    
    # create model and load weights from checkpoint
    model = Classifier().to(device)
    model.load_state_dict(torch.load(model_path))

    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []

    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):
        # A batch consists of image data and corresponding labels.
        # But here the variable "labels" is useless since we do not have the ground-truth.
        # If printing out the labels, you will find that it is always 0.
        # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
        # so we have to create fake labels to make it work normally.
        imgs, labels = batch

        # We don't need gradient in testing, and we don't even have labels to compute loss.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    # 儲存預測結果
    with open("predict.csv", "w") as f:
        f.write("Id,Category\n")
        for i, pred in  enumerate(predictions):
            f.write(f"{i},{pred}\n")

    # 繪製學習曲線
    plot_learning_curve(train_losses, valid_losses, n_epochs)
    plot_accuracy_curve(train_accuracies, valid_accuracies, n_epochs)

    # 計算程式執行時間
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程式執行時間: {elapsed_time:.2f} 秒")