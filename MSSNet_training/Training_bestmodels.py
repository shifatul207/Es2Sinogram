#%%
import os
import h5py
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

srctype= 'matched' # fs, matched
noise= 'clean'
# noise = 40 # 40,60,80
if noise=='clean':
    loadfilename= 'Ez_s_m'
    noise_str= 'clean'
else:
    noise_str= str(noise)
    loadfilename= 'Ez_s_m' + noise_str

srcfoldername= "DNN_MWI2CTHead_1G_" + srctype 


##___________________set the data path according the the location where the srcfoldername folder exists
data_path= r"pwd\Python codes and models" ## for example pwd is the working directory
##________________________________


datapath_srcfolder = os.path.join(data_path, srcfoldername)
input_data, target_data, test_input, test_target = [], [], [],[]
sample_count = 0
num=750; #prevent class imbalance
for idx, file in enumerate(os.listdir(datapath_srcfolder)):
    if not file.endswith(".mat"):
        continue
    path = os.path.join(datapath_srcfolder, file)
    with h5py.File(path, "r") as f:
        real = f[loadfilename]['real'][:]
        imag = f[loadfilename]['imag'][:]
        sino = f['sinogram_2D'][:]
        mwi = np.concatenate([real, imag], axis=1)
        input_data.append(mwi[:num])
        test_input.append(mwi[num:])
        target_data.append(np.transpose(sino[:num, :, :], (0, 2, 1)))
        test_target.append(np.transpose(sino[num:, :, :], (0, 2, 1)))
        print(input_data[0].shape, target_data[0].shape, test_input[0].shape,test_target[0].shape)
        sample_count += num
        print("Sample size= {}".format(sample_count))

#%%
input_tensor = torch.tensor(np.concatenate(input_data), dtype=torch.float32)
# load statistics
datapath_stat=  os.path.join(datapath_srcfolder, 'stat')
stat_file= os.listdir(datapath_stat)
stat = loadmat( os.path.join(datapath_stat, stat_file[0]))
mean= torch.tensor(stat['mean'])
std= torch.tensor(stat['std'])
scale= torch.tensor(stat['scale'])
delta= torch.tensor(stat['delta'])

# normalizing the features
x_norm = (input_tensor - mean) / std

# normalizing the target
target_tensor = torch.tensor(np.concatenate(target_data), dtype=torch.float32).unsqueeze(1)  # [N, 1, H, W]
target_tensor = target_tensor.permute(0, 1, 2, 3)  # [N, 1, 287, 361]
target_tensor= scale*torch.log10 (target_tensor+delta)

print(input_tensor.shape)
print(target_tensor.shape)

#%% creatingthe dataloaders
dataset = TensorDataset(x_norm, target_tensor)

random_seed = 42
torch.manual_seed(random_seed)
train_size = int(0.8 * len(dataset))
val_size = int(0.2* len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))
train_loader = DataLoader(train_ds, batch_size= 8, shuffle=True) #16
val_loader = DataLoader(val_ds, batch_size=8)

#%% DNN block
class SpatialSpectralBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SSRFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = SpatialSpectralBlock(1, 32)
        self.pool = nn.MaxPool2d(2)
        self.down2 = SpatialSpectralBlock(32, 64)
        self.up = nn.Upsample(scale_factor=2)
        self.upconv = nn.Conv2d(64, 32, 1)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x_up = self.up(x2)
        x_up = self.upconv(x_up)
        x_up = F.interpolate(x_up, size=x1.shape[-2:])
        return self.final(x_up + x1)


class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):  # x: [B, T, D]
        attn_weights = self.attn(x)  # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        attended = x * attn_weights  # [B, T, D]
        return attended


class MWIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True, bidirectional=True)  # Output: 128 dim
        self.attn = Attention(128)
        self.conv1 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)  # BatchNorm after conv1
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)  # BatchNorm after conv2
        self.dense = nn.Linear(132 * 64, 287 * 361)
        self.ssr = SSRFormer()

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, 132, 1)
        x, _ = self.lstm(x)  # (B, 132, 128)
        x = x.transpose(1, 2)  # (B, 128, 132)
        x = F.relu(self.bn1(self.conv1(x)))  # Apply BatchNorm after conv1
        x = F.relu(self.bn2(self.conv2(x)))  # Apply BatchNorm after conv2
        x = x.flatten(start_dim=1)  # (B, 132 * 64)
        x = self.dense(x)  # (B, 287*361)
        x = x.view(-1, 1, 287, 361)  # (B, 1, 287, 361)
        return self.ssr(x)  # Final enhanced sinogram

#%% criterion
model = MWIModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
loss_fn = nn.MSELoss()
early_stopping_patience = 6
best_val_loss = float('inf')
patience_counter = 0

#%% training
bestmodelname= 'best_model_'+ srctype + noise_str + ".pth"

## training the model
NUM_EPOCHS = 200  # define once at the top
epoch_train_loss = []
epoch_val_loss = [];
prev_lr = optimizer.param_groups[0]['lr']
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        # for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred_sino = model(x)
        loss = loss_fn(pred_sino, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    epoch_train_loss.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred_sino = model(x)
            loss = loss_fn(pred_sino, y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    epoch_val_loss.append(val_loss)
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    if epoch > 0 and current_lr != prev_lr:
        print(f"➡️ Learning rate reduced to {current_lr:.6e}")
    prev_lr = current_lr

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), bestmodelname)
        patience_counter = 0
        print('Model updated')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
    print("epoch {} completed, loss= {} ,LR= {}".format(epoch + 1, val_loss, current_lr))

#%% plotting and saving
print(best_val_loss)
plt.figure(figsize=(6, 6))
plt.plot(range(1, len(epoch_train_loss)+1),epoch_train_loss, 'bo', label= "train loss")
plt.plot(range(1, len(epoch_train_loss)+1),epoch_val_loss, 'r', label= "val loss")
plt.yscale('log')
plt.legend()

from scipy.io import savemat
lossdataname= 'loss' + srctype + noise_str + ".mat"
path_savestate = os.path.join(datapath_srcfolder, lossdataname)
savemat(path_savestate, {'epoch_train_loss': epoch_train_loss, 'epoch_val_loss': epoch_val_loss})
print("Loss data saved")


#%% testing over test data
test_tensor = torch.tensor(np.concatenate(test_input), dtype=torch.float32)
test_norm = (test_tensor - mean) / std
test_target_tensor = torch.tensor(np.concatenate(test_target), dtype=torch.float32).unsqueeze(1)  # [N, 1, H, W]
test_target_tensor = test_target_tensor.permute(0, 1, 2, 3)  # [N, 1, 287, 361]
test_target_tensor=  scale*torch.log10 (test_target_tensor+delta)
test_dataset= TensorDataset(test_norm, test_target_tensor)
test_loader = DataLoader(test_dataset, batch_size==1)

#%%
import matplotlib.pyplot as plt
from skimage.transform import iradon

model.eval()
sampleidx = 16  # 10,
samples = list(test_loader)[sampleidx:sampleidx + 1]

with torch.no_grad():
    for i, (x, y) in enumerate(samples):
        x = x.to(device)
        pred_sino = torch.pow(10, model(x).cpu() / scale)
        pred_sino = pred_sino.squeeze().numpy() - delta  # Shape: (287, 361)

        y_back = torch.pow(10, y / scale)
        true_sino = y_back.squeeze().numpy() - delta
        # Inverse Radon transform (reconstruction)
        theta = np.linspace(0., 180., 361, endpoint=False) + 180
        pred_img = iradon(pred_sino, theta=theta, circle=False)  # offset for log images
        true_img = iradon(true_sino, theta=theta, circle=False)  # offset for log images

        # Visualization
        plt.figure(figsize=(16, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(true_sino, cmap="plasma", aspect='auto')
        plt.title("True Sinogram")
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.imshow(np.log10(true_img + 0.1), cmap="gray")
        plt.title("Reconstructed from True Sinogram")
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow(pred_sino, cmap="plasma", aspect='auto')
        plt.title("Predicted Sinogram")
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.imshow(np.log10(pred_img + 0.1), cmap="gray")
        plt.title("Reconstructed from Predicted Sinogram")
        plt.colorbar()

        plt.suptitle(f"Sample {i + 1}")
        plt.tight_layout()
        plt.show()
