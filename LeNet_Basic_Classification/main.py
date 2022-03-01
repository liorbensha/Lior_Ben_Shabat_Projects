# Importing Libraries
from time import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:', device)

# Defining Datasets and Dataloaders
dataset_path = "C:\\Users\\Lior Ben Shabat\\Documents\\GitHub\\Lior_Ben_Shabat_Projects" \
               "\\LeNet_Basic_Classification\\Data\\Images"
transform = transforms.Compose([transforms.Resize(300), transforms.CenterCrop(256),
                               transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(dataset_path, transform)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (21000, 5179))
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16,
                                               shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16,
                                              shuffle=False, num_workers=4)

# Plot Samples from the data
figsize = (16, 16)


def plot_samples(dataloader, title='Images'):
    sample_data = next(iter(dataloader))[0].to(device)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title)
    plt.imshow(np.transpose(torchvision.utils.make_grid(sample_data, padding=2, normalize=True)
                            .cpu(), (1, 2, 0)))


def plot_class(dataloader, mclass, title='Images', num=64):
    ret = []

    for data in dataloader.dataset:
        if data[1] == mclass:
            ret.append(data[0])

            if len(ret) == num:
                break

    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title)
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        ret, padding=2, normalize=True
    ).cpu(), (1, 2, 0)))


# Defining LeNet model architecture
class LeNetModel(nn.Module):
    def __init__(self):
        super(LeNetModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features=186050, out_features=2000),
                                        nn.ReLU(), nn.Linear(in_features=2000, out_features=10),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


# Defining loss and optimizer
lr = 0.0001
model = LeNetModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# Train Procedure
def Train(epoch, print_every=50):
    total_loss = 0
    start_time = time()

    accuracy = []

    for i, batch in enumerate(train_dataloader, 1):
        minput = batch[0].to(device)
        target = batch[1].to(device)

        moutput = model(minput)

        loss = criterion(moutput, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        argmax = moutput.argmax(dim=1)
        accuracy.append(
            (target == argmax).sum().item() / target.shape[0])

        if i % print_every == 0:
            print('Epoch: [{}]/({}/{}), Train Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
                epoch, i, len(train_dataloader), loss.item(), sum(accuracy) / len(accuracy), time() - start_time
            ))

    return total_loss / len(train_dataloader)

# Test Procedure
def Test(epoch):
    total_loss = 0
    start_time = time()

    accuracy = []

    for i, batch in enumerate(train_dataloader, 1):
        minput = batch[0].to(device)
        target = batch[1].to(device)
        moutput = model(minput)

        with torch.no_grad():
            loss = criterion(moutput, target)
            total_loss += loss.item()

            argmax = moutput.argmax(dim=1)
            accuracy.append(
                (target == argmax).sum().item() / target.shape[0])

    print('Epoch: [{}], Test Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(
        epoch, total_loss / len(test_dataloader), sum(accuracy) / len(accuracy), time() - start_time))

    return total_loss / len(test_dataloader)


# main loop
if __name__ == "__main__":
    # plot_samples(train_dataloader)
    # model = LeNetModel().to(device)
    # summary(model, (3, 256, 256))
    Test(0)

    train_loss = []
    test_loss = []

    for epoch in range(1, 51):
        train_loss.append(Train(epoch, 200))
        test_loss.append(Test(epoch))

        print('\n')

    # Plot train-test loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss)+1), train_loss, 'g', label='Training Loss')
    plt.plot(range(1, len(test_loss)+1), test_loss, 'b', label='Testing Loss')

    plt.title('Training and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
