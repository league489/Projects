import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.classification import MulticlassAccuracy

# #Data import
# train_data = torchvision.datasets.MNIST(
#     root='./data',          
#     train=True,             
#     download=True,          
#     transform=torchvision.transforms.ToTensor()  
# )

# test_data = torchvision.datasets.MNIST(
#     root='./data',
#     train=False,
#     download=True,
#     transform=torchvision.transforms.ToTensor()
# )

# # #Select first image from train dataset
# # img = train_data[0][0]
# # #print(img.shape)

# # #Switch dimensions
# # image_np = img.permute(1, 2, 0).numpy()

# # # Display the image
# # plt.imshow(image_np)
# # plt.axis("off")
# # plt.show()

# #Prepare dataloaders for train and test datasets
# train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


#Define CNN class
class SimpleCNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Convolutional part
        self.convolutional_1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.activation_1 = torch.nn.ReLU()
        self.pooling_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.convolutional_2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.activation_2 = torch.nn.ReLU()
        self.pooling_2 = torch.nn.MaxPool2d(kernel_size=2)
        #Classificational part
        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(in_features=3136,out_features=128)
        self.activation_3 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.25)
        self.linear_2 = torch.nn.Linear(in_features=128,out_features=10)
    #forward pass   
    def forward(self, x):
        x = self.convolutional_1(x)
        x = self.activation_1(x)
        x = self.pooling_1(x)
        x = self.convolutional_2(x)
        x = self.activation_2(x)
        x = self.pooling_2(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.activation_3(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x    

# # #Create new network
# model = SimpleCNN()
# # #print(model)

# #Single batch
# images,labels= next(iter(train_dataloader))
# print(images)
# outputs = model(images)
# print(outputs)

# #Move model to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# #Choose loss function
# criterion = torch.nn.CrossEntropyLoss()

# #Choose optimizer
# optimizer = torch.optim.Adam(model.parameters(),lr= 0.001)

# #Choose accuarcy metric
# accuracy_metric = MulticlassAccuracy(num_classes=10).to(device)

# #Set seed for reproducibility 
# torch.manual_seed(0)

# #Define training loop
# def training_loop(t_model:SimpleCNN,t_train:DataLoader,t_test:DataLoader,t_optim,t_loss,n_epochs:int):
#     train_losses = []
#     test_losses = []
#     train_batch_size = t_train.batch_size
#     test_batch_size = t_test.batch_size
#     for epoch in range(1,n_epochs+1):
#         epoch_train_loss = 0
#         epoch_test_loss = 0
#         t_model.train()
#         for I,L in t_train:
#             I,L = I.to(device),L.to(device)
#             outputs = t_model(I)
#             loss = t_loss(outputs,L)
#             t_optim.zero_grad()
#             loss.backward()
#             t_optim.step()
#             epoch_train_loss += loss
#         avg_epoch_train_loss = (epoch_train_loss/train_batch_size).item()
#         train_losses.append(avg_epoch_train_loss)
#         print(f"Epoch {epoch} training loss: {avg_epoch_train_loss}")
#         t_model.eval()
#         with torch.no_grad():
#             for I,L in t_test:
#                 I,L = I.to(device),L.to(device)
#                 outputs = t_model(I)
#                 loss = t_loss(outputs,L)
#                 epoch_test_loss += loss
#                 accuracy_metric.update(outputs,L)
#             avg_epoch_test_loss = (epoch_test_loss/test_batch_size).item()
#             test_losses.append(avg_epoch_test_loss)
#             total_acc = accuracy_metric.compute()
#             print(f"Epoch {epoch} test loss: {avg_epoch_test_loss}")
#             print(f"Accuaracy at epoch {epoch}: {total_acc}")       
#     epochs = np.arange(1,n_epochs+1)
#     train_losses = np.array(train_losses)
#     test_losses = np.array(test_losses)
#     plt.plot(epochs,train_losses,label ="Train loss")
#     plt.plot(epochs,test_losses,label ="Test loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("The course of the training process")
#     plt.legend()
#     plt.show()

# ##Run training loop
# #training_loop(model,train_dataloader,test_dataloader,optimizer,criterion,20)

# #Training process plot indicates, that optimal epoch number to avoid overfitting is 18
# training_loop(model,train_dataloader,test_dataloader,optimizer,criterion,18)

# #Save trained model
# torch.save(model.state_dict(), "final_model.pth")


