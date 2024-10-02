import torch 
import nn 
import torch.nn as nn 
import torch.optim as optim 
# Define the Model 

class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__() 
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x): 
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

model = HousePriceModel(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Training the model 

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype = torch.float32).unsqueeze(1)

for epoch in range(500):
    model.train() 
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step() 

    if epoch%100 == 0: 
        print(f'Epoch {epoch}, Loss: {loss.item()}')



