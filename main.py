# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# TensorFlow code to save model weights
# tensorflow_model.save_weights("tensorflow_model_weights.h5")

# Python code to load and transfer weights to a PyTorch model
import torch
import torch.nn as nn

# Define the PyTorch model with the same architecture
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(12 * 6 * 6, 216)
        self.fc2 = nn.Linear(216, 12)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 12 * 6* 6)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load TensorFlow weights
import h5py
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tf_weights = h5py.File("trial_400.h5", "r")

    # Initialize PyTorch model
    pytorch_model = PyTorchModel()
    tmp = tf_weights
    print('f.keys():', tf_weights.keys())
    print('f.keys():', tf_weights["conv2d_3"].keys())
    print("f.attrs['layer_names']:", tf_weights.attrs['layer_names'])
    print("f['block1_conv1'].attrs.keys():", tf_weights['conv2d_3'].attrs.keys())
    print("f['block1_conv1'].attrs['weight_names']:", tf_weights['conv2d_3'].attrs['weight_names'])

    print("------")
    print("f['block1_conv1/block1_conv1_W_1:0']:", tf_weights['conv2d_3']['conv2d_3/kernel:0'])
    print("f['block1_conv1/block1_conv1_W_1:0']:", tf_weights['conv2d_4']['conv2d_4/kernel:0'])

    print("**dense**")
    print("f['block1_conv1/block1_conv1_W_1:0']:", tf_weights['dense_3']['dense_3/kernel:0'])
    print("f['block1_conv1/block1_conv1_W_1:0']:", tf_weights['dense_4']['dense_4/kernel:0'])

    tmp2=tf_weights["conv2d_3"]
    # Transfer weights layer by layer
    pytorch_model.conv1.weight.data = torch.from_numpy(np.array(tf_weights['conv2d_3']["conv2d_3/kernel:0"]).transpose(3,2,1,0))
    pytorch_model.conv1.bias.data = torch.from_numpy(np.array(tf_weights['conv2d_3']["conv2d_3/bias:0"]))
    pytorch_model.conv2.weight.data = torch.from_numpy(np.array(tf_weights["conv2d_4"]["conv2d_4/kernel:0"]).transpose(3,2,1,0))
    pytorch_model.conv2.bias.data = torch.from_numpy(np.array(tf_weights["conv2d_4"]["conv2d_4/bias:0"]))
    pytorch_model.fc1.weight.data = torch.from_numpy(np.array(tf_weights["dense_3"]["dense_3/kernel:0"])).t()  # Transpose for PyTorch
    pytorch_model.fc1.bias.data = torch.from_numpy(np.array(tf_weights["dense_3"]["dense_3/bias:0"]))
    pytorch_model.fc2.weight.data = torch.from_numpy(np.array(tf_weights["dense_4"]["dense_4/kernel:0"])).t()  # Transpose for PyTorch
    pytorch_model.fc2.bias.data = torch.from_numpy(np.array(tf_weights["dense_4"]["dense_4/bias:0"]))

    # Close the TensorFlow weights file
    tf_weights.close()
    torch.save(pytorch_model.state_dict(), "trial_400.pth")
    print("end")

    action_space = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Action Space Structure
        (-1, 1, 0), (0, 1, 0), (1, 1, 0),  # (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)
    ],
    # Initialize PyTorch model
    pytorch_model = PyTorchModel()
    pytorch_model.load_state_dict(torch.load("trial_400.pth"))
    data = np.load("tf_data.npz", allow_pickle=True)
    obs = data['obs']
    acts = data['acts']
    correct = 0
    total = 0
    for i in range(len(obs)):
        total += 1
        img = obs[i].transpose(2, 0, 1)
        tensor = torch.tensor(img).unsqueeze(0).float()
        print(acts[i][2])
        print(acts[i][0])
        out = pytorch_model(tensor)
        _, predicted = torch.max(out.data, 1)
        print(predicted)
        print(action_space[0][predicted])
        if predicted.item() == acts[i][2]:
            correct += 1
        print("------")
    print(correct / total)
    print(correct)
    print(total)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
