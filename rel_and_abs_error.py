import math
import pandas as pd
import os
import csv
import embeddings as embeddings
import glob
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torch import optim, from_numpy
from torch.utils.data import DataLoader

number_of_samples = 30






class MLP(torch.nn.Module):
    def __init__(self, size_h_layers, activations):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(size_h_layers) - 1):
            self.layers.append(torch.nn.Linear(size_h_layers[i], size_h_layers[i + 1]))
            self.layers.append(activations[i])

    def forward(self, myInput):
        x = myInput
        for layer in self.layers:
            # print(layer)
            x = layer(x)
        return x


# our model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP([27, 20, 10, 5, 1], [nn.CELU(), nn.CELU(), nn.CELU(), nn.Identity()])

print(model)
sum([param.nelement() for param in model.parameters()])

# optimizer = optim.SGD(model.parameters(), lr=1e-2)
optimizer = optim.Adam(model.parameters(), lr=1e-2)


checkpoint = torch.load("./best_model_300k_data_1200_epoch_CELU_x3)_Identity.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()





#path_testing = f'Same_circle_overlap_curvature_h1_radius_{radius_value}_{number_of_samples}_data.csv'

class myData(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, path):
        self.xy = np.loadtxt(path,
                             delimiter=',', dtype=np.float32)
        self.len = self.xy.shape[0]
        self.input_data = from_numpy(self.xy[:, 0:-1])
        print("input_data shape: ", self.input_data.shape)
        self.target_data = from_numpy(self.xy[:, -1:])
        print("target_data shape : ", self.target_data.shape)

    def __getitem__(self, index):
        # print(self.x_data[index], self.y_data[index])
        return self.input_data[index], self.target_data[index]

    def __len__(self):
        return self.len
#path_testing = f'Same_circle_overlap_curvature_h1_radius_{radius_value}_{number_of_samples}_data.csv'

class myData(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, path):
        self.xy = np.loadtxt(path,
                             delimiter=',', dtype=np.float32)
        self.len = self.xy.shape[0]
        self.input_data = from_numpy(self.xy[:, 0:-1])
        print("input_data shape: ", self.input_data.shape)
        self.target_data = from_numpy(self.xy[:, -1:])
        print("target_data shape : ", self.target_data.shape)

    def __getitem__(self, index):
        # print(self.x_data[index], self.y_data[index])
        return self.input_data[index], self.target_data[index]

    def __len__(self):
        return self.len

all_median_rel_error_values = []
all_median_absol_error_values = []
for i in range(4, 42, 2):
    i = i
    radius_value = i

    BATCH_SIZE = 2


    path_testing = f'./Same_circle_data_ordered/Same_circle_overlap_curvature_h1_radius_{radius_value}_{number_of_samples}_data.csv'

    test_dataset = myData(path_testing) #path_testing
    testing_dataloader = DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)



    relative_error = [] # |target - prediction| / target
    absolute_error = [] # |target - prediction|
    #all_median_error_values = []

    for i, data in enumerate(testing_dataloader, 0):  # train_validation

        (inputs, targets) = data

        with torch.no_grad():
            output_test = model(inputs)  # <1>
            #loss_fn = nn.MSELoss()
            #output_test = loss_fn(output_test, targets)
            output_test = output_test.detach().numpy()
        #print(targets, output_test, i)
        relative_error.append(np.array(abs(targets - output_test) / targets)[0, 0])
        absolute_error.append(np.array(abs(targets - output_test)))
        #print("shape ",np.array(abs(targets-output_test)/targets)[0,0])
        # PLOT : y_axis_min_validation
    print("this is the error list: ", relative_error)
    sum_error = sum(relative_error)
    print("this is the sum error 1: ", sum_error)
    median_error = sum_error / number_of_samples
    print(f"this is the median error 1 for radius {radius_value}: ", median_error)
    all_median_rel_error_values.append(median_error)
    print("this is the sum error 2: ", sum_error)
    print("this is the All median error values: ", all_median_rel_error_values)
    with open(f"./Same_circle_data_ordered/rel_error_for_radius{radius_value}, {number_of_samples} number of samples.csv", 'w') as f_output:
        f_output.write(str(median_error))

    rad = np.arange(i+1)*360/(i+1)
    print("The shape is: ",rad.shape)

    print("(relative_error.shape is ",np.array(relative_error).shape)

    #plt.plot(rad, error)
    #plt.show()

    '''
    fig = plt.figure(dpi=300)
    plt.xlabel(r"Rad $\left[\frac{index}{nr samples} 360\right]$")
    plt.ylabel(r'Loss $\left[\frac{|target - predicted|}{target}\right]$')
    #plt.yscale("log")
    plt.plot(rad, relative_error, '-', c='r', label="test (using best model)", linewidth=0.7)
    #plt.plot(plot_x, total_loss_training_epoch, '-', c='r', label="training", linewidth=0.7)
    #plt.plot(x_axis_epoch, y_axis_min_validation, '3', c='y', label="model minimum" "\n" "validation value epoch")
    plt.legend()
    plt.title(f"Curvature relative error- {number_of_samples} samples, for radius = {radius_value}", fontsize=10)
    plt.savefig(f'./Loss_testing_plots_best_model/Curvature_relative_error_-{number_of_samples} samples, radius = {radius_value}.pdf')
    plt.show()
    plt.close("all")
'''
    #-----------------------

    #For Absolute error
    print("this is the error list: ", absolute_error)
    sum_absolute_error = sum(absolute_error)
    print("this is the sum error 1: ", sum_absolute_error)
    median_absolute_error = sum_absolute_error / number_of_samples
    print(f"this is the median error 1 for radius {radius_value}: ", median_absolute_error)
    all_median_absol_error_values.append(median_absolute_error)
    print("this is the sum error 2: ", sum_absolute_error)
    print("this is the All median error values: ", all_median_absol_error_values)
    with open(f"./Same_circle_data_ordered/error_for_radius{radius_value}, {number_of_samples} number of samples.csv", 'w') as f_output:
        f_output.write(str(median_absolute_error))

    rad = np.arange(i+1)*360/(i+1)

    print(absolute_error)

    absolute_error = np.array(absolute_error)
    absolute_error = absolute_error.reshape(-1)

    '''

    fig = plt.figure(dpi=300)
    plt.xlabel(r"Rad $\left[\frac{index}{nr samples} 360\right]$")
    plt.ylabel(r'Loss: |target - predicted|' )
    #plt.yscale("log")
    plt.plot(rad, absolute_error, '-', c='r', label="test (using best model)", linewidth=0.7)
    plt.legend()
    plt.title(f"Curvature absolute error- {number_of_samples} samples, for radius = {radius_value}", fontsize=10)
    plt.savefig(f'./Loss_testing_plots_best_model/Curvature_relative_error_-{number_of_samples} samples, radius = {radius_value}.pdf')
    plt.show()
    plt.close("all")

'''

    #print("all_median_error_values ", all_median_error_values)




outfile = open("./Same_circle_data_ordered/All_median_error.csv",'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], all_median_rel_error_values))
outfile.close()

print("The len is: ", len(all_median_rel_error_values))


x_axis_median_values = []
for i in range(4, 42, 2):
    x_axis_median_values.append(i)
#x_axis_median_values = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]


length_all_median_error_values = len(all_median_rel_error_values)
print("The length of all_median_error_values is: ", length_all_median_error_values)

sum_all_median_error_values = sum(all_median_rel_error_values)
#print("The sum of all_median_error_values is: ", sum_all_median_error_values)

relative_error_all_median_error_values = sum_all_median_error_values / length_all_median_error_values
print("The relative error is:  ", relative_error_all_median_error_values)

fig = plt.figure(dpi=300)
plt.xlabel(r"Radius")
plt.ylabel(r'$Loss_{avg}$ $\left[\sum_{n}^{i} \frac{Loss_{i}}{n}\right]$')
plt.plot(x_axis_median_values, all_median_rel_error_values, '-', c='r', label="test (using best model)", linewidth=0.7)
#plt.plot(x_axis_median_values, all_median_error_values, '-', c='r', label="test (using best model)", linewidth=0.7)
plt.legend()
plt.title(f"Curvature - Average Loss from relative error, {number_of_samples} samples, with relative error: {relative_error_all_median_error_values.item():.4f}, MLP( [nn.CELU() x3, nn.Identity()])", fontsize=7)
plt.savefig(f'./Loss_testing_plots_best_model/Average_Loss_Relative_error_over_Radius_for_{number_of_samples}_samples,_CELU_x3_Idenity_end_not_normalized data.pdf')
plt.show()
plt.close()


all_median_absol_error_values = np.array(all_median_absol_error_values)
all_median_absol_error_values = all_median_absol_error_values.reshape(-1)

fig = plt.figure(dpi=300)
plt.xlabel(r"Radius")
plt.ylabel(r'$Loss_{avg}$ $\left[\sum_{n}^{i} \frac{Loss_{i}}{n}\right]$')
plt.plot(x_axis_median_values, all_median_absol_error_values, '-', c='r', label="test (using best model)", linewidth=0.7)
plt.legend()
plt.title(f"Curvature - Average Loss from absolute error, {number_of_samples} samples, with relative error: {relative_error_all_median_error_values.item():.4f}, MLP( [nn.CELU() x3, nn.Identity()])", fontsize=7)
plt.savefig(f'./Loss_testing_plots_best_model/Average_Loss_Absolute_error_over_Radius_for_{number_of_samples}_samples,_CELU_x3_Idenity_end_not_normalized data.pdf')
plt.show()
plt.close()



