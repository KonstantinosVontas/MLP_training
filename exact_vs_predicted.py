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
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torch import optim, from_numpy
from torch.utils.data import DataLoader



number_of_samples = 30
n_epochs = 3000 # change

with open('./not_normalized_data/overlap_curvature_h1_400000_4_to_40_train.csv') as file:
    reader = csv.reader(file)
    training_data = []
    for row in reader:
        training_data.append(row)
file.close()


for i in range(4, 42, 2):
    i = i
    radius_value = i

    BATCH_SIZE = 1



    path_testing = f'./Same_circle_data_ordered/Same_circle_overlap_curvature_h1_radius_{radius_value}_{number_of_samples}_data.csv'

    with open(path_testing) as file:
        reader = csv.reader(file)
        test_data = []
        for row in reader:
            test_data.append(row)
    file.close()



    training_data = np.array(training_data, dtype='float64')
    test_data = np.array(test_data, dtype='float64')

    training_data = training_data[0:400000][:]
    test_data = test_data[0:30][:]

    print("This is the training shape ", training_data.shape)
    print("This is the test shape: ", test_data.shape)

    # Training
    input_training_data_only = training_data[:, :-1]
    target_training_data_only = training_data[:, 27:]
    print("target training data only", target_training_data_only)


    # Testing
    input_test_data_only = test_data[:, :-1]
    target_test_data_only = test_data[:, 27:]



    # print("Target training data: ", target_training_data)


    input_standardscaler = preprocessing.StandardScaler()
    target_standardscaler = preprocessing.StandardScaler()

    # -------------------- INPUT AND TARGET SEPARATED ----------------
    # Training
    input_scaler = input_standardscaler.fit(input_training_data_only)
    target_scaler = target_standardscaler.fit(target_training_data_only)

    print("input_training_data_only mean: ", input_standardscaler.mean_)
    print("target_training_data_only mean: ", target_standardscaler.mean_)

    input_training_data_only_normalized = input_standardscaler.transform(input_training_data_only)
    target_training_data_only_normalized = target_standardscaler.transform(target_training_data_only)

    print("input_training_data_only transformed", input_training_data_only_normalized)
    print("target_training_data_only transformed", target_training_data_only_normalized)

    print("input_training_data_only transformed normalised mean: ", input_training_data_only_normalized.mean(axis=0))
    print("target_training_data_only transformed normalised mean: ", target_training_data_only_normalized.mean(axis=0))

    print("input_training_data_only transformed normalised std: ", input_training_data_only_normalized.std(axis=0))
    print("target_training_data_only transformed normalised std: ", target_training_data_only_normalized.std(axis=0))



    # Testing
    input_test_data_only_normalised = input_scaler.transform(input_test_data_only)
    target_test_data_only_normalised = target_scaler.transform(target_test_data_only)

    print("\ninput_testing normalised mean: ", input_test_data_only_normalised.mean(axis=0))
    print("target_testing mean: ", target_test_data_only_normalised.mean(axis=0))

    print("input_testing normalised std: ", input_test_data_only_normalised.std(axis=0))
    print("target_testing normalised std: ", target_test_data_only_normalised.std(axis=0))


    # -------------------- DATA TOGETHER ----------------

    # with open("training_dataset_scaled.csv", 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(zip(input_training_data_only_normalised, target_training_data_only_normalised))


    np.savetxt("./scaled_data/Error_calc_training_dataset_scaled.csv", np.c_[input_training_data_only_normalized, target_training_data_only_normalized], delimiter=",")
    np.savetxt(f"./scaled_data/Error_calc_test_dataset_scaled_{radius_value}_radius_{number_of_samples}_samples.csv", np.c_[input_test_data_only_normalised, target_test_data_only_normalised], delimiter=",")



# ------------- TRAINING NN





#path = "./scaled_data/training_dataset_scaled.csv"
#path_training = "./scaled_data/Error_calc_training_dataset_scaled.csv"
#path_testing = "./scaled_data/Error_calc_test_dataset_scaled.csv"








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
#model = MLP([27, 20, 10, 5, 1], [nn.CELU(), nn.CELU(), nn.CELU(), nn.Sigmoid()])

print(model)
sum([param.nelement() for param in model.parameters()])

# optimizer = optim.SGD(model.parameters(), lr=1e-2)
optimizer = optim.Adam(model.parameters(), lr=1e-2)


checkpoint = torch.load("./best_model_400k_data_3000_epoch_CELU_x3_Identity.pth") # change
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







all_median_outpout_values = []
output_test_list_all_values = []
targets_all_values =[]

for i in range(4, 42, 2):
    i = i
    radius_value = i

    BATCH_SIZE = 1

    path_testing = f'./scaled_data/Error_calc_test_dataset_scaled_{radius_value}_radius_{number_of_samples}_samples.csv'
    #path_testing = f'./Same_circle_data_ordered/Same_circle_overlap_curvature_h1_radius_{radius_value}_{number_of_samples}_data.csv'

    test_dataset = myData(path_testing) #path_testing
    testing_dataloader = DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)

    output_test_list =[]
    for i, data in enumerate(testing_dataloader, 0):  # train_validation

        (inputs, targets) = data

        with torch.no_grad():
            output_test_normalized = model(inputs)  # <1>
            output_test_normalized = output_test_normalized.detach().numpy()

            targets_denormalized = target_standardscaler.inverse_transform(targets)  # NOW THEY ARE DEORMALIZED ADDED
            output_test_denormalized = target_standardscaler.inverse_transform(output_test_normalized)




        # print(targets, output_test, i)
        output_test_list.append(np.array(output_test_denormalized))
        output_test_list_all_values.append(np.array(output_test_denormalized))
        targets_all_values.append(np.array(targets_denormalized))
        # print("shape ",np.array(abs(targets-output_test)/targets)[0,0])
        # PLOT : y_axis_min_validation
    print(f"this is the target test list for radius {radius_value}: ", targets_denormalized)
    print(f"this is the output test list for radius {radius_value}: ", output_test_list)
    sum_output_test_list = sum(output_test_list)
    #print("this is the sum error 1: ", sum_error)
    median_output_test_list = sum_output_test_list / number_of_samples
    #print(f"this is the median error 1 for radius {radius_value}: ", median_error)
    all_median_outpout_values.append(median_output_test_list)
    #print("this is the sum error 2: ", sum_error)
    #print("this is the All median error values: ", all_median_error_values)
    with open(f"./Same_circle_data_ordered_exact_vs_target/exact_vs_predicted_error_for_radius{radius_value}, {number_of_samples} number of samples.csv",
              'w') as f_output:
        f_output.write(str(output_test_list))

    rad = np.arange(i+1)*360/(i+1)

    print(output_test_list)

    #plt.plot(rad, error_output_test)
    #plt.show()


    #fig = plt.figure(dpi=300)
    #plt.xlabel(r"Rad $\left[\frac{index}{nr samples} 360\right]$")
    #plt.ylabel(r'Loss $\left[\frac{|target - predicted|}{target}\right]$')
    #plt.yscale("log")
    #plt.plot(rad, error_output_test, '-', c='r', label="test (using best model)", linewidth=0.7)
    #plt.plot(plot_x, total_loss_training_epoch, '-', c='r', label="training", linewidth=0.7)
    #plt.plot(x_axis_epoch, y_axis_min_validation, '3', c='y', label="model minimum" "\n" "validation value epoch")
    #plt.legend()
    #plt.title(f"Curvature - {number_of_samples} samples, radius = {radius_value}", fontsize=10)
    #plt.savefig(f'./Exact_loss_testing_plots_best_model/Curvature - {number_of_samples} samples, radius = {radius_value}.pdf')
    #plt.show()
    #plt.close("all")



    print("all_median_error_values ", all_median_outpout_values)

#y_axis_target_data = [0.5, 0.333333, 0.25, 0.2, 0.16666667, 0.142857, 0.125, 0.11111111, 0.1, 0.0909090909, 0.0833333,
#                      0.0769230, 0.07142857, 0.066666667, 0.0625, 0.058823529, 0.0555555556, 0.0526315, 0.05]
#x_axis_median_values = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]



x_axis_median_values = []
for i in range(42):
    if i % 2 == 0 and i > 2:
        x_axis_median_values.append(i)
    #print("x_axis_median_values ", x_axis_median_values)
print("x_axis_median_values list ", x_axis_median_values)



target_test_data = []
for i in range(4, 42, 2):
    i = 2/i
    target_test_data.append(i)
    #print("This is ", i)
print("exact_curvature_value list: ", target_test_data)



all_median_outpout_values = np.array(all_median_outpout_values)
all_median_outpout_values = all_median_outpout_values.reshape(-1)


#Figure of curvature prediction vs exact comparison -
fig = plt.figure(dpi=300)
plt.xlabel(r"Radius")
plt.ylabel(r'Curvature $\left[\frac{2}{Radius}\right]$')
plt.plot(x_axis_median_values, all_median_outpout_values, '-', c='r', label="test prediction (using best model)", linewidth=0.7)
plt.plot(x_axis_median_values, target_test_data, 'o', c='b', label="target test data (using best model)", markersize=2)
plt.legend()
plt.title(f"Curvature- 400k data, {number_of_samples} samples, MLP([CELU() x3, Identity()])", fontsize=6) # CHANGE
plt.savefig(f'./Exact_loss_testing_plots_best_model/Average_Loss_over_Radius_for_{number_of_samples}_samples_{n_epochs}_epochs_400k_data.pdf', dpi=300,  bbox_inches='tight')
plt.savefig(f'./Exact_loss_testing_plots_best_model/Average_Loss_over_Radius_for_{number_of_samples}_samples_{n_epochs}_epochs_400k_data.eps',
            dpi=700, format='eps', bbox_inches='tight') # CHANGE
plt.savefig(f'./Exact_loss_testing_plots_best_model/Average_Loss_over_Radius_for_{number_of_samples}_samples_{n_epochs}_epochs_400k_data.png',
            dpi=1200, format='png', bbox_inches='tight') # CHANGE
plt.show()
plt.close()




#Figure of curvature prediction vs exact comparison - averaged values
fig = plt.figure(dpi=300)
plt.plot(target_test_data, all_median_outpout_values, 'o', c='b', label="test prediction (using best model)", markersize=2)
plt.xlabel(r"Exact")
plt.ylabel(r"Prediction")
axes = plt.gca()
m, b = np.polyfit(target_test_data, all_median_outpout_values, 1)# Add correlation line
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r', linewidth = 0.7)
plt.legend()
plt.title(f"Curvature - 400k data (samples averaged per tested radius), for {number_of_samples} samples, MLP([CELU() x3, Identity()])", fontsize=6) # CHANGE
plt.savefig(f'./Exact_loss_testing_plots_best_model/Averaged_values_per_radius_predicted_vs_exact_for_{number_of_samples}_samples_all_values_{n_epochs}_epochs_400k_data.pdf', dpi=300,  bbox_inches='tight')
plt.savefig(f'./Exact_loss_testing_plots_best_model/Averaged_values_per_radius_predicted_vs_exact_for_{number_of_samples}_samples_averaged_{n_epochs}_epochs_400k_data.eps',
            dpi=700, format='eps', bbox_inches='tight') # CHANGE
plt.savefig(f'./Exact_loss_testing_plots_best_model/Averaged_values_per_radius_predicted_vs_exact_for_{number_of_samples}_samples_averaged_{n_epochs}_epochs_400k_data.png',
            dpi=1200, format='png', bbox_inches='tight') # CHANGE
plt.show()







targets_all_values = np.array(targets_all_values)
targets_all_values = targets_all_values.reshape(-1)


output_test_list_all_values = np.array(output_test_list_all_values)
output_test_list_all_values = output_test_list_all_values.reshape(-1)

print("output_test_list_all_values ALL: ", output_test_list_all_values)
print("output_test_list_all_values LENGTH: ",len(output_test_list_all_values))

print("targets_all_values ALL: ", targets_all_values)
print("targets_all_values LENGTH: ",len(targets_all_values))






#Figure of curvature prediction vs exact comparison - actual values
fig = plt.figure(dpi=300)
plt.plot(targets_all_values, output_test_list_all_values, 'o', c='b', label="test prediction (using best model)", markersize=2)
plt.xlabel(r"Exact")
plt.ylabel(r"Prediction")
axes = plt.gca()
m, b = np.polyfit(targets_all_values, output_test_list_all_values, 1)# Add correlation line
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-', c='r', linewidth = 0.7)
plt.legend()
plt.title(f"Curvature - 400k data, {number_of_samples} samples for each examined radius, MLP([CELU() x3, Identity()])", fontsize=6) # CHANGE
plt.savefig(f'./Exact_loss_testing_plots_best_model/Actual_values_per_radius_predicted_vs_exact_for_{number_of_samples}_samples_all_values_{n_epochs}_epochs_400k_data.pdf', dpi=300,  bbox_inches='tight')
plt.savefig(f'./Exact_loss_testing_plots_best_model/Actual_values_per_radius_predicted_vs_exact_for_{number_of_samples}_samples_all_values_{n_epochs}_epochs_400k_data.eps',
            dpi=700, format='eps', bbox_inches='tight') # CHANGE
plt.savefig(f'./Exact_loss_testing_plots_best_model/Actual_values_per_radius_predicted_vs_exact_for_{number_of_samples}_samples_all_values_{n_epochs}_epochs_400k_data.png',
            dpi=1200, format='png', bbox_inches='tight') # CHANGE
plt.show()


