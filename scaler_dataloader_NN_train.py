import csv
import torch
from matplotlib import pyplot as plt
from numpy import reshape
from sklearn import preprocessing
import numpy as np
from statistics import median
from torch import from_numpy, nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchinfo import summary
from torchvision import models



# In this file we Normalize the data using StandardScaler() and we denormalize in order to see the Loss and Evolution of Training, Validation and Testing

with open('./not_normalized_data/overlap_curvature_h1_200000_4_to_40_train.csv') as file: #change
    reader = csv.reader(file)
    training_data = []
    for row in reader:
        training_data.append(row)
file.close()

with open('./not_normalized_data/overlap_curvature_h1_40000_4_to_40_val.csv') as file: #change
    reader = csv.reader(file, delimiter=',')
    val_data = []
    for row in reader:
        val_data.append(row)
file.close()

with open('./not_normalized_data/overlap_curvature_h1_40000_4_to_40_test.csv') as file: #change
    reader = csv.reader(file)
    test_data = []
    for row in reader:
        test_data.append(row)
file.close()

training_data = np.array(training_data, dtype='float64')
val_data = np.array(val_data, dtype='float64')
test_data = np.array(test_data, dtype='float64')

training_data = training_data[0:200000][:] #change
val_data = val_data[0:40000][:] #change
test_data = test_data[0:40000][:] #change

print("This is the training shape ", training_data.shape)
print("This is the validation shape ", val_data.shape)
print("This is the test shape: ", test_data.shape)

# Training
input_training_data_only = training_data[:, :-1]
target_training_data_only = training_data[:, 27:]
print("target training data only", target_training_data_only)

# Validation
input_val_data_only = val_data[:, :-1]
target_val_data_only = val_data[:, 27:]
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

# Validation
input_val_data_only_normalised = input_scaler.transform(input_val_data_only)
target_val_data_only_normalised = target_scaler.transform(target_val_data_only)

print("\ninput_validation normalised mean: ", input_val_data_only_normalised.mean(axis=0))
print("target_validation mean: ", target_val_data_only_normalised.mean(axis=0))

print("input_validation normalised std: ", input_val_data_only_normalised.std(axis=0))
print("target_validation normalised std: ", target_val_data_only_normalised.std(axis=0))

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


np.savetxt("./scaled_data/training_dataset_scaled.csv", np.c_[input_training_data_only_normalized, target_training_data_only_normalized], delimiter=",")
np.savetxt("./scaled_data/val_dataset_scaled.csv", np.c_[input_val_data_only_normalised, target_val_data_only_normalised], delimiter=",")
np.savetxt("./scaled_data/test_dataset_scaled.csv", np.c_[input_test_data_only_normalised, target_test_data_only_normalised], delimiter=",")



# ------------- TRAINING NN




radiusTest = 10

n_epochs = 4

path = "./scaled_data/training_dataset_scaled.csv"
path_training = "./scaled_data/training_dataset_scaled.csv"
path_validation = "./scaled_data/val_dataset_scaled.csv"
path_testing = "./scaled_data/test_dataset_scaled.csv"


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


BATCH_SIZE = 128

train_dataset = myData(path_training)  # path_training
train_dataloader = DataLoader(train_dataset,
                              # batch_sampler=batch_sampler(),
                              # collate_fn=collate_batch,
                              batch_size=BATCH_SIZE,
                              shuffle=False)# to chage to False if 200k is still less loss. All other  runs are with Fals

valid_dataset = myData(path_validation)  # path_validation
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
# collate_fn=collate_batch)

test_dataset = myData(path_testing)  # path_testing
testing_dataloader = DataLoader(test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)


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
print("This is the model: ", model)
sum([param.nelement() for param in model.parameters()])
loss_fn = torch.nn.MSELoss()


#model = models.vgg16()



print("This is the model summary(): ", model)


# optimizer = optim.SGD(model.parameters(), lr=1e-2)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# trainingLoss = []
validationLoss = []

min_Train_Value = {}
dict_validation_values = {}

# if __name__ == '__main__':  # Input this line

best_loss_from_train = 1e6
best_loss_from_validation = 1e6
# batch_counter = 1

total_loss_training_epoch = []
total_loss_validation_epoch = []
total_loss_testing_epoch = []

total_train_denormalized_epoch_rel_error = []
total_train_denormalized_epoch_abs_error = []

total_val_denormalized_epoch_rel_error = []
total_val_denormalized_epoch_abs_error = []

total_test_denormalized_epoch_rel_error = []
total_test_denormalized_epoch_abs_error = []


for epoch in range(n_epochs):
    trainingLoss = []
    running_loss = 0.0
    sum_train_loss_normalized_iter = 0.0
    sum_validation_loss_normalized_iter = 0.0
    sum_testing_loss_normalized_iter = 0.0

    sum_train_loss_denormalized_iter = 0.0
    sum_validation_loss_denormalized_iter = 0.0
    sum_testing_loss_denormalized_iter = 0.0

    sum_denormalized_iter_train_RL = 0.0
    sum_denormalized_iter_train_AE = 0.0

    sum_denormalized_iter_val_RL = 0.0
    sum_denormalized_iter_val_AE = 0.0

    sum_denormalized_iter_test_RL = 0.0
    sum_denormalized_iter_test_AE = 0.0

    relative_error_denormalized = []  # |target - prediction| / target
    absolute_error_denormalized = []  # |target - prediction|

    model.train()
    # -------------- training
    for i, data1 in enumerate(train_dataloader, start=0):  # train_loader
        # get the inputs
        (inputs1, targets1) = data1

        is_train = True
        with torch.set_grad_enabled(is_train):
            optimizer.zero_grad()
            output_train_normalized = model(inputs1.float())  #

            loss_train_normalized = loss_fn(output_train_normalized, targets1)

            loss_train_normalized.backward()  # <2>
            optimizer.step()
            loss_train_normalized = loss_train_normalized.detach().numpy()

            output_train_normalized = output_train_normalized.detach().numpy()   #DETACH ADDED
            targets_denormalized = target_standardscaler.inverse_transform(targets1)  #NOW THEY ARE DEORMALIZED ADDED
            #print("output_train_denormalized IS: ", targets_denormalized)
            output_train_denormalized = target_standardscaler.inverse_transform(output_train_normalized)

        #Loss
        sum_train_loss_normalized_iter = sum_train_loss_normalized_iter + (loss_train_normalized)

        #Relative Error
        sum_denormalized_iter_train_RL += sum(np.array(abs(targets_denormalized - output_train_denormalized) / targets_denormalized))
        #Absolute Error
        sum_denormalized_iter_train_AE += sum(np.array(abs(targets_denormalized - output_train_denormalized)))

        if epoch % 100 == 0:
            print('[Epoch: %d, Iteration no: %5d] Training loss: %.5f' %
                  (epoch + 1, i + 1, sum_train_loss_normalized_iter))
    #LossActual_values_and_Averaged
    sum_train_loss_normalized_iter = sum_train_loss_normalized_iter / (i + 1)  # NORMALISED
    total_loss_training_epoch.append(sum_train_loss_normalized_iter)

    #Relative Error Train
    sum_denormalized_iter_train_RL = sum_denormalized_iter_train_RL / ((i + 1) * BATCH_SIZE) # NORMALISED
    total_train_denormalized_epoch_rel_error.append(sum_denormalized_iter_train_RL)

    #Absolute Error Train
    sum_denormalized_iter_train_AE = sum_denormalized_iter_train_AE / ((i + 1) * BATCH_SIZE) # NORMALISED
    total_train_denormalized_epoch_abs_error.append(sum_denormalized_iter_train_AE)



    # trainingLoss.append(loss_train)

    # print(f"Epoch {epoch + 1} | Batch: {i + 1} |, Training loss {loss_train.item():.4f},")
    # -------------- validation
    for i, data2 in enumerate(valid_dataloader, 0):  # train_validation

        (inputs2, targets2) = data2
        model.train()

        with torch.no_grad():
            output_val_normalised = model(inputs2)  # <1>

            loss_val_normalized = loss_fn(output_val_normalised, targets2)
            loss_val_normalized = loss_val_normalized.detach().numpy()
            # validationLoss.append(loss_val)

            output_val_normalised = output_val_normalised.detach().numpy()   #DETACH ADDED
            targets_denormalized = target_standardscaler.inverse_transform(targets2)  #NOW THEY ARE DEORMALIZED ADDED
            #print("output_val_denormalized IS: ", targets_denormalized)
            output_val_denormalized = target_standardscaler.inverse_transform(output_val_normalised)

        #Loss
        sum_validation_loss_normalized_iter = sum_validation_loss_normalized_iter + (loss_val_normalized)
        #print('[Epoch: %d, Iteration no: %5d] Validation loss: %.5f' %
        #      (epoch + 1, i + 1, sum_validation_loss_normalized_iter))

        #Relative Error validation
        sum_denormalized_iter_val_RL += sum(np.array(abs(targets_denormalized - output_val_denormalized) / targets_denormalized))
        #Absolute Error validation
        sum_denormalized_iter_val_AE += sum(np.array(abs(targets_denormalized - output_val_denormalized)))



    #Loss
    sum_validation_loss_normalized_iter = sum_validation_loss_normalized_iter / (i + 1)
    # total_loss_validation_epoch.append(sum_train_loss_iter)

    # Save the Best Model
    if sum_validation_loss_normalized_iter < best_loss_from_validation:
        best_loss_from_validation = sum_validation_loss_normalized_iter
        # torch.save(model.state_dict(), './model.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
        }, f'./best_model_200k_data_{n_epochs}_epoch_CELU_x3_Identity.pth')
        # w = model.weight


    dict_validation_values[epoch] = sum_validation_loss_normalized_iter.item()

    #Loss val
    total_loss_validation_epoch.append(sum_validation_loss_normalized_iter)


    #Relative Error val
    sum_denormalized_iter_val_RL = sum_denormalized_iter_val_RL / ((i + 1) * BATCH_SIZE) # NORMALISED
    total_val_denormalized_epoch_rel_error.append(sum_denormalized_iter_val_RL)

    #Absolute Error val
    sum_denormalized_iter_val_AE = sum_denormalized_iter_val_AE / ((i + 1) * BATCH_SIZE) # NORMALISED
    total_val_denormalized_epoch_abs_error.append(sum_denormalized_iter_val_AE)



    # -------------- testing
    for i, data3 in enumerate(testing_dataloader, start=0):  # train_validation
        (inputs3, targets3) = data3

        with torch.no_grad():
            output_test_normalized = model(inputs3)  # <1>

            loss_test_normalized = loss_fn(output_test_normalized, targets3)
            loss_test_normalized = loss_test_normalized.detach().numpy()

            output_test_normalized = output_test_normalized.detach().numpy()   #DETACH ADDED
            targets_denormalized = target_standardscaler.inverse_transform(targets3)  #NOW THEY ARE DEORMALIZED ADDED
            #print("output_val_denormalized IS: ", targets_denormalized)
            output_test_denormalized = target_standardscaler.inverse_transform(output_test_normalized)

        #Loss test
        sum_testing_loss_normalized_iter = sum_testing_loss_normalized_iter + (loss_test_normalized)
        #print('[Epoch: %d, Iteration no: %5d] Test loss: %.5f' %
        #      (epoch + 1, i + 1, sum_validation_loss_normalized_iter))

        # Relative Error test
        sum_denormalized_iter_test_RL += sum(np.array(abs(targets_denormalized - output_test_denormalized) / targets_denormalized))
        # Absolute Error test
        sum_denormalized_iter_test_AE += sum(np.array(abs(targets_denormalized - output_test_denormalized)))

    #Loss test
    sum_testing_loss_normalized_iter = sum_testing_loss_normalized_iter / (i + 1)
    total_loss_testing_epoch.append(sum_testing_loss_normalized_iter)

    #Relative Error val
    sum_denormalized_iter_test_RL = sum_denormalized_iter_test_RL / ((i + 1) * BATCH_SIZE) # NORMALISED
    total_test_denormalized_epoch_rel_error.append(sum_denormalized_iter_test_RL)

    #Absolute Error val
    sum_denormalized_iter_test_AE = sum_denormalized_iter_test_AE / ((i + 1) * BATCH_SIZE) # NORMALISED
    total_test_denormalized_epoch_abs_error.append(sum_denormalized_iter_test_AE)

    # -------- end of testing


    #print("Loss training per epoch is: ", total_loss_training_epoch)
    #print("Loss validation per epoch is: ", total_loss_validation_epoch)
print("Loss training per epoch is: ", total_loss_training_epoch)
print("Loss validation per epoch is: ", total_loss_validation_epoch)

    # print("total_loss_training is: ", np.array(total_loss_training).shape)
    # print("total_loss_validation is: ", total_loss_validation)

    # epoch += 1
    # return (dict_validation_value, min_Train_Value)  # w

# ----------------Calling the Function---------------


print("Dictionary of Validation Values  ", dict_validation_values)
# print("The shape is:1 ", np.shape(minimum_validation_value))

# min_Train_value = minTrainreturn[min(minTrainreturn, key=minTrainreturn.get)]
# minimum_validation_value = dict_validation_values[min(dict_validation_values, key=dict_validation_values.get)]

minimum_validation_value = min(dict_validation_values.items(), key=lambda x: x[1])
minimum_validation_value = list(minimum_validation_value)

# print("\nThe minimum Training loss is: ", min_Train_value)
print("\nThe epoch and minimum Validation loss is:", minimum_validation_value, " respectively")
# print("\nThese are W", w)

# print("These are the weights of Layer [0]",model.layers[0].weight)
# print("These are the weights of Layer [2]",model.layers[2].weight)
# print("These are the weights of Layer [4]",model.layers[4].weight)
# print("These are the weights of Layer [6]",model.layers[6].weight)


x_axis_epoch = minimum_validation_value[0] + 1
y_axis_min_validation = minimum_validation_value[1]

print("x_axis_epoch ", x_axis_epoch)
print("y_axis_min_validation ", y_axis_min_validation)

# Training
total_loss_training_epoch_min_value = min(total_loss_training_epoch)
total_loss_training_epoch_min_index = total_loss_training_epoch.index(total_loss_training_epoch_min_value) + 1
total_loss_training_epoch_average_value = sum(total_loss_training_epoch) / len(total_loss_training_epoch)

print("\ntotal_loss_training_epoch_minimum_value: ", total_loss_training_epoch_min_value)
print("total_loss_training_epoch_min_index: ", total_loss_training_epoch_min_index)
print("total_loss_training_epoch_average_value: ", total_loss_training_epoch_average_value)

# Validation
total_loss_validation_epoch_min_value = min(total_loss_validation_epoch)
total_loss_validation_epoch_min_index = total_loss_validation_epoch.index(total_loss_validation_epoch_min_value) + 1
total_loss_validation_epoch_average_value = sum(total_loss_validation_epoch) / len(total_loss_validation_epoch)

print("\ntotal_loss_validation_epoch_minimum_value: ", total_loss_validation_epoch_min_value)
print("total_loss_validation_epoch_min_index: ", total_loss_validation_epoch_min_index)
print("total_loss_validation_epoch_average_value: ", total_loss_validation_epoch_average_value)

# Testing
total_loss_testing_epoch_min_value = min(total_loss_testing_epoch)
total_loss_testing_epoch_min_index = total_loss_testing_epoch.index(total_loss_testing_epoch_min_value) + 1
total_loss_testing_epoch_average_value = sum(total_loss_testing_epoch) / len(total_loss_testing_epoch)

print("\ntotal_loss_testing_epoch_minimum_value: ", total_loss_testing_epoch_min_value)
print("total_loss_testing_epoch_min_index: ", total_loss_testing_epoch_min_index)
print("total_loss_testing_epoch_average_value: ", total_loss_testing_epoch_average_value)

# Training Relative Error
total_train_denormalized_epoch_rel_error_min_value = min(total_train_denormalized_epoch_rel_error)
total_train_denormalized_epoch_rel_error_min_index = total_train_denormalized_epoch_rel_error.index(
    total_train_denormalized_epoch_rel_error_min_value) + 1
total_train_denormalized_epoch_rel_error_average_value = sum(total_train_denormalized_epoch_rel_error) / len(
    total_train_denormalized_epoch_rel_error)

print("\ntotal_train_denormalized_epoch_rel_error_minimum_value: ", total_train_denormalized_epoch_rel_error_min_value)
print("total_train_denormalized_epoch_rel_error_min_index: ", total_train_denormalized_epoch_rel_error_min_index)
print("total_train_denormalized_epoch_rel_error_average_value: ", total_train_denormalized_epoch_rel_error_average_value)

# Training Absolute Error
total_train_denormalized_epoch_abs_error_min_value = min(total_train_denormalized_epoch_abs_error)
total_train_denormalized_epoch_abs_error_min_index = total_train_denormalized_epoch_abs_error.index(
    total_train_denormalized_epoch_abs_error_min_value) + 1
total_train_denormalized_epoch_abs_error_average_value = sum(total_train_denormalized_epoch_abs_error) / len(
    total_train_denormalized_epoch_abs_error)

print("\ntotal_train_denormalized_epoch_abs_error_minimum_value: ", total_train_denormalized_epoch_abs_error_min_value)
print("total_train_denormalized_epoch_abs_error_min_index: ", total_train_denormalized_epoch_abs_error_min_index)
print("total_train_denormalized_epoch_abs_error_average_value: ", total_train_denormalized_epoch_abs_error_average_value)

# total_val_denormalized_epoch_rel_error =

# Validation Relative Error
total_val_denormalized_epoch_rel_error_min_value = min(total_val_denormalized_epoch_rel_error)
total_val_denormalized_epoch_rel_error_min_index = total_val_denormalized_epoch_rel_error.index(
    total_val_denormalized_epoch_rel_error_min_value) + 1
total_val_denormalized_epoch_rel_error_average_value = sum(total_val_denormalized_epoch_rel_error) / len(
    total_val_denormalized_epoch_rel_error)

print("\ntotal_val_denormalized_epoch_rel_error_minimum_value: ", total_val_denormalized_epoch_rel_error_min_value)
print("total_val_denormalized_epoch_rel_error_min_index: ", total_val_denormalized_epoch_rel_error_min_index)
print("total_val_denormalized_epoch_rel_error_average_value: ", total_val_denormalized_epoch_rel_error_average_value)

# total_val_denormalized_epoch_abs_error =
# Validation Absolute Error
total_val_denormalized_epoch_abs_error_min_value = min(total_val_denormalized_epoch_abs_error)
total_val_denormalized_epoch_abs_error_min_index = total_val_denormalized_epoch_abs_error.index(
    total_val_denormalized_epoch_abs_error_min_value) + 1
total_val_denormalized_epoch_abs_error_average_value = sum(total_val_denormalized_epoch_abs_error) / len(
    total_val_denormalized_epoch_abs_error)

print("\ntotal_val_denormalized_epoch_abs_error_minimum_value: ", total_val_denormalized_epoch_abs_error_min_value)
print("total_val_denormalized_epoch_abs_error_min_index: ", total_val_denormalized_epoch_abs_error_min_index)
print("total_val_denormalized_epoch_abs_error_average_value: ", total_val_denormalized_epoch_abs_error_average_value)

# total_test_denormalized_epoch_rel_error =
# Test Relative Error
total_test_denormalized_epoch_rel_error_min_value = min(total_test_denormalized_epoch_rel_error)
total_test_denormalized_epoch_rel_error_min_index = total_test_denormalized_epoch_rel_error.index(
    total_test_denormalized_epoch_rel_error_min_value) + 1
total_test_denormalized_epoch_rel_error_average_value = sum(total_test_denormalized_epoch_rel_error) / len(
    total_test_denormalized_epoch_rel_error)

print("\ntotal_test_denormalized_epoch_rel_error_minimum_value: ", total_test_denormalized_epoch_rel_error_min_value)
print("total_test_denormalized_epoch_rel_error_min_index: ", total_test_denormalized_epoch_rel_error_min_index)
print("total_test_denormalized_epoch_rel_error_average_value: ", total_test_denormalized_epoch_rel_error_average_value)

# total_test_denormalized_epoch_abs_error =
# Test Absolute Error
total_test_denormalized_epoch_abs_error_min_value = min(total_test_denormalized_epoch_abs_error)
total_test_denormalized_epoch_abs_error_min_index = total_test_denormalized_epoch_abs_error.index(
    total_test_denormalized_epoch_abs_error_min_value) + 1
total_test_denormalized_epoch_abs_error_average_value = sum(total_test_denormalized_epoch_abs_error) / len(
    total_test_denormalized_epoch_abs_error)

print("\ntotal_test_denormalized_epoch_abs_error_minimum_value: ", total_test_denormalized_epoch_abs_error_min_value)
print("total_test_denormalized_epoch_abs_error_min_index: ", total_test_denormalized_epoch_abs_error_min_index)
print("total_test_denormalized_epoch_abs_error_average_value: ", total_test_denormalized_epoch_abs_error_average_value)

# --------------------Plot----------------------#

n_epochs = n_epochs
plot_x = []
for i in range(n_epochs):
    plot_x.append(i)

print(plot_x)
print("4Jul")

plot_x = range(1, len(plot_x) + 1)
print(plot_x)



fig = plt.figure(dpi=300)
plt.xlabel("Epoch")
plt.ylabel(r'Relative error $\left[\frac{|target - predicted|}{target}\right]$')
plt.yscale("log")
plt.plot(plot_x, total_val_denormalized_epoch_rel_error, 'o', c='b', label="validation", markersize=2)
plt.plot(plot_x, total_train_denormalized_epoch_rel_error, '-', c='r', label="training", linewidth=0.7)
plt.plot(plot_x,total_test_denormalized_epoch_rel_error, '2', c='y', label="test")
plt.legend()
plt.title(f"Curvature Relative Error - 200k data, MLP([CELU_x3(), Identity()]), lr=1e-2", fontsize=7)
plt.savefig(f'plots_for_thesis/Curvature_Rel_error-200k_data_{n_epochs}_epochs_CELU_x3_Identity_end.pdf', dpi=700,  bbox_inches='tight')
plt.savefig(f'./plots_for_thesis/Curvature_Rel_error-200k_data_{n_epochs}_epochs_CELU_x3_Identity_end.eps',
            dpi=700, format='eps', bbox_inches='tight')
plt.savefig(f'./plots_for_thesis/Curvature_Rel_error-200k_data_{n_epochs}_epochs_CELU_x3_Identity_end.png',
            dpi=1200, format='png', bbox_inches='tight')
plt.show()


fig = plt.figure(dpi=300)
plt.xlabel("Epoch")
plt.ylabel(r'Absolute error $\left[|target - predicted|\right]$')
plt.yscale("log")
plt.plot(plot_x, total_val_denormalized_epoch_abs_error, 'o', c='b', label="validation", markersize=2)
plt.plot(plot_x, total_train_denormalized_epoch_abs_error, '-', c='r', label="training", linewidth=0.7)
plt.plot(plot_x,total_test_denormalized_epoch_abs_error, '1', c='y', label="test" "\n" "validation value epoch")
plt.legend()
plt.title(f"Curvature Absolute error- 200k data, MLP([CELU_x3(), Identity()]), lr=1e-2", fontsize=7)
plt.savefig(f'plots_for_thesis/Curvature_Abs_error-200k_data_{n_epochs}_epochs_CELU_x3_Identity_end.pdf', dpi=700,  bbox_inches='tight')
plt.savefig(f'./plots_for_thesis/Curvature_Abs_error-200k_data_{n_epochs}_epochs_CELU_x3_Identity_end.eps',
            dpi=700, format='eps', bbox_inches='tight')
plt.savefig(f'./plots_for_thesis/Curvature_Abs_error-200k_data_{n_epochs}_epochs_CELU_x3_Identity_end.png',
            dpi=1200, format='png', bbox_inches='tight')
plt.show()



def myPlot():
    fig = plt.figure(dpi=300)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.plot(plot_x, total_loss_training_epoch, '-', c='r', label="training", linewidth=0.7)
    plt.plot(plot_x, total_loss_validation_epoch, 'o', c='b', label="validation", linewidth=0.7, markersize=2)
    plt.plot(plot_x, total_loss_testing_epoch, '3', c='k', label="testing", linewidth=0.7, markersize=2)
    plt.plot(x_axis_epoch, y_axis_min_validation, '*', c='y', label="model minimum" "\n" "validation value epoch")
    plt.legend()
    plt.title("Curvature Loss - 200k data, MLP([CELU_x3(), Identity()]), lr=1e-2", fontsize=7)
    plt.savefig(f'./plots_for_thesis/Curvature - 200k data, ADAM,_{n_epochs} epochs, lr_0.01.pdf',
                dpi=700, format='pdf', bbox_inches='tight')
    plt.savefig(f'./plots_for_thesis/Curvature - 200k data, ADAM,_{n_epochs} epochs, lr_0.01.eps',
                dpi=700, format='eps', bbox_inches='tight')
    plt.savefig(f'./plots_for_thesis/Curvature - 200k data, ADAM,_{n_epochs} epochs, lr_0.01.png',
               dpi=1200, format='png', bbox_inches='tight')
    plt.show()


myPlot()

# -----------------------Write CSV file- ----------------------------#

# TRAINING
outfile = open("./lists_for_plots/200k_total_loss_training_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_loss_training_epoch))
outfile.close()

# VALIDATION
outfile = open("./lists_for_plots/200k_total_loss_validation_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_loss_validation_epoch))
outfile.close()

# Test
outfile = open("./lists_for_plots/200k_total_loss_test_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_loss_testing_epoch))
outfile.close()

# PLOT x_axis_epoch - for y_axis_min_validation
with open("./lists_for_plots/200k_x_axis_best_model_epoch.csv", 'w') as f_output:
    f_output.write(str(x_axis_epoch))

# PLOT : y_axis_min_validation
with open("./lists_for_plots/200k_y_axis_min_validation.csv", 'w') as f_output:
    f_output.write(str(y_axis_min_validation))



#Relative Error


# TRAINING
outfile = open("./lists_for_plots/200k_total_Relative_error_training_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_train_denormalized_epoch_rel_error))
outfile.close()

# VALIDATION
outfile = open("./lists_for_plots/200k_total_Relative_error_validation_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_val_denormalized_epoch_rel_error))
outfile.close()

# Test
outfile = open("./lists_for_plots/200k_total_Relative_error_test_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_test_denormalized_epoch_rel_error))
outfile.close()





#Absolute Error


# TRAINING
outfile = open("./lists_for_plots/200k_total_Absolute_error_training_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_train_denormalized_epoch_abs_error))
outfile.close()

# VALIDATION
outfile = open("./lists_for_plots/200k_total_Absolute_error_validation_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_val_denormalized_epoch_abs_error))
outfile.close()

# Test
outfile = open("./lists_for_plots/200k_total_Absolute_error_test_epoch.csv", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_test_denormalized_epoch_abs_error))
outfile.close()




# PLOT X AXIS - EPOCHS
# outfile = open("./lists_for_plots/epochs_nr_of_x_axis", 'w')
# out = csv.writer(outfile)
# out.writerows(map(lambda x: [x], plot_x))
# outfile.close()




# creating and initializing a nested list
errors = [
    [total_loss_training_epoch_min_value,
     total_loss_training_epoch_min_index,
     total_loss_training_epoch_average_value,
     total_loss_validation_epoch_min_value,
     total_loss_validation_epoch_min_index,
     total_loss_validation_epoch_average_value,
     total_loss_testing_epoch_min_value,
     total_loss_testing_epoch_min_index,
     total_loss_testing_epoch_average_value,
     total_train_denormalized_epoch_rel_error_min_value,
     total_train_denormalized_epoch_rel_error_min_index,
     total_train_denormalized_epoch_rel_error_average_value,
     total_train_denormalized_epoch_abs_error_min_value,
     total_train_denormalized_epoch_abs_error_min_index,
     total_train_denormalized_epoch_abs_error_average_value,
     total_val_denormalized_epoch_rel_error_min_value,
     total_val_denormalized_epoch_rel_error_min_index,
     total_val_denormalized_epoch_rel_error_average_value,
     total_val_denormalized_epoch_abs_error_min_value,
     total_val_denormalized_epoch_abs_error_min_index,
     total_val_denormalized_epoch_abs_error_average_value,
     total_test_denormalized_epoch_rel_error_min_value,
     total_test_denormalized_epoch_rel_error_min_index,
     total_test_denormalized_epoch_rel_error_average_value,
     total_test_denormalized_epoch_abs_error_min_value,
     total_test_denormalized_epoch_abs_error_min_index,
     total_test_denormalized_epoch_abs_error_average_value]]

# Create a DataFrame object
df = pd.DataFrame(errors,
                  columns=['total_loss_training_epoch_min_value',
                           'total_loss_training_epoch_min_index',
                           'total_loss_training_epoch_average_value',
                           'total_loss_validation_epoch_min_value',
                           'total_loss_validation_epoch_min_index',
                           'total_loss_validation_epoch_average_value',
                           'total_loss_testing_epoch_min_value',
                           'total_loss_testing_epoch_min_index',
                           'total_loss_testing_epoch_average_value',
                           'total_train_denormalized_epoch_rel_error_min_value',
                           'total_train_denormalized_epoch_rel_error_min_index',
                           'total_train_denormalized_epoch_rel_error_average_value',
                           'total_train_denormalized_epoch_abs_error_min_value',
                           'total_train_denormalized_epoch_abs_error_min_index',
                           'total_train_denormalized_epoch_abs_error_average_value',
                           'total_val_denormalized_epoch_rel_error_min_value',
                           'total_val_denormalized_epoch_rel_error_min_index',
                           'total_val_denormalized_epoch_rel_error_average_value',
                           'total_val_denormalized_epoch_abs_error_min_value',
                           'total_val_denormalized_epoch_abs_error_min_index',
                           'total_val_denormalized_epoch_abs_error_average_value',
                           'total_test_denormalized_epoch_rel_error_min_value',
                           'total_test_denormalized_epoch_rel_error_min_index',
                           'total_test_denormalized_epoch_rel_error_average_value',
                           'total_test_denormalized_epoch_abs_error_min_value',
                           'total_test_denormalized_epoch_abs_error_min_index',
                           'total_test_denormalized_epoch_abs_error_average_value'])

# df.to_csv (r'C:\Users\John\Desktop\export_dataframe.csv', index = None, header=True)
df.to_csv(
    f"./plots_for_thesis/200k_Celux3_Identity_end_{n_epochs}_epochs_All_Losses_Errors_.csv",
    index=None, header=True)

