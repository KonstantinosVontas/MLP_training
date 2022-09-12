import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import numpy as np
import csv

#from data_generate import radiusTest
radiusTest = 10

n_epochs = 5

path_training = './not_normalized_data/overlap_curvature_h1_300k_4_to_40_training.csv'
path_testing = './not_normalized_data/overlap_curvature_h1_100k_4_to_40_test.csv'
path_validation = './not_normalized_data/overlap_curvature_h1_100k_4_to_40_val.csv'
#path = 'Normalized_100K_data/overlap_curvature_h1_No_kappa_100k.csv'
path = './not_normalized_data/overlap_curvature_h1_300k_4_to_40_training.csv'
#path_testing = 'Testoverlap_curvature_h1_radius_10_10k_data.csv'



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

train_dataset = myData(path_training)#path_training
train_dataloader = DataLoader(train_dataset,
                              # batch_sampler=batch_sampler(),
                              # collate_fn=collate_batch,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

valid_dataset = myData(path_validation)#path_validation
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
# collate_fn=collate_batch)

test_dataset = myData(path_testing) #path_testing
testing_dataloader = DataLoader(test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)


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
#model = MLP([27, 20, 10, 5, 1], [nn.CELU(), nn.CELU(), nn.CELU(), nn.Identity()])
model = MLP([27, 20, 10, 5, 1], [nn.CELU(), nn.CELU(), nn.CELU(), nn.Sigmoid()])
# CELU, CELU, CELU, identity
print(model)
sum([param.nelement() for param in model.parameters()])

# optimizer = optim.SGD(model.parameters(), lr=1e-2)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# optimizer = optim.SGD(model.parameters(), lr=0.1)




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
total_loss_test_epoch = []
output_train_x_for_nmse = []

for epoch in range(n_epochs):
    trainingLoss = []
    running_loss = 0.0
    sum_train_loss_iter = 0.0
    sum_validation_loss_iter = 0.0
    sum_test_loss_iter = 0.0

    model.train()

    for i, data1 in enumerate(train_dataloader, 0):  # train_loader
        # get the inputs
        (inputs1, targets1) = data1

        is_train = True
        with torch.set_grad_enabled(is_train):
            optimizer.zero_grad()
            output_train = model(inputs1.float())  #
            loss_fn = torch.nn.MSELoss()
            #output_train_x_for_nmse.append(output_train)
            loss_train = loss_fn(output_train, targets1)

            iterations_counter = i + 1

            loss_train.backward()  # <2>
            optimizer.step()
            loss_train = loss_train.detach().numpy()
        sum_train_loss_iter = sum_train_loss_iter + (loss_train)
        print('[Epoch: %d, Iteration no: %5d] Training loss: %.5f' %
              (epoch + 1, i + 1, sum_train_loss_iter))
    sum_train_loss_iter = sum_train_loss_iter / (i + 1)
    total_loss_training_epoch.append(sum_train_loss_iter)
    # trainingLoss.append(loss_train)

    # print(f"Epoch {epoch + 1} | Batch: {i + 1} |, Training loss {loss_train.item():.4f},")

    for i, data2 in enumerate(valid_dataloader, 0):  # train_validation

        (inputs2, targets2) = data2

        with torch.no_grad():
            output_val = model(inputs2)  # <1>
            loss_val = loss_fn(output_val, targets2)
            loss_val = loss_val.detach().numpy()
            # validationLoss.append(loss_val)

        sum_validation_loss_iter = sum_validation_loss_iter + (loss_val)
        print('[Epoch: %d, Iteration no: %5d] Validation loss: %.5f' %
              (epoch + 1, i + 1, sum_validation_loss_iter))
    sum_validation_loss_iter = sum_validation_loss_iter / (i + 1)
    # total_loss_validation_epoch.append(sum_validation_loss_iter)

    # Save the Best Model
    if sum_validation_loss_iter < best_loss_from_validation:
        best_loss_from_validation = sum_validation_loss_iter
        # torch.save(model.state_dict(), './model.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
        }, f'./best_model_300k_data_{n_epochs}_epoch_CELU_x3_Identity.pth')
        # w = model.weight

    dict_validation_values[epoch] = sum_validation_loss_iter.item()

    total_loss_validation_epoch.append(sum_validation_loss_iter)


    #Test
    for i, data3 in enumerate(valid_dataloader, 0):  # train_validation

        (inputs3, targets3) = data3

        with torch.no_grad():
            output_test = model(inputs3)  # <1>
            loss_test = loss_fn(output_test, targets3)
            loss_test = loss_test.detach().numpy()
            # validationLoss.append(loss_val)

        sum_test_loss_iter = sum_test_loss_iter + (loss_test)
        print('[Epoch: %d, Iteration no: %5d] Test loss: %.5f' %
              (epoch + 1, i + 1, sum_validation_loss_iter))
    sum_test_loss_iter = sum_test_loss_iter / (i + 1)
    total_loss_test_epoch.append(sum_test_loss_iter)


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

# --------------------Plot----------------------#

n_epochs = n_epochs
plot_x = []
for i in range(n_epochs):
    plot_x.append(i)

print(plot_x)
print("4Jul")

plot_x = range(1, len(plot_x) + 1)
print(plot_x)

def myPlot():
    fig = plt.figure(dpi=300)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.plot(plot_x, total_loss_validation_epoch, '-', c='b', label="validation", linewidth=0.7)
    plt.plot(plot_x, total_loss_training_epoch, 'o', c='r', label="training", linewidth=0.5)
    plt.plot(plot_x, total_loss_test_epoch, 'x', c='k', label="training", linewidth=0.5)
    plt.plot(x_axis_epoch, y_axis_min_validation, '*', c='y', label="model minimum" "\n" "validation value epoch")
    plt.legend()
    plt.title("Curvature - 300K data, opt=ADAM, MLP([27, 20, 20, 10, 1]), lr=1e-2", fontsize=10)
    plt.savefig('Curvature - 300K data, ADAM, lr_0.01_nn_training2.pdf')
    plt.show()


myPlot()





#-----------------------Write CSV file- ----------------------------#

#TRAINING
outfile = open("./lists_for_plots/total_loss_training_epoch.csv",'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_loss_training_epoch))
outfile.close()

#VALIDATION
outfile = open("./lists_for_plots/total_loss_validation_epoch.csv",'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], total_loss_validation_epoch))
outfile.close()

# PLOT x_axis_epoch - for y_axis_min_validation
with open("./lists_for_plots/x_axis_best_model_epoch.csv", 'w') as f_output:
    f_output.write(str(x_axis_epoch))


# PLOT : y_axis_min_validation
with open("./lists_for_plots/y_axis_min_validation.csv", 'w') as f_output:
    f_output.write(str(y_axis_min_validation))



