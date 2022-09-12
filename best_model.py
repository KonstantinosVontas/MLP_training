import math
import pandas as pd
import os
import csv
import embeddings as embeddings
import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from nn_training import minimum_validation_value, model, optimizer, valid_dataloader, train_dataloader, loss_fn, \
    testing_dataloader, plot_x, total_loss_validation_epoch, total_loss_training_epoch, y_axis_min_validation, \
    x_axis_epoch, n_epochs, total_loss_test_epoch

radiusTest = 10
print('test5Jul')
print("\n Best model file, minimum val loss: ", minimum_validation_value)


checkpoint = torch.load(f"./best_model_300k_data_{n_epochs}_epoch_CELU_x3_Identity.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()


# model.load_state_dict(torch.load(model.pt))
# model.load_state_dict(torch.load("./model.pt"))
# model.eval()


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
#model = MLP([27, 20, 10, 5, 1], [nn.CELU(), nn.CELU(), nn.CELU(), nn.Sigmoid()])

print(model)
sum([param.nelement() for param in model.parameters()])


total_loss_train_epoch_best_m = []
total_loss_validation_epoch_best_m = []
total_loss_testing_epoch_best_m = []

print("this is test epoch: n", epoch)


def loss_test():
    # loss_fn = nn.MSELoss()

    # I HAVE TO PUT THE FOR LOOP ALSO HERE
    i = 0.0
    sum_train_loss_best_m = 0.0
    sum_validation_loss_best_m = 0.0
    sum_test_loss_best_m = 0.0
    # or for i
    for i, data1 in enumerate(train_dataloader, 0):  # train_loader
        (inputs1, targets1) = data1

        with torch.no_grad():
            output_train_best_m = model(inputs1.float())  #
            loss_train_best_m = loss_fn(output_train_best_m, targets1)
            loss_train_best_m = loss_train_best_m.detach().numpy()
        sum_train_loss_best_m = sum_train_loss_best_m + (loss_train_best_m)
        # print('[Iteration no: %5d] Validation loss: %.5f' %
        #      (i + 1, um_validation_loss_best_m))
    sum_train_loss_best_m = sum_train_loss_best_m / (i + 1)
    total_loss_train_epoch_best_m.append(sum_train_loss_best_m)

    for i, data2 in enumerate(valid_dataloader, 0):  # train_validation
        (inputs2, targets2) = data2

        with torch.no_grad():
            output_val_best_m = model(inputs2)  # <1>
            loss_val_best_m = loss_fn(output_val_best_m, targets2)
            loss_val_best_m = loss_val_best_m.detach().numpy()
            # iterations_counter2 = i + 1
        sum_validation_loss_best_m = sum_validation_loss_best_m + (loss_val_best_m)
        print('[Iteration no: %5d] Validation loss: %.5f' %
              (i + 1, sum_validation_loss_best_m))
    sum_validation_loss_best_m = sum_validation_loss_best_m / (i + 1)
    total_loss_validation_epoch_best_m.append(sum_validation_loss_best_m)

    for i, data3 in enumerate(testing_dataloader, 0):
        (inputs3, targets3) = data3

        with torch.no_grad():
            output_test_best_m = model(inputs3)  # <1>
            loss_test_best_m = loss_fn(output_test_best_m, targets3)
            loss_test_best_m = loss_test_best_m.detach().numpy()
            # iterations_counter2 = i + 1
        sum_test_loss_best_m = sum_test_loss_best_m + (loss_test_best_m)
        print('[Iteration no: %5d] Validation loss: %.5f' %
              (i + 1, sum_test_loss_best_m))
    sum_test_loss_best_m = sum_test_loss_best_m / (i + 1)
    total_loss_testing_epoch_best_m.append(sum_test_loss_best_m)


loss_test()

print("From loading the best model the minimum TRAINING loss is: ", total_loss_train_epoch_best_m)
print("From loading the best model the minimum VALIDATION loss is: ", total_loss_validation_epoch_best_m)
print("From loading the best model the minimum TESTING loss is: ", total_loss_testing_epoch_best_m)

print("30Jun")
plot_x = range(1, len(plot_x) + 1)
print(plot_x)

#PLOT X AXIS - EPOCHS
outfile = open("./lists_for_plots/epochs_nr_of_x_axis", 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], plot_x))
outfile.close()


## THIS PLOT IS GIVING THE BEST MODEL

fig = plt.figure(dpi=300)
plt.xlabel("Epoch")  # Fahrenheit
plt.ylabel("Loss")  # Celsius
plt.yscale("log")
plt.plot(plot_x, total_loss_validation_epoch, 'o', c='b', label="validation", markersize=2)
plt.plot(plot_x, total_loss_training_epoch, '-', c='r', label="training", linewidth=0.7)
plt.plot(plot_x, total_loss_test_epoch, 'x', c='k', label="test", linewidth=0.5)
plt.plot(x_axis_epoch, y_axis_min_validation, '*', c='y', label="model minimum" "\n" "validation value epoch")
plt.legend()
plt.title(f"Curvature - 300k data, opt=ADAM, MLP ), lr=1e-2", fontsize=10)
plt.savefig(f'Curvature-300k_data_{n_epochs}_epochs_CELU_x3_Sigmoid_at_end.pdf', dpi=700,  bbox_inches='tight')
plt.show()





