###############################################Modified Code to 3D#####################################################
# Import matplotlib for plotting purposes
import matplotlib
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

# Sometimes, we need Numpy, but wherever possible, we prefer torch.
import numpy as np

# Import PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Float (32 bit) or Double (64 bit) precision? Choose!
torch.set_default_dtype(torch.float32)#64)
torch.set_num_threads(4) # Use _maximally_ 4 CPU cores

# This is a list to store the hyperparameters of the training
hyperparam_log = []

#device = torch.device("cpu")
# Choose a device for major calculations (if there is a special GPU card, you usually want that).
# My GPU is not very performant for double precision, so I stay on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
device = torch.device(device)

# File path for saving the trained NN later.
# If only providing the file name (like here), the file will be generated in the same folder as the Python script
model_file = "mymodel.torch"

## The following script shows a data-driven training of a Neural Network.
## The data is generated in this script, too, by sampling from a function.
## The variables below are used to control this sampling
samples = 16
sample_min = -5
sample_max = 5
sample_span = sample_max - sample_min

## These parameters are usual hyperparameters of the NN and its training.
batch_size = 8 # How many samples shall be presented to the NN, before running another optimizer step?
hidden_dim = 16 # How many neurons shall there be in the hidden layer(s)?
input_dim = 2 # Input dimension of the NN (i.e. how many neurons are in the input layer?)
output_dim = 1 # Some for output

epochs = 2000 # Number of training iterations to be performed
lr = 0.003 # Which learning rate is passed to the training algorithm?

## Choose a criterion to evaluate the results. Here, we choose Mean Square error.
## The term "loss" means about the same as "remaining error" or "residual".
criterion = nn.MSELoss(reduction="mean")

## Here, we create the training data. In this example, we draw samples within the sampling interval and then pass it to
## e.g. the sin function (choose other functions to experiment with this script)
train_x = (sample_span * torch.rand(samples, 2) + sample_min * torch.ones(samples, 2))
train_y = torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1]) ##torch.pow(test_x, 2)


# add noise to the training data
noise_level = 0.3  # noise level
train_y += noise_level * torch.randn_like(train_y)

## This function logs the hyperparameters of the training
def log_hyperparameters(config, train_loss, test_loss, train_losses):
    hyperparam_log.append({
        "learning_rate": config["lr"],
        "hidden_dim": config["hidden_dim"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
        "train_losses": train_losses  # Include train_losses here
    })


print ("train_x", train_x)
print ("train_y", train_y)

## Here, we create the test data. To show the effect of extrapolation and the detailed behavior of the NN,
## we choose points a bit outside the interval that we used for sampling training data. Also, we very dense points.
# test_x = torch.zeros(20*samples, 2)
# test_x[:, 0] = torch.linspace(sample_min - 0.5*sample_span, sample_max + 0.5*sample_span, test_x.size()[0])
# test_x[:, 1] = torch.linspace(sample_min - 0.5*sample_span, sample_max + 0.5*sample_span, test_x.size()[0])

# create test_x with meshgrid
grid_size = int(samples ** 0.5) * 2 
margin = 1  # only extend 1. from -5 to 5, extend to -6 to 6 
x1 = torch.linspace(sample_min - margin, sample_max + margin, 2*samples)
x2 = torch.linspace(sample_min - margin, sample_max + margin, 2*samples)
x1, x2 = torch.meshgrid(x1, x2) 
test_x = torch.zeros(2*samples*2*samples, 2)
test_x[:, 0] = x1.reshape(-1)
test_x[:, 1] = x2.reshape(-1)
test_y = torch.sin(test_x[:, 0]) + torch.cos(test_x[:, 1]) ##torch.pow(test_x, 2)

## We want a DataLoader to handle batching and shuffling of the training data for us.
## The DataLoader needs a TensorDataset, hence we create one from the Training data.
train_data = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)


## This class creates the actual Neural Network.
class MLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLNet, self).__init__()
        self.hidden_dim = hidden_dim

        ## In this case, I decided to pack the layers into a single container of type nn.Sequential
        ## This container is useful for Fully Connected NNs, where the output of each layer is just fed into the following layer.
        ## added dropout layer, which is a regularization technique to prevent overfitting
        self.fcnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.05),  # Dropout layer with 30% of neurons being dropped，but it's not working well，the model underfitting, so change to 0.05
            #nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ) ## That network here has exactly 1 hidden layer (comment out some lines or write new ones to add more layers)
        ## nn.Linear is in the pictures represented by the arrows and has the weights (and biases) which are the
        ## parameters of the NN.
        ## nn.ReLU and Tanh are examples for activation functions - in the pictures, these are the bubbles / neurons.
        ## Usually, the pattern is interchanging between one linear layer, one activation function, one linear layer, one activation function and so on

    ## It is required to write a function that is executed when calling the NN model.
    ## Usually, it takes the input data and passes it through the layers.
    ## Here, it is sufficient to call the sequential container since this container does exactly this job.
    ## But this function "forward" could also be used to set up more complex NN architectures with more complex data flows
    def forward(self, x):
        out = self.fcnn1(x)
        return out

## This function performs a test run with the NN.
## It takes the NN model and the test data, passes the test inputs through the network and
## compares it with the target values = test outputs = targets
## and based on that calculates the loss value
def evaluate(model, test_x, test_y):
    ## For testing, we don't need the autograd feature/ protocol of all calculation steps
    ## So, save some time and disable grad tracking.
    with torch.no_grad():
        model.eval() ## Set the NN model into evaluation mode
        outputs = [] ## Create empty lists to store the results
        targets = []
        testlosses = []

        out = model(test_x.to(device)) ## Call the model, i.e. perform the actual inference

        ## Move the output quantities to the CPU, detach them from the tensor operation book-keeping and convert them to numpy arrays/ vectors.
        ## This is all necessary for plotting
        outputs.append(out.cpu().detach().numpy())
        targets.append(test_y.cpu().detach().numpy())
        testlosses.append(criterion(out, test_y.to(device)).item())

    ## Now return that in form of a triple of variables
    return outputs, targets, testlosses

## This calls the evaluate function and takes care of the plotting.
def eval_and_plot(model):
        ## matplotlib tries to be similar to the plot functions of matlab  (admittedly, the commands have to begin with "plt.", but the rest
        ## is quite similar.)
        '''
        plt.subplot(1, 1, 1)

        ## Call the network on the test data
        net_outputs_test, targets_test, testlosses = evaluate(model, test_x, test_y)
        # Plot the targets first in blue (which means to plot the actual function over the whole test interval)
        plt.plot(test_x, targets_test[0], "-x", color="b", label="Target")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

        ## Call the network on the training data
        net_outputs_train, targets_train, testlosses = evaluate(model, train_x, train_y)
        ## First, plot the targets in red, i.e. plot the training data set
        plt.plot(train_x, targets_train[0], "^", color="r", label="Target")

        ## Now, plot the output of the NN on the whole test interval in green
        ## This allows us to see how the NN performs for interpolation as well as for extrapolation
        plt.plot(test_x, net_outputs_test[0], "-o", color="g", label="Learned")

        ## All plotting is done, open the plot window
        plt.show()
        '''
        # create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ## Call the network on the test data
        net_outputs_test, targets_test, testlosses = evaluate(model, test_x, test_y)   
        # Plot the targets first in blue (which means to plot the actual function over the whole test interval)
        #ax.plot(test_x[:, 0], test_x[:, 1], targets_test[0], color="-b", label="Target")
        ax.scatter(test_x[:, 0], test_x[:, 1], targets_test[0], c='b', marker='x', label='Target')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

        ## Call the network on the training data
        net_outputs_train, targets_train, testlosses = evaluate(model, train_x, train_y)
        ## First, plot the targets in red, i.e. plot the training data set
        ax.scatter(train_x[:, 0], train_x[:, 1], targets_train[0], c='r', s=100, marker='^', label='Target')

        ## Now, plot the output of the NN on the whole test interval in green
        ## This allows us to see how the NN performs for interpolation as well as for extrapolation
        ax.scatter(test_x[:, 0], test_x[:, 1], net_outputs_test[0].reshape(-1), c='g', marker='o', label='Learned')
        print("net_outputs_test[0]", net_outputs_test[0].reshape(-1))

        print("Prediction min:", net_outputs_test[0].min())
        print("Prediction max:", net_outputs_test[0].max())


        ## All plotting is done, open the plot window
        plt.show()

## This function plots the overfitting effect
def plot_overfitting_effect(model):
    net_outputs_train, targets_train, _ = evaluate(model, train_x, train_y)
    net_outputs_test, targets_test, _ = evaluate(model, test_x, test_y)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # testing data target (blue)
    ax.scatter(
        test_x[:,0].cpu().numpy(),
        test_x[:,1].cpu().numpy(),
        targets_test[0],
        c='b', marker='x', alpha=0.5, s=20,
        label='Test Target'
    )
    
    # testing data prediction (green, add alpha/s)
    ax.scatter(
        test_x[:,0].cpu().numpy(),
        test_x[:,1].cpu().numpy(),
        net_outputs_test[0],
        c='g', marker='o', alpha=0.3, s=10,
        label='Test Prediction'
    )
    
    # training data target (red)
    ax.scatter(
        train_x[:,0].cpu().numpy(),
        train_x[:,1].cpu().numpy(),
        targets_train[0],
        c='r', marker='^', s=80,
        label='Train Target'
    )
    
    # training data prediction (orange)
    ax.scatter(
        train_x[:,0].cpu().numpy(),
        train_x[:,1].cpu().numpy(),
        net_outputs_train[0],
        c='orange', marker='o', s=60,
        label='Train Prediction'
    )
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.legend()
    plt.title("Overfitting Effect (3D)")
    plt.show()



## That function takes care of the whole training
def train(train_loader, learn_rate, EPOCHS):  # 10):

    # Instantiate the NN
    model = MLNet(input_dim, hidden_dim, output_dim)
    model.to(device) # and move it to the "device" (in case we use a GPU)

    ## Choose an optimizer. Adam is quite robust and thus very popular. Technically, it's based on
    ## gradient descent, but mixes in the gradient of the last time step to improve robustness.
    ## It is given the model parameters, which are the weights and biases of the Linear layers
    ## and the learn_rate. 
    # add L2 regularization(weight_decay) to the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    print("Starting Training of the model")

    # StepLR scheduler: every 'step_size' epochs, lr = lr * gamma
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.3)
    #print("Starting Training of the model with StepLR scheduler")

    # We want to keep track of the losses averaged over each epoch, to plot them in those famous
    # decreasing graphs
    avg_losses = torch.zeros(EPOCHS)

    ## In the end, epoch is just another word for "training iteration", so we have a simple for loop.
    for epoch in range(EPOCHS):
        model.train() # Set the model into train mode
        avg_loss = 0. # initializations
        counter = 0

        ## DataLoader is iterable so that this for-loop loops over the batches of the training data set
        ## and the DataLoader gives us readily paired combinations of training inputs and targets (which are called x and label, here).
        ## The term "label" is more common in classification, but used for all supervised training tasks
        for x, label in train_loader:
            counter += 1 # We count, how many batches we did within the current epoch
            model.zero_grad() # Important: reset the gradients of the NN before passing the training inputs.
            # Otherwise, we would accumulate the gradient information which might ruin the results
            # or simply run into PyTorch exceptions

            ## Now, we can call the model on the training inputs.
            ## Therefor, we move that data to the device (just in case we use a GPU).
            out = model(x.to(device))
            out = out.squeeze(-1) ## The output is a tensor of shape (batch_size, 1), but we want to have a tensor of shape (batch_size)
            loss = criterion(out, label) # both are [batch_size,1]


            ## Here, the whole magic happens:
            ## PyTorch offers the autograd feature, i.e. calculations on tensors are tracked (there are exceptions, e.g.
            ## this is not possible for in-place operations). This allows to calculate the derivative of each output value
            ## w.r.t. all input values (its "gradient").
            ## In this case, we want to know the derivatives of the loss value w.r.t. all the NN parameters (weights and biases).
            ## Our optimizers are usually based on "Gradient Descent", so we need the gradients...
            loss.backward()
            ## Alright, now let the optimizer do the magic.
            ## We passed the optimizer the NN parameters by reference when we initialized the optimizer.
            ## And the gradient information is stored in the tensors of the parameters, too (not in the loss or so).
            ## That's why the step() function does not require any arguments.
            optimizer.step()

            ## For the plot at the end, save the loss values
            avg_loss += loss.item()

            if counter % 20 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {} = {} + {}".format(epoch, counter,
                                                                                                len(train_loader),
                                                                                                avg_loss / counter, loss.item(), 0))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        ## It's an average loss, so divide by the number of samples/ size of the training data set
        avg_losses[epoch] = avg_loss / len(train_loader)

        ## StepLR scheduler step, update learning rate, every 500 epochs reduce lr by 0.3
    #    scheduler.step()

        ## To understand how the NN learns, all 500 epochs a plot is shown (Close the plot window and wait for the next plot)
        if epoch%500 == 0:
            eval_and_plot(model)
    #       current_lr = scheduler.get_last_lr()[0]
    #       print(f"Current LR after scheduler step: {current_lr}")


    # Log hyperparameters and losses
    final_train_loss = avg_losses[-1].item()
    _, _, final_test_loss = evaluate(model, test_x, test_y)

    ## This logs the hyperparameters and the losses
    log_hyperparameters(
        config={
            "lr": learn_rate,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "epochs": EPOCHS,
        },
        train_loss=final_train_loss,
        test_loss=final_test_loss,
        train_losses=avg_losses.cpu().tolist()  # Convert to a list to store
    )


    ## This plots the loss curve
    plt.figure(figsize=(12, 8))
    plt.plot(avg_losses, "x-")
    plt.title("Train loss (MSE, reduction=mean, averaged over epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.grid(visible=True, which='both', axis='both')
    plt.show()

    ## This plots the final evaluation
    '''
    plt.figure(figsize=(12, 8))
    for log in hyperparam_log:
        plt.plot(log["train_losses"], label=f'lr={log["learning_rate"]}, hidden={log["hidden_dim"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss across Hyperparameter Configurations')
    plt.legend()
    plt.show()
    '''

    ## Now save the trained model with all its properties to the model_file
    torch.save(model, model_file)

    ## And return the model in case we want to use it for other tasks
    return model

## This function is used to plot the hyperparameter logs
def plot_hyperparameter_logs():
    plt.figure(figsize=(12, 8))
    for log in hyperparam_log:
        plt.plot(log["final_train_loss"], label=f'lr={log["learning_rate"]}, hidden={log["hidden_dim"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Final Training Loss')
    plt.title('Training Loss across Hyperparameter Configurations')
    plt.legend()
    plt.show()

## As the functions written before are just function definitions,
## those functions still have to be called.
## This is done here.
## Train the model
model = train(train_loader, lr, epochs)

## Prove overfitting (this should come after training is done)
plot_overfitting_effect(model)

## Test that saving the model worked: load it from file
model = torch.load(model_file)
## and evaluate and plot once
eval_and_plot(model)


print(model)
'''
print("layer1, weights: ", model.layer1.weight)
print("layer1, bias: ", model.layer1.bias)
#print(model.layer2.weight)
#print(model.layer2.bias)
print("layer3, weights: ", model.layer3.weight)
print("layer3, bias: ", model.layer3.bias)
'''

print("layer1, weights: ", model.fcnn1[0].weight)
print("layer1, bias: ", model.fcnn1[0].bias)
# Uncomment and adapt these for additional layers if needed
# print("layer2, weights: ", model.fcnn1[2].weight)
# print("layer2, bias: ", model.fcnn1[2].bias)
print("layer3, weights: ", model.fcnn1[-1].weight)
print("layer3, bias: ", model.fcnn1[-1].bias)

## This function is used to plot the hyperparameter logs
plot_hyperparameter_logs()