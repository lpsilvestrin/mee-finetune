import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from lightning_lite import seed_everything
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader

from algorithms.custom_loss_torch_lit_class import loss_fn

_SEED = 42


def linear_simulation(n, xmean, xcov, std, slope, intercept):
    """
    simulate a linear relationship between x and y with non-gaussian noise
    x is a multivariate gaussian with mean xmean and covariance xcov

    """
    x = np.random.multivariate_normal(xmean, xcov, n)
    y = x.dot(slope) + intercept + np.random.laplace(0, std, n)
    return x, y.reshape(-1, 1)


def linear_regression_torch(x, y, xtst, num_epochs=100, learning_rate=0.1, loss_name='MEE'):
    # Convert x and y to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create a dataset from x and y tensors
    dataset = TensorDataset(x_tensor, y_tensor)

    # Create a dataloader to load the data in batches
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model parameters (slope and intercept)
    model = Linear(x.shape[1], y.shape[1])

    # Define optimizer (stochastic gradient descent)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            # Compute predictions
            y_pred = model(batch_x)

            # Compute loss
            batch_loss = loss_fn(batch_x, y_pred, batch_y, name=loss_name, s_y=2)
            epoch_loss += batch_loss.item()

            # Compute gradients
            batch_loss.backward()

            # Update model parameters
            optimizer.step()

            # Zero gradients
            optimizer.zero_grad()

        # Print progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss / len(dataloader)))

    # Compute predictions
    y_pred = model(torch.tensor(xtst, dtype=torch.float32))
    return y_pred.float().detach().numpy()


def run_simulation():
    """
    plot MSE of a linear regression model learned from a train set and evaluated on a test set
    with different mean of the covariates
    the x-axis is the mean of the covariates and the y-axis is the MSE
    Returns:

    """
    n_train = 1000
    n_test = 1000
    xmean = np.array([0, 0])
    xcov = np.array([[1, 0], [0, 1]])
    std = 1
    slope = np.array([1, 1])
    intercept = 5
    repetitions = 2
    max_shift = 5
    res = []
    for s in np.linspace(0, max_shift, 2):
        seed_everything(_SEED)
        for i in range(repetitions):
            x_train, y_train = linear_simulation(n_train, xmean, xcov, std, slope, intercept)
            x_test, y_test = linear_simulation(n_test, xmean + s, xcov, std, slope, intercept)
            # msl = LinearRegression().fit(x_train, y_train)
            # error = mean_squared_error(y_test, msl.predict(x_test))
            # res.append((s, error, 'MSL'))
            y_pred = linear_regression_torch(x_train, y_train, x_test, num_epochs=100, loss_name='mse')
            error = mean_squared_error(y_test, y_pred)
            res.append((s, error, 'MSLt'))
            y_pred = linear_regression_torch(x_train, y_train, x_test, num_epochs=100)
            error = mean_squared_error(y_test, y_pred)
            res.append((s, error, 'MEE'))

    df = pd.DataFrame(res, columns=['shift', 'MSE', 'loss'])
    sns.lineplot(data=df, x='shift', y='MSE', hue='loss')
    plt.show()


if __name__ == '__main__':
    run_simulation()