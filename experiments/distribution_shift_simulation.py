import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer, LightningModule
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import mean_squared_error as tmse, mean_absolute_error as tmae

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


def polinomial_simulation(n, xrange, std, slope, intercept):
    """
    simulate a polinomial relationship between x and y with non-gaussian noise
    x is a univariate gaussian with mean xmean and covariance xcov

    """
    x = np.random.uniform(xrange[0], xrange[1], n)
    order = len(slope)
    x = np.array([x ** (i+1) for i in range(order)]).T

    y = x.dot(slope) + intercept + np.random.laplace(0, std, n)
    return x, y.reshape(-1, 1)


def hsic_paper_simulation_exp(n, std, slope, shift=0, x=None):
    if x is None:
        x = np.random.uniform(0+shift, 2+shift, size=(n, 100))
    x = x + shift
    y = x.dot(slope) + np.random.laplace(0, std, n)
    return x, y.reshape(-1, 1)


def linear_regression_torch(x, y, num_epochs=100, learning_rate=0.1, loss_name='MEE'):
    # Convert x and y to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create a dataset from x and y tensors
    dataset = TensorDataset(x_tensor, y_tensor)

    # Create a dataloader to load the data in batches
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

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
            batch_loss = loss_fn(batch_x, y_pred, batch_y, name=loss_name, s_x=1, s_y=1, debug=False)
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

    # compute bias
    bias = 0
    if loss_name in ['MEE', 'HSIC', 'MI']:
        with torch.no_grad():
            y_pred = model(x_tensor)
            bias = torch.mean(y_tensor - y_pred)

    return model, bias


def evaluate_model(model, xtst, ytst, bias=0):

    # Compute predictions
    with torch.no_grad():
        y_pred = model(torch.tensor(xtst, dtype=torch.float32))
        y_pred = y_pred + bias
        mse = tmse(y_pred, torch.tensor(ytst, dtype=torch.float32)).item()

    return mse


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
    # slope = np.array([1, 1])
    slope = np.random.normal(0, 0.1, 100)
    intercept = 5
    repetitions = 2
    max_shift = 2
    res = []

    # x_train = np.random.normal(1, 0.1, size=(n_train, 100))
    x_train = np.random.uniform(-1, 1, size=(n_train, 100))

    # train_data = [linear_simulation(n_train, xmean, xcov, std, slope, intercept) for _ in range(repetitions)]
    train_data = [hsic_paper_simulation_exp(n_train, std, slope, x=x_train) for _ in range(repetitions)]
    msl_models = [linear_regression_torch(x, y, num_epochs=500, learning_rate=1e-4, loss_name='mse') for x, y in train_data]
    mee_models = [linear_regression_torch(x, y, num_epochs=500, learning_rate=1e-4, loss_name='MEE') for x, y in train_data]
    hsic_models = [linear_regression_torch(x, y, num_epochs=500, learning_rate=1e-4, loss_name='HSIC') for x, y in train_data]

    for s in np.linspace(0, 2, 5):
        seed_everything(_SEED)
        x_test = np.random.normal(0, 1, size=(n_test, 100))
        # x_test, y_test = linear_simulation(n_test, xmean + s, xcov, std, slope, intercept)
        x_test, y_test = hsic_paper_simulation_exp(n_test, std, slope, shift=s, x=x_test)
        msl_res = [(s, evaluate_model(m, x_test, y_test, bias=0), 'MSL') for m, _ in msl_models]
        mee_res = [(s, evaluate_model(m, x_test, y_test, bias=b), 'MEE') for m, b in mee_models]
        hsic_res = [(s, evaluate_model(m, x_test, y_test, bias=b), 'HSIC') for m, b in hsic_models]
        res = res + msl_res + mee_res + hsic_res

    df = pd.DataFrame(res, columns=['shift', 'MSE', 'loss'])
    sns.lineplot(data=df, x='shift', y='MSE', hue='loss')
    plt.savefig('../plots/linear_regression_shift.png')
    plt.show()


if __name__ == '__main__':
    run_simulation()