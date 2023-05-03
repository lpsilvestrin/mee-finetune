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


def linear_regression_torch(x, y, num_epochs=100, learning_rate=0.1, loss_name='MEE', s_x=1, s_y=1):
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
            batch_loss = loss_fn(batch_x, y_pred, batch_y, name=loss_name, s_x=s_x, s_y=s_y, debug=False)
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


def model_residuals(model, xtst, ytst, bias=0):

    # Compute predictions
    with torch.no_grad():
        y_pred = model(torch.tensor(xtst, dtype=torch.float32))
        y_pred = y_pred + bias
        res = torch.tensor(ytst, dtype=torch.float32) - y_pred

    return res.numpy().reshape(-1)


def kernel_sim():
    """
    plot MSE of a linear regression model learned from a train set and evaluated on a test set
    with different mean of the covariates
    the x-axis is the mean of the covariates and the y-axis is the MSE
    Returns:

    """
    noise_type = 'lap'
    gen = np.random.default_rng(_SEED)
    def gen_noise(n, type):
        if type == 'lap':
            return gen.laplace(0, 1, n)
        elif type == 'gaus':
            return gen.normal(0, 1, n)
        elif type == 'exp':
            return 1 - gen.exponential(1, n)
        else:
            raise ValueError('noise type not supported')

    def linsim(slope, x, noise):
        y = x.dot(slope) + noise
        return x, y.reshape(-1, 1)

    n_train = 1000
    n_test = 1000

    slope = gen.normal(0, 0.1, 100)
    repetitions = 2
    epochs = 500

    # train_data = [linear_simulation(n_train, xmean, xcov, std, slope, intercept) for _ in range(repetitions)]
    # train_data = [hsic_paper_simulation_exp(n_train, slope, x_train) for _ in range(repetitions)]
    train_data = [linsim(slope, gen.uniform(-1, 1, size=(n_train, 100)), gen_noise(n_train, noise_type)) for _ in range(repetitions)]
    seed_everything(_SEED+1)
    mse_models = [linear_regression_torch(x, y, num_epochs=epochs, learning_rate=1e-4, loss_name='mse') for x, y in train_data]
    # seed_everything(_SEED+1)
    # mae_models = [linear_regression_torch(x, y, num_epochs=epochs, learning_rate=1e-4, loss_name='MAE') for x, y in train_data]
    seed_everything(_SEED+1)
    mee_models = [linear_regression_torch(x, y, num_epochs=epochs, learning_rate=1e-4, loss_name='MEE', s_y=1) for x, y in train_data]
    seed_everything(_SEED + 1)
    mee2_models = [linear_regression_torch(x, y, num_epochs=epochs, learning_rate=1e-4, loss_name='MEE', s_y=1e-2) for x, y in train_data]
    seed_everything(_SEED + 1)
    mee3_models = [linear_regression_torch(x, y, num_epochs=epochs, learning_rate=1e-4, loss_name='MEE', s_y=20) for x, y in train_data]
    # seed_everything(_SEED+1)
    # hsic_models = [linear_regression_torch(x, y, num_epochs=epochs, learning_rate=1e-4, loss_name='HSIC') for x, y in train_data]

    res = []
    x_test, y_test = linsim(slope, gen.normal(0, 1, size=(n_test, 100)), gen_noise(n_test, noise_type))
    mse_res = np.concatenate([model_residuals(m,  x_test, y_test, bias=b) for m, b in mse_models])
    res.append(pd.DataFrame({'error': mse_res, 'loss': ["MSE"] * len(mse_res)}))
    mee_res = np.concatenate([model_residuals(m,  x_test, y_test, bias=b) for m, b in mee_models])
    res.append(pd.DataFrame({'error': mee_res, 'loss': ["MEE $\sigma=1$"] * len(mee_res)}))
    mee2_res = np.concatenate([model_residuals(m,  x_test, y_test, bias=b) for m, b in mee2_models])
    res.append(pd.DataFrame({'error': mee2_res, 'loss': ["MEE $\sigma=0.01$"] * len(mee2_res)}))
    mee3_res = np.concatenate([model_residuals(m, x_test, y_test, bias=b) for m, b in mee3_models])
    res.append(pd.DataFrame({'error': mee3_res, 'loss': ["MEE $\sigma=20$"] * len(mee3_res)}))

    df = pd.concat(res).reset_index(drop=True)
    # df = pd.DataFrame(res, columns=['error', 'loss'])
    # df = df.sample(2000, random_state=0)
    print(df.groupby('loss')['error'].agg(['mean', 'std']))
    sns.set_context('paper', font_scale=1.6)
    sns.set_style('whitegrid')
    # sns.lineplot(data=df, x='shift', y='MSE', hue='loss', style='loss', markers=True, dashes=False)
    p = sns.kdeplot(data=df, x='error', hue='loss')
    lss = [':', '--', '-.', '-']

    handles = p.legend_.legendHandles[::-1]

    for line, ls, handle in zip(p.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)

    plt.savefig(f'../plots/kernel_size_{noise_type}-noise_{repetitions}reps.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    kernel_sim()