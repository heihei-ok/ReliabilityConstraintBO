import os

import numpy as np
import torch
import gpytorch
import botorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from gpytorch.mlls.exact_marginal_log_likelihood import  ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import heapq
from utils.data_load import data_load, Get_stand
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class MultivariateQuadraticMean(gpytorch.means.Mean):
    def __init__(self, input_dim=3, batch_shape=torch.Size(), bias=True):
        super().__init__()
        # 二次项参数矩阵 (input_dim x input_dim)
        self.register_parameter(
            name="second",
            parameter=torch.nn.Parameter(
                torch.randn(*batch_shape, input_dim, 1, dtype=torch.float64)))
        # 线性项参数向量 (input_dim)
        self.register_parameter(
            name="first",
            parameter=torch.nn.Parameter(
                torch.randn(*batch_shape, input_dim, 1, dtype=torch.float64)))

        if bias:
            self.register_parameter(
                name="bias",
                parameter=torch.nn.Parameter(
                    torch.randn(*batch_shape, 1, dtype=torch.float64)))
        else:
            self.bias = None

    def forward(self, x):
        # x shape: (..., input_dim)
        res = x.pow(2).matmul(self.second).squeeze(-1) + x.matmul(self.first).squeeze(
            -1
        )

        if self.bias is not None:
            res = res + self.bias.squeeze(-1)
        return res

class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = MultivariateQuadraticMean()
        # self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=1)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp_model(train_x, train_y, num_train_iters=1500):
    # declare the GP
    noise = 1e-4

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.likelihood.noise = noise

    # train the hyperparameter (the constant)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for i in range(num_train_iters):
        optimizer.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_y)

        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def fit_stgp_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y, mean_module = MultivariateQuadraticMean())
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll



def fit_MultiFidelity_gp_model(train_x, train_yf,data_fidelity = 3):
    torch.set_default_dtype(torch.double)
    model = SingleTaskMultiFidelityGP(train_x, train_yf, data_fidelity = data_fidelity)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll


def pre_gy(test_x, policy, num_obj, model,  Constraint_model, scaler_fea, scaler_lab, Constraint = True, MultiFidelity = False):
    ehvi_values = []

    #  Computational constraints
    utility_pred_mean = Constraint_model.posterior(test_x).mean.detach().numpy()
    utility_pred_var = Constraint_model.posterior(test_x).variance.detach().numpy()

    if MultiFidelity:
        test_x_fideity = torch.column_stack([test_x, torch.ones(test_x.size(0))])
        for i, x in enumerate(test_x_fideity):
            ehvi_value = policy(x.unsqueeze(0))
            ehvi_values.append(ehvi_value.item())
    else:
        for i, x in enumerate(test_x):
            ehvi_value = policy(x.unsqueeze(0))
            ehvi_values.append(ehvi_value.item())
    # 创建约束
    constrained_values = utility_pred_mean.reshape(-1, 1)
    scaler_ev = Get_stand()
    scaler_ev.get_scaler('MinMaxScaler')
    ehvi_values = np.array(ehvi_values).reshape(-1, 1)
    stand_ev = scaler_ev.star_stand(ehvi_values).to_numpy()
    constrained_ehvi_values = stand_ev * constrained_values
    if Constraint:
        max_indices = heapq.nlargest(num_obj, range(len(constrained_ehvi_values)),
                                     key=constrained_ehvi_values.__getitem__)
    else:
        max_indices = heapq.nlargest(num_obj, range(len(ehvi_values)),
                                     key=ehvi_values.__getitem__)
    selected_values = [test_x[i].tolist() for i in max_indices]
    selected_values = np.asarray(selected_values)
    pre_gy_para = scaler_fea.end_stand(selected_values)
    # pre uts ei
    if MultiFidelity:
        pred_lab_mean = model.posterior(test_x_fideity).mean.detach().numpy()
        pred_lab_var = model.posterior(test_x_fideity).variance.detach().numpy()
    else:
        pred_lab_mean = model.posterior(test_x).mean.detach().numpy()
        pred_lab_var = model.posterior(test_x).variance.detach().numpy()

    inv_pre_lab = scaler_lab.end_stand(pred_lab_mean[max_indices])

    return pre_gy_para, inv_pre_lab, pred_lab_var, ehvi_values, constrained_values, constrained_ehvi_values


class poster_info:
    def __init__(self,test_x, policy, num_obj, model,  Constraint_model, scaler_fea, scaler_lab, Constraint = True, MultiFidelity = False):
        super(poster_info, self).__init__()
        self.test_x = test_x
        self.policy = policy
        self.num_obj = num_obj
        self.model = model
        self.Constraint_model = Constraint_model
        self.scaler_fea = scaler_fea
        self.scaler_lab = scaler_lab
        self.Constraint = Constraint
        self.MultiFidelity = MultiFidelity
        self.ehvi_values = []
        self.constrained_values = None
        self.constrained_ehvi_values = None
        (self.pre_gy_para, self.inv_pre_lab, self.pred_lab_mean, self.pred_lab_var,
         self.pre_para_var,self.uninv_pre_lab)\
            = self.pre_gy(self.test_x, self.policy, self.num_obj, self.model,
                          self.Constraint_model, self.scaler_fea, self.scaler_lab, self.Constraint, self.MultiFidelity)

    def pre_gy(self, test_x, policy, num_obj, model, Constraint_model, scaler_fea, scaler_lab, Constraint=True,
               MultiFidelity=False):        #  Computational constraints
        utility_pred_mean = Constraint_model.posterior(test_x).mean.detach().numpy()
        utility_pred_var = Constraint_model.posterior(test_x).variance.detach().numpy()

        if MultiFidelity:
            test_x_fideity = torch.column_stack([test_x, torch.ones(test_x.size(0))])
            for i, x in enumerate(test_x_fideity):
                ehvi_value = policy(x.unsqueeze(0))
                self.ehvi_values.append(ehvi_value.item())
        else:
            for i, x in enumerate(test_x):
                ehvi_value = policy(x.unsqueeze(0))
                self.ehvi_values.append(ehvi_value.item())
        # 创建约束
        self.constrained_values = utility_pred_mean.reshape(-1, 1)
        scaler_ev = Get_stand()
        scaler_ev.get_scaler('MinMaxScaler')
        self.ehvi_values = np.array(self.ehvi_values).reshape(-1, 1)
        self.ehvi_values = scaler_ev.star_stand(self.ehvi_values).to_numpy()
        self.constrained_ehvi_values = self.ehvi_values * self.constrained_values
        if Constraint:
            max_indices = heapq.nlargest(num_obj, range(len(self.constrained_ehvi_values)),
                                         key=self.constrained_ehvi_values.__getitem__)
        else:
            max_indices = heapq.nlargest(num_obj, range(len(self.ehvi_values)),
                                         key=self.ehvi_values.__getitem__)
        selected_values = [test_x[i].tolist() for i in max_indices]
        selected_values = np.asarray(selected_values)
        pre_gy_para = scaler_fea.end_stand(selected_values)
        # pre uts ei
        if MultiFidelity:
            pred_lab_mean = model.posterior(test_x_fideity).mean.detach().numpy()
            pred_lab_var = model.posterior(test_x_fideity).variance.detach().numpy()
        else:
            pred_lab_mean = model.posterior(test_x).mean.detach().numpy()
            pred_lab_var = model.posterior(test_x).variance.detach().numpy()
        uninv_pre_lab = pred_lab_mean[max_indices]
        pred_lab_mean = scaler_lab.end_stand(pred_lab_mean)
        inv_pre_lab = pred_lab_mean[max_indices]
        pre_para_var = pred_lab_var[max_indices]
        return pre_gy_para, inv_pre_lab, pred_lab_mean, pred_lab_var, pre_para_var, uninv_pre_lab