import torch
import numpy as np
from utils.data_load import data_load, Get_stand
import pandas as pd
import heapq
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from utils.model import fit_gp_model, fit_MultiFidelity_gp_model, fit_stgp_model, poster_info
torch.manual_seed(0)
lab_fid = True
# load train datas
baseline = ["electrode induction melting gas atomization",
            33, 30, 70, "zig-zag", 67, 80]
info = data_load(r'data/data_train_1.xlsx', baseline)
fea = info.features
fidelity = info.info['fidelity']
if lab_fid:
    ori_lab = info.lab
    # lab添加置信度
    lab = (torch.from_numpy(ori_lab.to_numpy()) *
           (( (torch.from_numpy(fidelity.to_numpy()).unsqueeze(1)-0.5) / 0.3)*0.1+0.8))
    lab[138:] = torch.from_numpy(ori_lab.to_numpy())[138:]
else:
    lab = info.lab
stand_f = info.stand_f
fidelity[138:] = 1
total_cost = info.total_cost['Total_Cost']

# load test data
power = np.linspace(start=25, stop=500, num=20)
speed = np.linspace(start=100, stop=2000, num=39)
X1, X2 = np.meshgrid(power, speed)
test_data = pd.DataFrame(np.vstack([X1.ravel(), X2.ravel()]).T, columns=['laser power, W','scan speed, mm/s'])
test_data['volumetric energy density (VED)=P/hvt , J/mm3'] = test_data['laser power, W'] / (test_data['scan speed, mm/s'] * 0.03 * 0.07)
test_x_fea = test_data.copy()

# feature stardardization
all_fea =  pd.concat([fea, test_x_fea], axis=0)
scaler_fea = Get_stand()
scaler_fea.get_scaler('MinMaxScaler')
stand_fea = scaler_fea.star_stand(all_fea)
split_idx = len(fea)
train_fea = stand_fea.iloc[:split_idx]
test_fea = stand_fea.iloc[split_idx:]
train_x = torch.from_numpy(train_fea.to_numpy())
test_x = torch.from_numpy(test_fea.to_numpy())

# label stardardization

scaler_lab = Get_stand()
scaler_lab.get_scaler('StandardScaler')
stand_lab = scaler_lab.star_stand(lab)
train_y = torch.from_numpy(stand_lab.to_numpy())
fidelity = torch.from_numpy(fidelity.to_numpy()).unsqueeze(1)
train_x_full = torch.cat([train_x, fidelity], dim=1)

# Reference point
ref_point = torch.tensor([train_y[:, 0].min(), train_y[:, 1].min()])
current_value  = torch.tensor([train_y[:, 0].max(), train_y[:, 1].max()])

# difine collection function
MultiFidelity = True
if MultiFidelity:
    model, likelihood = fit_MultiFidelity_gp_model(train_x_full, train_y)
else:
    model, likelihood = fit_stgp_model(train_x, train_y)

# difine Constraint model
Constraint_model, Constraint_likelihood = fit_stgp_model(train_x, torch.from_numpy(total_cost.to_numpy().reshape(-1,1)))
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement, qExpectedHypervolumeImprovement

if MultiFidelity:
    policy = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=NondominatedPartitioning(ref_point, train_y),
        X_baseline=train_x_full
    )
else:
        policy = ExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=NondominatedPartitioning(ref_point, train_y)
    )

num_obj = 2
Constraint = True
poster_info = poster_info(test_x, policy, num_obj, model,  Constraint_model,
                          scaler_fea, scaler_lab, Constraint, MultiFidelity)

print("Predicted process parameters (Power, Speed, VED):", poster_info.pre_gy_para)
print("Predicted process parameters (UTS, TE):", poster_info.inv_pre_lab)
# print("uncertainty of Predicted process parameters (UTS, TE):", poster_info.pre_para_var)
# print("Uninv Predicted process parameters (UTS, TE):", poster_info.uninv_pre_lab)