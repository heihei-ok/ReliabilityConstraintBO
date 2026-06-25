import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class data_load:
    def __init__(self, path, baseline):
        super(data_load, self).__init__()
        self.path = path
        self.baseline = baseline
        self.data = pd.read_excel(self.path)
        self.features = self.get_fea()
        self.lab = self.get_lab()
        self.info, self.stand_f = self.get_info()
        self.total_cost = self.get_cost()


    def get_fea(self):
        features = self.data.iloc[:, 1:4]
        return features

    def get_lab(self):
        lab = self.data.iloc[:, 4:6]
        return lab

    def get_info(self):
        info = self.data.iloc[:, 6:13]
        # Fill in the missing values with the mode of each column
        info_filled = info.apply(lambda col: col.fillna(col.mode()[0]))
        # text2num
        text_value = [x for x in self.baseline if isinstance(x, str)]
        text_ind = [i for i, x in enumerate(self.baseline) if isinstance(x, str)]
        info_text = info_filled.iloc[:, text_ind].copy()
        result = pd.DataFrame(index=info_text.index)
        for idx, col_idx in enumerate(text_ind):
            col_name = info_text.columns[idx]
            result[col_name] = np.where(
                info_text.iloc[:, idx] == text_value[idx],
                1.0,
                0.8
            )
        for col in result.columns:
            info_filled[col] = result[col]

        # Basic indicators
        num_base = copy.deepcopy(self.baseline)
        for ind in text_ind:
            num_base[ind] = 1

        # Calculate similarity
        base_vector = np.array(num_base).reshape(1, -1)
        similarities = []
        for _, row in info_filled.iterrows():
            row_vector = np.array(row).reshape(1, -1)
            sim = cosine_similarity(row_vector, base_vector)[0][0]
            similarities.append(sim)

        # Add constraints within the range of 0.5 to 0.8.
        similarities_arr = np.array(similarities).reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_lab = scaler.fit_transform(similarities_arr)
        fidelity = normalized_lab * 0.3 + 0.5
        info_filled['fidelity'] = fidelity
        return info_filled, normalized_lab

    def get_cost(self, ratio = [0.3, 0.3 ,0.4]):
        # Add constraints
        fea = self.features.to_numpy()
        x, y = fea.shape
        cost_array = np.zeros([x, y + 1])
        for i, (power, speed, density) in enumerate(fea):
            # Power_Cost
            if 150 <= power <= 175:
                p_cost = 1.0
            elif power < 150:
                p_cost =  0.8 * power / 150
            else:
                p_cost = 175 / power
            cost_array[i, 0] = p_cost
            # Speed_Cost
            if 800 <= speed <= 1200:
                v_cost = 1.0
            elif speed < 800:
                v_cost =  0.8 * speed / 800
            else:
                v_cost = 1200.0 / speed
            cost_array[i, 1] = v_cost
            # Density_Cost
            if 50.0 <= density <= 90.0:
                d_cost = 1.0
            elif density < 50.0:
                d_cost = 0.8 * density / 50.0
            else:
                d_cost = 90.0 / density
            cost_array[i, 2] = d_cost
            cost_array[i, 3] = (p_cost * v_cost * d_cost)
            # cost_array[i, 3] =ratio[0] * p_cost +ratio[1] * v_cost + ratio[2] * d_cost
        total_cost = pd.DataFrame(
                cost_array, columns=['Power_Cost', 'Speed_Cost', 'Density_Cost', 'Total_Cost'])
        return total_cost


class Get_stand:
    def __init__(self):
        super (Get_stand, self).__init__()
        self.scaler = None
    def get_scaler(self, Standardization):
        if Standardization=='MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif  Standardization=='StandardScaler':
            self.scaler = StandardScaler()
        else:
            raise AssertionError("please give the right Standardization")

    def star_stand(self, data):
        normalized_data = pd.DataFrame(self.scaler.fit_transform(data))
        return normalized_data

    def end_stand(self, data):
        original_data = self.scaler.inverse_transform(data)
        return original_data
