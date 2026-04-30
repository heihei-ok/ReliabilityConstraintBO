"""Data loading helpers for the LPBF reliability-constrained BO workflow."""

import copy

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DEFAULT_BASELINE = [
    "electrode induction melting gas atomization",
    33,
    30,
    70,
    "zig-zag",
    67,
    80,
]


def extract_features(data):
    """Return process features: power, speed, and VED."""
    return data.iloc[:, 1:4].copy()


def extract_labels(data):
    """Return target labels: UTS and TE."""
    return data.iloc[:, 4:6].copy()


def extract_process_info(data):
    """Return auxiliary process information used to derive fidelity."""
    return data.iloc[:, 6:13].copy()


def _fill_modes(frame):
    def fill_column(col):
        modes = col.mode(dropna=True)
        if len(modes) == 0:
            return col.fillna(0)
        return col.fillna(modes.iloc[0])

    return frame.apply(fill_column)


def compute_fidelity(process_info, baseline=None):
    """Compute the literature fidelity score used by the original scripts.

    Text-valued baseline columns are converted to 1.0 for exact matches and
    0.8 otherwise. The cosine similarity to the baseline vector is then scaled
    into the [0.5, 0.8] fidelity interval.
    """
    if baseline is None:
        baseline = DEFAULT_BASELINE
    if process_info.shape[1] != len(baseline):
        raise ValueError(
            "Expected {} process-info columns, got {}".format(
                len(baseline), process_info.shape[1]
            )
        )

    info_filled = _fill_modes(process_info.copy())
    text_values = [x for x in baseline if isinstance(x, str)]
    text_indices = [i for i, x in enumerate(baseline) if isinstance(x, str)]

    if text_indices:
        text_frame = info_filled.iloc[:, text_indices].copy()
        text_result = pd.DataFrame(index=text_frame.index)
        for idx, col_idx in enumerate(text_indices):
            col_name = text_frame.columns[idx]
            text_result[col_name] = np.where(
                text_frame.iloc[:, idx] == text_values[idx], 1.0, 0.8
            )
        for col in text_result.columns:
            info_filled[col] = text_result[col]

    numeric_baseline = copy.deepcopy(list(baseline))
    for idx in text_indices:
        numeric_baseline[idx] = 1

    numeric_info = info_filled.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    base_vector = np.asarray(numeric_baseline, dtype=float).reshape(1, -1)

    similarities = []
    for _, row in numeric_info.iterrows():
        row_vector = np.asarray(row, dtype=float).reshape(1, -1)
        similarities.append(cosine_similarity(row_vector, base_vector)[0][0])

    similarities_arr = np.asarray(similarities, dtype=float).reshape(-1, 1)
    normalized = MinMaxScaler().fit_transform(similarities_arr)
    fidelity = normalized * 0.3 + 0.5
    return pd.Series(fidelity.reshape(-1), index=process_info.index, name="fidelity")


def compute_total_cost(features):
    """Compute the original process utility / constraint score."""
    fea = features.to_numpy(dtype=float)
    cost_array = np.zeros([fea.shape[0], fea.shape[1] + 1], dtype=float)

    for i, (power, speed, density) in enumerate(fea):
        if 150 <= power <= 175:
            p_cost = 1.0
        elif power < 150:
            p_cost = 0.8 * power / 150.0
        else:
            p_cost = 175.0 / power

        if 800 <= speed <= 1200:
            v_cost = 1.0
        elif speed < 800:
            v_cost = 0.8 * speed / 800.0
        else:
            v_cost = 1200.0 / speed

        if 50.0 <= density <= 90.0:
            d_cost = 1.0
        elif density < 50.0:
            d_cost = 0.8 * density / 50.0
        else:
            d_cost = 90.0 / density

        cost_array[i, 0] = p_cost
        cost_array[i, 1] = v_cost
        cost_array[i, 2] = d_cost
        cost_array[i, 3] = p_cost * v_cost * d_cost

    return pd.DataFrame(
        cost_array,
        index=features.index,
        columns=["Power_Cost", "Speed_Cost", "Density_Cost", "Total_Cost"],
    )


def apply_fidelity_to_labels(labels, fidelity, data_rows=None, literature_rows=138):
    """Apply the original confidence adjustment to literature labels only."""
    label_values = labels.to_numpy(dtype=float)
    fidelity_values = np.asarray(fidelity, dtype=float).reshape(-1, 1)
    adjusted = label_values * (((fidelity_values - 0.5) / 0.3) * 0.1 + 0.8)

    if data_rows is not None:
        data_rows = np.asarray(data_rows, dtype=int)
        experiment_mask = data_rows > literature_rows
        adjusted[experiment_mask] = label_values[experiment_mask]

    return pd.DataFrame(adjusted, index=labels.index, columns=labels.columns)


class data_load:
    """Compatibility wrapper for the original scripts."""

    def __init__(self, path, baseline=None):
        self.path = path
        self.baseline = baseline or DEFAULT_BASELINE
        self.data = pd.read_excel(self.path)
        self.features = self.get_fea()
        self.lab = self.get_lab()
        self.info, self.stand_f = self.get_info()
        self.total_cost = self.get_cost()

    def get_fea(self):
        return extract_features(self.data)

    def get_lab(self):
        return extract_labels(self.data)

    def get_info(self):
        info = extract_process_info(self.data)
        fidelity = compute_fidelity(info, self.baseline)
        info_filled = _fill_modes(info)
        info_filled["fidelity"] = fidelity
        stand_f = MinMaxScaler().fit_transform(
            cosine_similarity(
                info_filled.drop(columns=["fidelity"], errors="ignore")
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0),
                np.asarray([1 if isinstance(x, str) else x for x in self.baseline])
                .reshape(1, -1)
                .astype(float),
            )
        )
        return info_filled, pd.DataFrame(stand_f, index=info.index)

    def get_cost(self):
        return compute_total_cost(self.features)


class Get_stand:
    """Small scaler wrapper preserved from the original implementation."""

    def __init__(self):
        self.scaler = None

    def get_scaler(self, standardization):
        if standardization == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif standardization == "StandardScaler":
            self.scaler = StandardScaler()
        else:
            raise AssertionError("please give the right Standardization")

    def star_stand(self, data):
        if self.scaler is None:
            raise RuntimeError("Call get_scaler before star_stand.")
        return pd.DataFrame(
            self.scaler.fit_transform(data),
            index=getattr(data, "index", None),
            columns=getattr(data, "columns", None),
        )

    def end_stand(self, data):
        if self.scaler is None:
            raise RuntimeError("Call get_scaler before end_stand.")
        return self.scaler.inverse_transform(data)
