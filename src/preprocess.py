from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def standardization(X, desc=None):
    scaler = StandardScaler()
    if desc:
        for i in tqdm(range(0, X.shape[1], 1000), desc=desc):
            end_col = min(i + 1000, X.shape[1])
            X[:, i:end_col] = scaler.fit_transform(X[:, i:end_col])
    else:
        for i in range(0, X.shape[1], 1000):
            end_col = min(i + 1000, X.shape[1])
            X[:, i:end_col] = scaler.fit_transform(X[:, i:end_col])
    return X
