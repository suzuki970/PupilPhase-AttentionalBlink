import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def rejectBlink_PCA(y):
    """Trials outside the ellipse of 6 SD on the first two principal components"""

    pipeline = make_pipeline(SimpleImputer(strategy='mean'), PCA(n_components=2))
    values = pipeline.fit_transform(y)

    stdX = np.std(values[:, 0])*6
    stdY = np.std(values[:, 1])*6

    P = (values[:, 0]**2/stdX**2) + (values[:, 1]**2/stdY**2) - 1
    rejctNum = np.argwhere(P > 0).reshape(-1)

    return values, rejctNum.tolist()
