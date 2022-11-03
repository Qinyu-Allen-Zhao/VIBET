import joblib
import numpy as np

noe = joblib.load('results/smooth_check/no_temporal_ThreeDPW.pkl')
gru = joblib.load('results/smooth_check/vibe_ThreeDPW.pkl')
tf = joblib.load('results/smooth_check/tf_ThreeDPW.pkl')


def eva_smoothness(data):
    kp_3d = data['pred_j3d']
    n = len(kp_3d)
    res = 0
    for i in range(n):
        s = []
        for j in range(int(len(kp_3d[0]) / 16) - 1):
            x = kp_3d[i][j * 16:(j + 1) * 16]
            dx = np.abs((x[1:] - x[:-1]).mean())
            s.append(dx)
        res += np.mean(s)

    return np.log(res / n)


print(eva_smoothness(noe), eva_smoothness(gru), eva_smoothness(tf))