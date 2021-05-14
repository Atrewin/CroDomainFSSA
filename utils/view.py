# coding=utf-8

from sklearn.decomposition import pca
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


# 降维
def view_on_two_dim(points=[], proto_points=[]):
    # list: points, proto_points
    # points = points.reshape(-1, 768).tolist()
    # proto_points = proto_points.reshape(-1, 768).tolist()
    matrix = []
    if len(points) > 0:
        matrix.extend(points)
    if len(proto_points) > 0:
        matrix.extend(proto_points)

    K = 2
    model = pca.PCA(n_components=K).fit(matrix)
    points_d = model.transform(points)
#     proto_points_d = model.transform(proto_points)

    return points_d


# 画图
def scalar_point(points_d=[], rels=[], pic_name="d", radius=None, is_circle=False):
    # colors = ["mediumseagreen", "blueviolet"]
    fig = plt.figure()
    ax = plt.subplot()
    plt.axis([-3, 4, -3, 3])
    for i, mat in enumerate(points_d):
        xs = []
        ys = []
        for (x, y) in mat:
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, alpha=0.9, marker="o", label=rels[i])


    plt.legend(loc="upper right")

#     plt.savefig(pic_name)

def cal_mean(mat, proto):
    sum = 0
    for v in mat:
        sum += np.sum(np.square(proto - v))
    mean = sum / len(mat)
    print("mean:{}".format(mean))
    return mean


def calc_radius(point_d, proto_points_d, scale):
    # 这是所用的类别一起做吗？
    R = []
    mean_R = []
    for i, mat in enumerate(point_d):
        proto = proto_points_d[i]
        mean = cal_mean(mat, proto)
        R.append(np.abs(scale[i]) * mean)
        mean_R.append(mean)
    print("R:{}".format(R))
    return R, mean_R

if __name__ == "__main__":


    pass

    import traceback

    try:

        from sklearn.decomposition import pca
        from sklearn import preprocessing
        import numpy as np
        import matplotlib.pyplot as plt
        import json

        ours_file = r"D:\project\python\workSpace\NLP\few-shot learning\Reiteration\DACroDomainFSSA\roberta_newGraphours.json"
        with open(ours_file, "r", encoding="utf-8") as json_file:
            roberta_newGraphours = json.load(json_file)


        print("")

        ours_k_pos = roberta_newGraphours["S_domain"]["positive"]
        ours_k_neg = roberta_newGraphours["S_domain"]["negative"]
        ours_e_pos = roberta_newGraphours["T_domain"]["positive"]
        ours_e_neg = roberta_newGraphours["T_domain"]["negative"]

        matrix = []
        label = []
        matrix.extend(ours_k_pos[0:10])
        label.extend(["K+"] * 10)
        matrix.extend(ours_k_neg[0:10])
        label.extend(["K-"] * 10)
        matrix.extend(ours_e_pos[0:10])
        label.extend(["E+"] * 10)
        matrix.extend(ours_e_neg[0:10])
        label.extend(["E-"] * 10)

        ours_points = view_on_two_dim(matrix)

        points = []
        rel = ["K+", "K-", "E+", "E-"]

        for i in range(4):
            points.append(ours_points[i * 10:i * 10 + 10])


        scalar_point(ours_points, label)
        pass
    except:

        print(traceback.print_exc())
        pass