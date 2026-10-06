"""Hungarian 算法（O(n^3) 增广路版本，纯 Python，无第三方依赖）。

用于主题-真值对齐与文档聚类标签对齐（标签置换不变性）。与 scipy.optimize.linear_sum_assignment
结果一致（在可用时由单测交叉验证）。
"""


def hungarian(cost):
    """最小代价二分匹配。

    cost: (n, m) 类数组，最小化总代价。返回 (row_to_col, total_cost)。
    row_to_col[i] = 分配给第 i 行的列下标。
    """
    import numpy as np

    c = np.asarray(cost, dtype=float)
    n, m = c.shape
    s = max(n, m)
    if s == 0:
        return np.zeros(0, dtype=int), 0.0
    a = np.zeros((s, s), dtype=float)
    a[:n, :m] = c

    INF = 1e18
    u = [0.0] * (s + 1)
    v = [0.0] * (s + 1)
    p = [0] * (s + 1)  # p[j] = 分配给列 j 的行（1-索引）
    way = [0] * (s + 1)

    for i in range(1, s + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (s + 1)
        used = [False] * (s + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, s + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(s + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_col = [0] * s
    for j in range(1, s + 1):
        if p[j] > 0:
            row_to_col[p[j] - 1] = j - 1
    total = float(a[range(s), row_to_col].sum())
    return np.asarray(row_to_col[:n], dtype=int), total
