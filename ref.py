import matplotlib.pyplot as plt

# 生成的路径点坐标数组
path_points = [[0, 0], [1, 1], [2, 2], [3, 3]]

x = [p[0] for p in path_points]
y = [p[1] for p in path_points]
plt.plot(x, y, '-o')
plt.show()
