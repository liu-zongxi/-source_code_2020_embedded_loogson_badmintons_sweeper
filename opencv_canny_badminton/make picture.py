#1.plt.subplot（ijn）形式，其中ij是行列数，n是第几个图，比如（221）则是一个有四个图，该图位于第一个

# import numpy as np
# import matplotlib.pyplot as plt
# x = np.arange(0, 100)
# #作图1
# plt.subplot(221)
# plt.plot(x, x)
# #作图2
# plt.subplot(222)
# plt.plot(x, -x)
#  #作图3
# plt.subplot(223)
# plt.plot(x, x ** 2)
# plt.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
# #作图4
# plt.subplot(224)
# plt.plot(x, np.log(x))
# plt.show()


#又或者是这样
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 100)
#作图1
plt.subplot(331)
plt.plot(x, x)
#作图2
plt.subplot(332)
plt.plot(x, -x)
 #作图3
plt.subplot(333)
plt.plot(x, x ** 2)
plt.grid(color='r', linestyle='--', linewidth=1,alpha=0.1)#最后一个表示线条粗细？
plt.show()
