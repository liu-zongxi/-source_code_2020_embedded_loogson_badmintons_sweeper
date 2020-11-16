# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
#matplotlib是一个python的2D绘图库
import matplotlib.pyplot as plt

print(tf.__version__)
#该数据集在keras中已存在可直接下载，=左边自由命名
fashion_mnist = keras.datasets.fashion_mnist
#调用load。data获得训练数据标签和测试数据标签,加载数据集将返回4个numpy数组
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#打印格式，数量等
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(train_labels)
#图片名字（注意引号），尺寸大小，像素，底色，边框颜色（看不见？？）
plt.figure(num='ha', figsize=(10,10), dpi=50, facecolor='white',frameon=None)
#plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示。其后跟着plt.show()才能显示出来。
plt.imshow(train_images[0])
#在图片旁边显示颜色条
plt.colorbar()
#是否显示坐标轴，True=1=显示=默认，Fasle=0=不显示，后面加axis='x/y'表示只显示x轴或y轴
plt.grid(False)
#将图片显示
plt.show()
#图像有0-255的值，但神经网络在标准化数据工作更好，除255可使其变为0至1之间的值
train_images = train_images / 255.0
test_images = test_images / 255.0
#为了验证数据的格式正确，并准备好构建和训练网络，让我们显示训练集中的前25个图像，并在每个图像下方显示类名。
plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)#分成5行5列，选图中第i+1个图
    #第一个中括号为x轴上的值，第二个中括号里为替代第一个中括号里的值
    plt.xticks([])#[1,10,20],['hello','hi','friend']

    plt.yticks([])
    plt.grid(1)
    #camp为绘图样式，例如gray为灰度图，binary为两端发散的色图，seismic为离散色图
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    plt.ylabel('hi')
plt.show()
#一下两行写得是干嘛的？屏蔽错误的输出？等级越高只输出越严重的错误
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'#默认下载了X86_64的SIMD版本，选择忽略报错，应删除并重新下载avx2

# #以下设计模型（1）
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),#扁平层，因为图像的输入为28*28像素，扁平化使其变为28*28的像素图
#     keras.layers.Dense(100, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)#数据集有10类服饰，所以为10
# ])
#一下设计模型（2）
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# #编译模型寻找损失函数和优化器。优化器的理解：是用来一步步优化参数W和b的
#编译（1）有误
# model.compile(optimizer=tf.optimizers.Adam(),
#               loss="sparse_categorical_crossentropy")
#编译（2）
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
              #accuracy
model.fit(train_images, train_labels, epochs=5)
#verbose可为0，1，2；0为不在标准输出流输出日志信息；1为输出进度条记录， 2为每个epoch输出一行记录；默认为 1（不显示？）
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
print('\nTest accuracy:', test_acc)
#With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits.
# Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

  # Plot the first X test images, their predicted labels, and the true labels.
  # Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
 plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
 plot_image(i, predictions[i], test_labels, test_images)
 plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
 plot_value_array(i, predictions[i], test_labels)
 plt.tight_layout()
 plt.show()

  # Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])
