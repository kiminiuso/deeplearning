from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 10
numpy.random.seed(seed)

#加载数据
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 区分前8个输入和第九个输出
X = dataset[0:len(dataset), 0:8]
Y = dataset[0:len(dataset), 8]

# 创建模型，包含激活函数、输入类型
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu',kernel_initializer="uniform"))
model.add(Dense(8, activation='relu', kernel_initializer="uniform"))
model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 开始训练
# 手动输入训练集
# model.train_on_batch(x_batch, y_batch)
model.fit(X, Y, epochs=200, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x) for x in predictions.ravel()]
print(rounded)
