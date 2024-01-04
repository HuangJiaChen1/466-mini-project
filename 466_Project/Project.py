import os.path

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.python.keras import models, layers
# load data
cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train, x_test  = x_train / 255.0, x_test / 255.0
plt.imshow(x_train[0])
plt.savefig("Sample.jpg")
plt.show()
# the number of datas
print("x train shape: ",x_train.shape) # 50000 data, 32*32 pixels, 3 channel RGB
print("y train shape: ",y_train.shape) # 50000 data
print("x test shape: ",x_test.shape) # 10000 data, 32*32 pixels, 3 channel RGB
print("y test shape: ", y_test.shape) # 10000 data

#Baseline
acc = 0
for i in range(len(x_test)):
    output = np.random.randint(0,10)
    if output == y_test[i]:
        acc +=1
acc /= y_test.shape[0]
print(f'Test accuracy of a simple baseline: {acc}')





# I will use CNN for feature selection, and a 2 layer Dense network(Dense128, Dense10)for training
# Define the CNN model
def create_CNN(num_layers, dense_units, dropout_rate):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    for _ in range(num_layers):
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

layer_combinations = [1, 2]
unit_combinations = [64, 128]
dropout_combinations = [0.3, 0.5]

best_accuracy = 0
best_model = None
best_lud = (0,0,0)
for num_layers in layer_combinations:
    for dense_units in unit_combinations:
        for dropout_rate in dropout_combinations:
            model = create_CNN(num_layers, dense_units, dropout_rate)
            history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
            val_accuracy = max(history.history['val_sparse_categorical_accuracy'])
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                best_lud = (num_layers,dense_units,dropout_rate)
                acc = history.history["sparse_categorical_accuracy"]
                val_acc = history.history["val_sparse_categorical_accuracy"]
                loss = history.history["loss"]
                val_loss = history.history["val_loss"]
best_model.summary()



print('The best hyperparameters are:\n' + f'{best_lud[0]} of layers')
print(f'{best_lud[1]} of dense units')
print(f'{best_lud[2]} dropout rate')
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
#plot
plt.subplot(1,2,1)
plt.plot(acc,label = "Training accuracy")
plt.plot(val_acc, label = "Validation accuracy")
plt.title("Accuracy in Training and Validation")
plt.legend()
plt.subplot(1,2,2)
plt.plot(loss, label = "Training Loss")
plt.plot(val_loss , label= "Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("Visualize_CNN.jpg")
plt.show()







def DenseNN(num_layers, dense_units, dropout_rate):
    model = models.Sequential()
    model.add(layers.Flatten())
    for _ in range(num_layers):
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

layer_combinations = [1, 2]
unit_combinations = [64, 128]
dropout_combinations = [0.3, 0.5]

best_accuracy = 0
best_model = None
best_lud = (0,0,0)
for num_layers in layer_combinations:
    for dense_units in unit_combinations:
        for dropout_rate in dropout_combinations:
            model = DenseNN(num_layers, dense_units, dropout_rate)
            history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
            val_accuracy = max(history.history['val_sparse_categorical_accuracy'])
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                best_lud = (num_layers,dense_units,dropout_rate)
                acc = history.history["sparse_categorical_accuracy"]
                val_acc = history.history["val_sparse_categorical_accuracy"]
                loss = history.history["loss"]
                val_loss = history.history["val_loss"]
best_model.summary()
print('The best hyperparameters are:\n' + f'{best_lud[0]} of layers')
print(f'{best_lud[1]} of dense units')
print(f'{best_lud[2]} dropout rate')
test_loss2, test_acc2 = best_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc2}')

plt.subplot(1,2,1)
plt.plot(acc,label = "Training accuracy")
plt.plot(val_acc, label = "Validation accuracy")
plt.title("Accuracy in Training and Validation")
plt.legend()
plt.subplot(1,2,2)
plt.plot(loss, label = "Training Loss")
plt.plot(val_loss , label= "Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("Visualize_Dense.jpg")
plt.show()







# RF
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=100)

n_trees = [50,100]
max_depths = [10,20]
best_n_tree = None
best_depth = None
max_val = -10000
best_RF_class = None
for n_tree in n_trees:
    for max_depth in max_depths:
        rf_classifier = RandomForestClassifier(n_estimators=n_tree, max_depth=max_depth,random_state=100)
        rf_classifier.fit(x_train, y_train)
        val_predictions = rf_classifier.predict(x_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(val_accuracy)
        if val_accuracy > max_val:
            max_val = val_accuracy
            best_n_tree = n_tree
            best_depth = max_depth
            best_RF_class = rf_classifier

print(f'The best hyperparameters are: {best_n_tree} trees, {best_depth} depth')
print(f"Validation accuracy: {val_accuracy}")

test_predictions = best_RF_class.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test accuracy: {test_accuracy}")