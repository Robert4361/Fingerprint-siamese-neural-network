import random
import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt
from make_pairs import get_pairs, remove_names
from sklearn.utils import shuffle

epochs = 10
batch_size = 16
margin = 1

matching_pairs, non_matching_pairs = get_pairs()
matching_pairs_no_name = remove_names(matching_pairs)
non_matching_pairs_no_name = remove_names(non_matching_pairs)

labels1 = np.repeat(0.0, 4160)
labels2 = np.repeat(1.0, 4160)


labels = np.concatenate((labels1, labels2))

added_pairs = np.concatenate((matching_pairs_no_name, non_matching_pairs_no_name)) 

added_pairs, labels = shuffle(added_pairs, labels, random_state=0)

labels_train = labels[3000:]
labels_test = labels[1500:3000]
labels_val = labels[0:1500]


pairs_train = added_pairs[3000:]
pairs_test = added_pairs[1500:3000]
pairs_val = added_pairs[0:1500]

x_train_1 = pairs_train[:, 0] 
x_train_2 = pairs_train[:, 1]

x_val_1 = pairs_val[:, 0] 
x_val_2 = pairs_val[:, 1]

x_test_1 = pairs_test[:, 0]
x_test_2 = pairs_test[:, 1]


def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    num_row = to_show // num_col if to_show // num_col != 0 else 1
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):
        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(ops.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

# print(type(matching_pairs[0][0]))
#import cv2
#cv2.imshow(matching_pairs[2][0].name, matching_pairs[2][0].img)
# cv2.imshow(matching_pairs[2][1].name, matching_pairs[2][1].img)
# visualize(matching_pairs_no_name, matching_labels, 4, 2)

#cv2.waitKey()
    
# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):

    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))


input = keras.layers.Input((143, 130, 1))
x = keras.layers.BatchNormalization()(input)
x = keras.layers.Conv2D(8, (5, 5), activation="tanh")(x)
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = keras.layers.Conv2D(16, (5, 5), activation="tanh")(x)
x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
x = keras.layers.Flatten()(x)

x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(10, activation="tanh")(x)
embedding_network = keras.Model(input, x)


input_1 = keras.layers.Input((143, 130, 1))
input_2 = keras.layers.Input((143, 130, 1))

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = keras.layers.Lambda(euclidean_distance, output_shape=(1,))(
    [tower_1, tower_2]
)
normal_layer = keras.layers.BatchNormalization()(merge_layer)
output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
def loss(margin=1):

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return contrastive_loss

siamese.compile(loss=loss(margin=margin), optimizer="Adam", metrics=["accuracy"])
siamese.summary()

history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
)


siamese.save('my_model.keras')

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the contrastive loss
plt_metric(history=history.history, metric="loss", title="Contrastive Loss")

results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)

predictions = siamese.predict([x_test_1, x_test_2])
visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)