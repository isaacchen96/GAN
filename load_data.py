from tensorflow.keras.datasets import mnist

def load_mnist(with_label = False):
    if with_label:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (x_train, _), (x_test, _) = mnist.load_data()
    x_train = normalized(x_train)
    x_test  = normalized(x_test)

    if with_label:
        return (x_train, y_train), (x_test, y_test)
    else:
        return x_train, x_test

def normalized(data):
    data = data.astype('float32')/255
    return (data*2)-1


