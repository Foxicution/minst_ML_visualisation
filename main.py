from streamlit import markdown
from matplotlib import pyplot as plt
from keras.datasets import mnist

if __name__ == '__main__':
    (trainX, trainy), (testX, testy) = mnist.load_data()
    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()
    # markdown("Hello **world**!")
