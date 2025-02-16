import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes: np.array=[784, 64, 10], acti_func: str='reLU'):
        '''
            Sources:
            Theory: http://neuralnetworksanddeeplearning.com/chap1.html
            Implementation (primarily the indices): https://github.com/lionelmessi6410/Neural-Networks-from-Scratch/blob/main/model.py
        '''
        self.params: dict = {
            'w1': np.random.randn(layer_sizes[1], layer_sizes[0]),
            'w2': np.random.randn(layer_sizes[2], layer_sizes[1]),
            'b1': np.random.randn(layer_sizes[1], 1),
            'b2': np.random.randn(layer_sizes[2], 1),
        }
        self.perc: dict = {}
        self.acti_func: function = self.reLU if acti_func == 'reLU' else self.sigmoid
    
    def sigmoid(self, x: np.array, derivative: bool=False) -> np.array:
        '''
            σ(x) = (1 + e^-x)^-1
            σ′(x) = e^-x / (1 + e^-x)^2
        '''
        if derivative:
            return np.exp(-x) / np.square(1 + np.exp(-x))
        return 1 / (1 + np.exp(-x))

    def reLU(self, x: np.array, derivative: bool=False) -> np.array:
        '''
            ReLU(x) = max(0, x)
            ReLU′(x) = 0 if x < 0, 1 if x > 0
        '''
        if derivative:
            return (x > 0).astype(int)
        return np.maximum(0, x)

    def soft_max(self, x: np.array) -> np.array:
        '''
            Convert -∞, +∞ to probabilities
            σ(x)_i = e^z_i / Σ e^z
        '''
        exps: np.array = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def forward(self, x: np.array) -> np.array:
        '''
            Forward step of NN
        '''
        self.perc['x'] = x
        self.perc['z1'] = self.params['w1'] @ self.perc['x'].T + self.params['b1']
        self.perc['a1'] = self.acti_func(self.perc['z1'])
        self.perc['z2'] = self.params['w2'] @ self.perc['a1'] + self.params['b2']
        self.perc['a2'] = self.soft_max(self.perc['z2'])
        return self.perc['a2']

    def backward(self, output: np.array, y: np.array) -> dict:
        '''
            Calculates the partial derrivates of the Softmax loss function (= Categorical Cross Entropy) using backpropagation
            Sources:
            https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
        '''
        dz2 = output - y.T
        da1 = self.params['w2'].T @ dz2
        dz1 = da1 * self.acti_func(self.perc['z1'], derivative=True)
        
        grad: dict = {
            'w1': dz1 @ self.perc['x'],
            'w2': dz2 @ self.perc['a1'].T,
            'b1': np.sum(dz1, axis=1, keepdims=True),
            'b2': np.sum(dz2, axis=1, keepdims=True),
        }
        return grad

    def SGD(self, grad: dict, batch_size: int, l_rate: float):
        '''
            Calculate the unbiased estimator of GD
            w′ = w − η / n * ∂C / ∂w
            b′ = b − η / n * ∂C / ∂b
        '''
        for param in self.params:
            self.params[param] -= l_rate / batch_size * grad[param]

    def one_hot_encode(self, y: np.array, decode: bool=False) -> np.array:
        if decode:
            return np.argmax(y, axis=1)
        y_hot = np.zeros((y.size, y.max() + 1))
        y_hot[np.arange(y.size), y] = 1
        return y_hot

    def accuracy(self, x: np.array, y: np.array) -> float:
        pred: np.array = np.argmax(x.T, axis=1)
        return np.sum(pred == y) / pred.shape[0]

    def train(
        self,
        x_train: np.array,
        y_train: np.array,
        epochs: int=30,
        batch_size: int=64,
        l_rate: float=0.1
    ) -> None:
        num_batches: int = x_train.shape[0] // batch_size
        y_train = self.one_hot_encode(y_train)
        for epoch in range(epochs):
            permu: np.array = np.random.permutation(x_train.shape[0])
            x_train, y_train = x_train[permu], y_train[permu]
            for batch in range(num_batches):
                start: int = batch * batch_size
                end: int = min(start + batch_size, x_train.shape[0] - 1)
                x, y = x_train[start:end], y_train[start:end]

                output: np.array = self.forward(x)
                grad: np.array = self.backward(output, y)
                self.SGD(grad, batch_size, l_rate)
            print(f'Epoch = {epoch}, accuracy = {self.accuracy(self.forward(x_train), self.one_hot_encode(y_train, decode=True))}')

    def evaluate(self, x_test: np.array, y_test: np.array) -> None:
        print(f'Accuracy = {self.accuracy(self.forward(x_test), y_test)}')

    def plot_prediction(self, x: np.array) -> None:
        labels: list[str] = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        pred: int = np.argmax(self.forward(np.reshape(x, (1, 784))).T, axis=1)
        plt.title(labels[pred[0]])
        plt.imshow(np.reshape(x, (28, 28)), cmap="gray")
        plt.show()

if __name__ == "__main__":
    import mnist_reader
    x_train, y_train = mnist_reader.load_mnist('data', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data', kind='t10k')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    nn: NeuralNetwork = NeuralNetwork(acti_func='sigmoid')
    nn.train(x_train, y_train)
    nn.evaluate(x_test, y_test) # ~0.82
    nn.plot_prediction(x_test[np.random.uniform(0, x_test.shape[0], 1).astype(int)[0]])
    nn.plot_prediction(x_test[np.random.uniform(0, x_test.shape[0], 1).astype(int)[0]])
    nn.plot_prediction(x_test[np.random.uniform(0, x_test.shape[0], 1).astype(int)[0]])