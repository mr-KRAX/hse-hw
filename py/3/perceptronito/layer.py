import numpy as np
import perceptronito.activation_functions as actfn


class BaseLayer:
  """BaseLayer
  
  Base layer class for perceptron
  """

  activations = {
      'sigmoid': actfn.Sigmoid(),
      'tanh':    actfn.TanH(),
      'Relu':    actfn.ReLU(),
  }

  def __init__(self, n_units: int, activation: str = 'sigmoid'):
    assert activation in self.activations.keys()

    self.n_units = n_units
    self.f = BaseLayer.activations[activation]

  def build(self, n_inputs: int, next_layer=None) -> None:
    self.next_l: BaseLayer = next_layer

  def _process_impl(self, inputs: np.ndarray) -> np.ndarray:
    raise NotImplementedError('_process_impl should be implemented')

  def process(self, inputs: np.ndarray) -> np.ndarray:
    return self.f(self._process_impl(inputs))

  def backpropagate(self, inputs: np.ndarray, y: np.ndarray) -> (np.ndarray, actfn.ActivationFunction):
    raise NotImplementedError(
        'Probably the layer type has no back-propagation mechanism')


class DenseLayer(BaseLayer):
  """DenseLayer
  
  Simple dense layer class
  """

  def __init__(self, n_units: int, activation: str = 'sigmoid', learning_rate: float = 0.01):
    super().__init__(n_units=n_units, activation=activation)
    self.lrate = learning_rate

  def build(self, n_inputs: int, next_layer: BaseLayer = None):
    super().build(next_layer)

    self.w: np.ndarray = np.random.rand(n_inputs, self.n_units)
    self.bias: np.ndarray = np.zeros(self.n_units)

  def _process_impl(self, inputs: np.ndarray) -> np.ndarray:
    return inputs.dot(self.w) + self.bias

  def backpropagate(self, inputs: np.ndarray, y: np.ndarray) -> (np.ndarray, actfn.ActivationFunction):
    """backpropagate

    Args:
        inputs (np.ndarray): one sample of train data or result from previous layer
        y (np.ndarray): one sample of train target result
    Returns:
        np.ndarray: delta error for prev layer
        actfn.ActivationFunction: activation function
    """

    z = self._process_impl(inputs)
    if self.next_l is None:
      delta = (y - self.f(z)) + self.f.gradient(z)
    else:
      (next_layer_delta, next_layer_f) = self.next_l.backpropagate(self.f(z), y)
      delta = next_layer_delta * next_layer_f.gradient(z)
    self.w += self.lrate * np.outer(inputs, delta)
    self.bias = self.lrate * delta

    return (delta.dot(self.w.T), self.f)
