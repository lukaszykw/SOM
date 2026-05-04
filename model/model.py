import numpy as np


class KohonenNetwork:
    def __init__(self, grid_shape, input_dim, lambda_param, neighborhood_type='gaussian'):
        """
        Initialization of the Kohonen Network (SOM).

        Parameters:
        - grid_shape: Tuple (M, N) defining the dimensions of the neuron grid.
        - input_dim: The number of dimensions of the input data (e.g., 3 for x, y, z).
        - lambda_param: The lambda parameter for learning rate decay.
        - neighborhood_type: The type of neighborhood function ('gaussian' or 'mexican_hat').
        """
        self.M, self.N = grid_shape
        self.input_dim = input_dim
        self.lambda_param = lambda_param
        self.neighborhood_type = neighborhood_type

        # 1. Initialize neuron weights
        self.weights = np.random.rand(self.M, self.N, self.input_dim)

        # 2. Optimization: prepare grid coordinates
        x, y = np.mgrid[0:self.M, 0:self.N]
        self.grid_coords = np.column_stack([x.ravel(), y.ravel()])

    def _find_bmu(self, x):
        """
        Finds the Best Matching Unit (BMU) for a given input vector x.

        Parameters:
        - x: A 1D numpy array representing a single input sample.

        Returns:
        - A tuple (row, col) representing the coordinates of the BMU in the grid.
        """
        distances = np.linalg.norm(self.weights - x, axis=2)

        bmu_idx = np.unravel_index(np.argmin(distances), (self.M, self.N))

        return bmu_idx

    def _decay_learning_rate(self, t):
        """
        Calculates the decayed learning rate for the current iteration 't'.
        Formula used: alpha(t) = exp(-t / lambda)

        Parameters:
        - t: The current iteration (time step).

        Returns:
        - The decayed learning rate (float).
        """
        return np.exp(-t / self.lambda_param)

    def _calculate_neighborhood(self, bmu_coords, sigma):
        """
        Calculates the neighborhood function for all neurons in the grid.

        Parameters:
        - bmu_coords: Tuple (row, col) representing the BMU coordinates.
        - sigma: The current neighborhood radius (width of the function).

        Returns:
        - A 2D numpy array of shape (M, N) containing the neighborhood
          influence value (strength of learning) for each neuron.
        """
        # 1. Convert BMU coordinates to a numpy array for vectorized math
        bmu_coords_arr = np.array(bmu_coords)

        # 2. Calculate squared distances on the 2D grid
        squared_distances = np.sum((self.grid_coords - bmu_coords_arr) ** 2, axis=1)

        # Reshape the flat list of distances back into our M x N grid format
        squared_distances = squared_distances.reshape(self.M, self.N)

        # 3. Apply the selected neighborhood function
        if self.neighborhood_type == 'gaussian':
            influence = np.exp(-squared_distances / (2 * (sigma ** 2)))

        elif self.neighborhood_type == 'mexican_hat':
            term1 = 1 - (squared_distances / (sigma ** 2))
            term2 = np.exp(-squared_distances / (2 * (sigma ** 2)))
            influence = term1 * term2

            influence = np.clip(influence, -1.0, 1.0)

        else:
            raise ValueError("Invalid neighborhood_type. Choose 'gaussian' or 'mexican_hat'.")

        return influence

    def train(self, data, num_epochs, sigma):
        """
        Trains the Kohonen Network on the provided data.

        Parameters:
        - data: A 2D numpy array of shape (num_samples, input_dim).
        - num_epochs: Number of times to iterate over the entire dataset.
        - sigma: The neighborhood radius parameter.
        """
        current_iteration = 0
        indices = np.arange(len(data))

        for epoch in range(num_epochs):
            # 1. Shuffle data at the beginning of each epoch
            np.random.shuffle(indices)

            for i in indices:
                # Shuffled pick
                x = data[i]

                # 2. Calculate learning rate for the current time step
                alpha = self._decay_learning_rate(current_iteration)

                # 3. Find the Best Matching Unit (BMU) for input 'x'
                bmu_coords = self._find_bmu(x)

                # 4. Calculate neighborhood influence
                influence = self._calculate_neighborhood(bmu_coords, sigma)

                # 5. Update weights
                influence_expanded = influence[:, :, np.newaxis]

                # The core Kohonen learning rule:
                # W_new = W_old + alpha * neighborhood_influence * (Input - W_old)
                self.weights += alpha * influence_expanded * (x - self.weights)

                self.weights = np.clip(self.weights, data.min(), data.max())

                current_iteration += 1

    def predict(self, data):
        """
        Predicts the Best Matching Unit (BMU) for each sample in the provided dataset.
        This is used for evaluating the network after training.

        Parameters:
        - data: A 2D numpy array of shape (num_samples, input_dim).

        Returns:
        - A list of tuples, where each tuple contains the (row, col) coordinates
          of the BMU for the corresponding input sample.
        """
        # We iterate over all samples in the provided data
        # and use our previously defined _find_bmu method for each of them.
        predictions = [self._find_bmu(x) for x in data]

        return predictions