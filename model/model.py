import numpy as np


class KohonenNetwork:
    def __init__(self, grid_shape, input_dim, lambda_param, neighborhood_type='gaussian', topology='rectangular', normalize_data=False):
        """
        Initialization of the Kohonen Network (SOM).

        Parameters:
        - grid_shape: Tuple (M, N) defining the dimensions of the neuron grid.
        - input_dim: The number of dimensions of the input data (e.g., 3 for x, y, z).
        - lambda_param: The lambda parameter for learning rate decay.
        - neighborhood_type: The type of neighborhood function ('gaussian' or 'mexican_hat').
        """
        self.M, self.N = grid_shape # Left for other functions
        self.grid_shape = grid_shape
        self.input_dim = input_dim
        self.lambda_param = lambda_param
        self.neighborhood_type = neighborhood_type
        self.topology = topology

        self.normalize_data = normalize_data
        self.data_min = None
        self.data_max = None

        self.weights = np.random.rand(self.grid_shape[0], self.grid_shape[1], self.input_dim)

        self._precompute_grid_coordinates()

    def _precompute_grid_coordinates(self):
        """
        Precomputes the physical (x, y) spatial coordinates for each neuron
        based on the selected topology.
        """
        M, N = self.grid_shape
        # Create a grid of shape (M, N, 2) to store x and y for each neuron
        self.spatial_coords = np.zeros((M, N, 2))

        for i in range(M):
            for j in range(N):
                if self.topology == 'hexagonal':
                    # Shift odd rows by 0.5 and scale y by sqrt(3)/2
                    x = j + 0.5 * (i % 2)
                    y = i * (np.sqrt(3) / 2)
                else:
                    # Standard rectangular grid
                    x = j
                    y = i
                self.spatial_coords[i, j] = [x, y]

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
        # Get the physical (x, y) coordinates of the BMU
        bmu_spatial = self.spatial_coords[bmu_coords[0], bmu_coords[1]]

        # Calculate squared Euclidean distances between BMU and all other neurons' spatial positions
        squared_distances = np.sum((self.spatial_coords - bmu_spatial) ** 2, axis=2)

        if self.neighborhood_type == 'gaussian':
            return np.exp(-squared_distances / (2 * (sigma ** 2)))

        elif self.neighborhood_type == 'mexican_hat':
            s2 = sigma ** 2
            term1 = 1 - (squared_distances / s2)
            term2 = np.exp(-squared_distances / (2 * s2))
            influence = term1 * term2
            return np.clip(influence, -0.1, 1.0)

    def train(self, data, num_epochs, sigma):
        """
        Trains the Kohonen Network on the provided data.

        Parameters:
        - data: A 2D numpy array of shape (num_samples, input_dim).
        - num_epochs: Number of times to iterate over the entire dataset.
        - sigma: The neighborhood radius parameter.
        """
        training_data = data.copy()

        if self.normalize_data:
            self.data_min = training_data.min(axis=0)
            self.data_max = training_data.max(axis=0)

            range_values = self.data_max - self.data_min
            range_values[range_values == 0] = 1e-8

            training_data = (training_data - training_data.min()) / range_values

        current_iteration = 0
        indices = np.arange(len(training_data))

        for epoch in range(num_epochs):
            # 1. Shuffle data at the beginning of each epoch
            np.random.shuffle(indices)

            for i in indices:
                # Shuffled pick
                x = training_data[i]

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
                clip_min = 0.0 if self.normalize_data else training_data.min()
                clip_max = 1.0 if self.normalize_data else training_data.max()

                self.weights = np.clip(self.weights, clip_min, clip_max)

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
        eval_data = data.copy()

        if self.normalize_data:
            if self.data_min is None or self.data_max is None:
                raise ValueError('Model must be trained before calling predict()')

            range_values = self.data_max - self.data_min
            range_values[range_values == 0] = 1e-8
            eval_data = (eval_data - eval_data.min()) / range_values

        bmu_predictions = []
        for x in eval_data:
            bmu_predictions.append(self._find_bmu(x))

        return bmu_predictions