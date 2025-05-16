from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object

    ax: matplotlib.axes.Axes
        axis
    """

    # ====> insert your code below here
    # Define range of network architectures to test
    architectures = list(range(1, 11))

    # Initialize tracking variables
    successful_trials = np.zeros(10, dtype=int)
    iterations_needed = np.full((10, 10), 0)

    # Run experiments for each architecture
    for arch_idx, neurons in enumerate(architectures):
        for run in range(10):
            # Configure neural network
            network = MLPClassifier(
                hidden_layer_sizes=(neurons,),
                max_iter=1000,
                alpha=0.0001,
                solver="sgd",
                learning_rate_init=0.1,
                random_state=run
            )

            # Train network
            network.fit(train_x, train_y)

            # Evaluate performance
            prediction_accuracy = network.score(train_x, train_y)

            # Track perfect solutions
            if prediction_accuracy == 1.0:  # 100% accuracy
                successful_trials[arch_idx] += 1
                iterations_needed[arch_idx][run] = network.n_iter_

    # Calculate efficiency metrics
    avg_iterations = []
    for arch_idx in range(10):
        # Get only non-zero values (successful runs)
        successful_iters = [i for i in iterations_needed[arch_idx] if i > 0]

        # Calculate average or default to max iterations
        if successful_iters:
            avg_iterations.append(sum(successful_iters) / len(successful_iters))
        else:
            avg_iterations.append(1000)  # Default to max iterations if no successes

    # Generate visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot reliability (success rate)
    ax[0].plot(architectures, successful_trials, marker='o', color='blue')
    ax[0].set_title("Reliability")
    ax[0].set_xlabel("Hidden Layer Width")
    ax[0].set_ylabel("Success Rate")
    ax[0].set_xticks(architectures)

    # Plot efficiency (training iterations)
    ax[1].plot(architectures, avg_iterations, marker='o', color='green')
    ax[1].set_title("Efficiency")
    ax[1].set_xlabel("Hidden Layer Width")
    ax[1].set_ylabel("Mean Epochs")
    ax[1].set_xticks(architectures)

    # Optimize layout
    plt.tight_layout()
    # <==== insert your code above here

    return fig, ax

# make sure you have the packages needed
from approvedimports import *

#this is the class to complete where indicated
class MLComparisonWorkflow:
    """ class to implement a basic comparison of supervised learning algorithms on a dataset """

    def __init__(self, datafilename:str, labelfilename:str):
        """ Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.

        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models:dict = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index:dict = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy:dict = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        # Read data from files
        self.data_x = np.genfromtxt(datafilename, delimiter=",")
        self.data_y = np.genfromtxt(labelfilename, delimiter=",")
        # <==== insert your code above here

    def preprocess(self):
        """ Method to
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if there are more than 2 classes

           Remember to set random_state = 12345 if you use train_test_split()
        """
        # ====> insert your code below here
        # Split data into training (70%) and testing (30%)
        number_of_samples = len(self.data_y)
        train_size = int(0.7 * number_of_samples)

        # Use train_test_split to split data
        train_x, test_x, train_y, test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, stratify=self.data_y, random_state=12345
        )
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

        # Normalize the features manually (scale to 0-1)
        number_of_features = len(self.train_x[0])
        train_x_normalized = []
        test_x_normalized = []

        for sample in self.train_x:
            normalized_sample = []
            for feature_index in range(number_of_features):
                feature_values = [row[feature_index] for row in self.data_x]
                min_value = min(feature_values)
                max_value = max(feature_values)
                if max_value == min_value:
                    normalized_value = 0
                else:
                    normalized_value = (sample[feature_index] - min_value) / (max_value - min_value)
                normalized_sample.append(normalized_value)
            train_x_normalized.append(normalized_sample)

        for sample in self.test_x:
            normalized_sample = []
            for feature_index in range(number_of_features):
                feature_values = [row[feature_index] for row in self.data_x]
                min_value = min(feature_values)
                max_value = max(feature_values)
                if max_value == min_value:
                    normalized_value = 0
                else:
                    normalized_value = (sample[feature_index] - min_value) / (max_value - min_value)
                normalized_sample.append(normalized_value)
            test_x_normalized.append(normalized_sample)

        # Convert normalized lists to NumPy arrays for scikit-learn
        self.train_x = np.array(train_x_normalized)
        self.test_x = np.array(test_x_normalized)

        # Create one-hot encoded labels for MLP if more than 2 classes
        unique_labels = list(set(self.data_y))
        number_of_classes = len(unique_labels)

        if number_of_classes > 2:
            # One-hot encode training labels
            self.train_y_onehot = []
            for label in self.train_y:
                onehot = [0] * number_of_classes
                label_index = unique_labels.index(label)
                onehot[label_index] = 1
                self.train_y_onehot.append(onehot)

            # One-hot encode testing labels
            self.test_y_onehot = []
            for label in self.test_y:
                onehot = [0] * number_of_classes
                label_index = unique_labels.index(label)
                onehot[label_index] = 1
                self.test_y_onehot.append(onehot)

            # Convert one-hot lists to NumPy arrays
            self.train_y_onehot = np.array(self.train_y_onehot)
            self.test_y_onehot = np.array(self.test_y_onehot)
        else:
            self.train_y_onehot = self.train_y
            self.test_y_onehot = self.test_y
        # <==== insert your code above here

    def run_comparison(self):
        """ Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.

        For each of the algorithms KNearest Neighbour, DecisionTreeClassifier and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination,
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.
        """
        # ====> insert your code below here
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        # KNN: Try different k values
        k_values = [1, 3, 5, 7, 9]
        knn_model_index = 0
        for k in k_values:
            # Create and train KNN model
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(self.train_x, self.train_y)
            self.stored_models["KNN"].append(knn_model)

            # Calculate accuracy
            correct_predictions = 0
            total_predictions = len(self.test_y)
            predictions = knn_model.predict(self.test_x)
            for i in range(total_predictions):
                if predictions[i] == self.test_y[i]:
                    correct_predictions += 1
            accuracy = (correct_predictions / total_predictions) * 100

            # Update best KNN model
            if accuracy > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy
                self.best_model_index["KNN"] = knn_model_index
            knn_model_index += 1

        # Decision Tree: Try different combinations
        max_depth_values = [1, 3, 5]
        min_split_values = [2, 5, 10]
        min_leaf_values = [1, 5, 10]
        dt_model_index = 0
        for depth in max_depth_values:
            for split in min_split_values:
                for leaf in min_leaf_values:
                    # Create and train Decision Tree model
                    dt_model = DecisionTreeClassifier(
                        max_depth=depth, min_samples_split=split, min_samples_leaf=leaf, random_state=12345
                    )
                    dt_model.fit(self.train_x, self.train_y)
                    self.stored_models["DecisionTree"].append(dt_model)

                    # Calculate accuracy
                    correct_predictions = 0
                    total_predictions = len(self.test_y)
                    predictions = dt_model.predict(self.test_x)
                    for i in range(total_predictions):
                        if predictions[i] == self.test_y[i]:
                            correct_predictions += 1
                    accuracy = (correct_predictions / total_predictions) * 100

                    # Update best Decision Tree model
                    if accuracy > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy
                        self.best_model_index["DecisionTree"] = dt_model_index
                    dt_model_index += 1

        # MLP: Try different combinations
        first_layer_sizes = [2, 5, 10]
        second_layer_sizes = [0, 2, 5]
        activation_functions = ["logistic", "relu"]
        mlp_model_index = 0
        for first_size in first_layer_sizes:
            for second_size in second_layer_sizes:
                for activation in activation_functions:
                    # Set hidden layer sizes
                    if second_size == 0:
                        hidden_layers = (first_size,)
                    else:
                        hidden_layers = (first_size, second_size)

                    # Create and train MLP model
                    mlp_model = MLPClassifier(
                        hidden_layer_sizes=hidden_layers, activation=activation, max_iter=1000, random_state=12345
                    )
                    mlp_model.fit(self.train_x, self.train_y_onehot)
                    self.stored_models["MLP"].append(mlp_model)

                    # Calculate accuracy
                    correct_predictions = 0
                    total_predictions = len(self.test_y)
                    predictions = mlp_model.predict(self.test_x)
                    number_of_classes = len(set(self.data_y))
                    if number_of_classes > 2:
                        # Convert one-hot predictions to labels
                        for i in range(total_predictions):
                            predicted_label_index = 0
                            max_value = predictions[i][0]
                            for j in range(1, number_of_classes):
                                if predictions[i][j] > max_value:
                                    max_value = predictions[i][j]
                                    predicted_label_index = j
                            true_label_index = 0
                            max_value = self.test_y_onehot[i][0]
                            for j in range(1, number_of_classes):
                                if self.test_y_onehot[i][j] > max_value:
                                    max_value = self.test_y_onehot[i][j]
                                    true_label_index = j
                            if predicted_label_index == true_label_index:
                                correct_predictions += 1
                    else:
                        # For binary classification, use direct comparison
                        for i in range(total_predictions):
                            if predictions[i] == self.test_y[i]:
                                correct_predictions += 1
                    accuracy = (correct_predictions / total_predictions) * 100

                    # Update best MLP model
                    if accuracy > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy
                        self.best_model_index["MLP"] = mlp_model_index
                    mlp_model_index += 1
        # <==== insert your code above here

    def report_best(self):
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN","DecisionTree" or "MLP"

        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        # ====> insert your code below here
        # Find the best model among all algorithms
        highest_accuracy = 0
        best_algorithm = ""
        best_model = None
        algorithms = ["KNN", "DecisionTree", "MLP"]
        for algorithm in algorithms:
            if self.best_accuracy[algorithm] > highest_accuracy:
                highest_accuracy = self.best_accuracy[algorithm]
                best_algorithm = algorithm
                best_model = self.stored_models[algorithm][self.best_model_index[algorithm]]
        return highest_accuracy, best_algorithm, best_model
        # <==== insert your code above here
