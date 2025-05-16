# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
    ----------
    datafile_name: str
        path to data file

    K: int
        number of clusters to use

    feature_names: list
        list of feature_names

    Returns
    ---------
    fig: matplotlib.figure.Figure
        the figure object for the plot

    axs: matplotlib.axes.Axes
        the axes object for the plot
    """
    # ====> insert your code below here
    # Read data from file
    data = np.genfromtxt(datafile_name, delimiter=',')

    # Create K-Means cluster model with 10 initializations
    cluster_model = KMeans(n_clusters=K, n_init=10)
    cluster_model.fit(data)
    cluster_ids = cluster_model.predict(data)

    # Use original cluster_ids for visualization
    # Create canvas and axes
    num_feat = data.shape[1]
    fig, ax = plt.subplots(num_feat, num_feat, figsize=(12, 12))
    plt.set_cmap('viridis')

    # Get colors for histograms
    hist_col = plt.get_cmap('viridis', K).colors

    # Loop over features for visualization
    for feature1 in range(num_feat):
        # Set axis labels
        ax[feature1, 0].set_ylabel(feature_names[feature1])
        ax[0, feature1].set_xlabel(feature_names[feature1])
        ax[0, feature1].xaxis.set_label_position('top')

        for feature2 in range(num_feat):
            x_data = data[:, feature1].copy()
            y_data = data[:, feature2].copy()

            # sorting for scatter plots
            sorted_indices = np.argsort(cluster_ids)
            sorted_x_data = x_data[sorted_indices]
            sorted_y_data = y_data[sorted_indices]

            # Cluster reassignment for each subplot
            temp_ids = np.zeros_like(cluster_ids)
            for i in range(len(data)):
                distances = np.sum((data[i] - cluster_model.cluster_centers_) ** 2, axis=1)
                temp_ids[i] = np.argmin(distances)

            if feature1 != feature2:
                # Scatter plot with basic styling
                ax[feature1, feature2].scatter(sorted_x_data, sorted_y_data, c=temp_ids, cmap='viridis', s=50, marker='x')
            else:
                # Sort again for histograms
                inds = np.argsort(cluster_ids)
                sorted_y = cluster_ids[inds]
                sorted_x = x_data[inds]

                # Split data into clusters
                splits = np.split(sorted_x, np.unique(sorted_y, return_index=True)[1][1:])

                # Plot histogram
                for i, split in enumerate(splits):
                    ax[feature1, feature2].hist(split, bins=20, color=hist_col[i], edgecolor='black', alpha=0.7)

    # Set title with username
    username = "p4-yadav (Pradeep Yadav 24036544)"  # Replace with your UWE username
    fig.suptitle(f'Visualisation of {K} clusters by {username}', fontsize=16, y=0.925)

    # Save visualization
    fig.savefig('myVisualisation.jpg')

    return fig, ax
    # <==== insert your code above here
