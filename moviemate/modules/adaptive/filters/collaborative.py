import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


class CollaborativeFiltering:
    """
    A collaborative filtering recommender system.

    This class supports:
    - Matrix factorization (SVD).
    - User-User Collaborative Filtering.
    - Item-Item Collaborative Filtering.
    """

    def __init__(self, ratings_file, algorithm="SVD", test_size=0.2, random_state=42):
        """
        Initialize the recommender system and load data.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file.
        algorithm : str, optional
            Algorithm type: "SVD", "user", or "item".
        test_size : float, optional
            Proportion of the dataset to include in the test split. Default is 0.2.
        random_state : int, optional
            Random seed for reproducibility. Default is 42.
        """
        self.algorithm = algorithm
        self.test_size = test_size
        self.random_state = random_state
        self.trainset = None
        self.testset = None
        self.model = None
        self.data_file = ratings_file
        self.data = None
        self._load_data()

    def _load_data(self):
        """Load the dataset and prepare data for training."""
        df = pd.read_csv(self.data_file, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
        self.data = Dataset.load_from_df(df[['user', 'item', 'rating']], Reader(rating_scale=(1, 5)))

        self.trainset, self.testset = train_test_split(self.data, test_size=self.test_size, random_state=self.random_state)

    def fit(self):
        """Fit the model on the training set."""
        if self.trainset is None:
            raise ValueError("Data not loaded. Ensure the data file is correctly loaded.")

        if self.algorithm == "SVD":
            #  SVD
            self.model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        elif self.algorithm == "user":
            # User-User CF
            sim_options = {"name": "cosine", "user_based": True}
            self.model = KNNBasic(sim_options=sim_options)
        elif self.algorithm == "item":
            # Item-Item CF
            sim_options = {"name": "cosine", "user_based": False}
            self.model = KNNBasic(sim_options=sim_options)
        else:
            raise ValueError(f"Invalid algorithm type: {self.algorithm}. Choose from 'SVD', 'user', or 'item'.")

        self.model.fit(self.trainset)

    def evaluate(self):
        """
        Evaluate the model using RMSE on the test set.

        Returns
        -------
        float
            Root Mean Square Error (RMSE) of the predictions.
        """
        if self.testset is None:
            raise ValueError("Data not loaded. Call `_load_data` first.")
        predictions = self.model.test(self.testset)
        return accuracy.rmse(predictions)

    def predict(self, user_id, item_id):
        """
        Predict the rating for a given user and item.

        Parameters
        ----------
        user_id : int
            ID of the user.
        item_id : int
            ID of the item.

        Returns
        -------
        float
            Predicted rating for the user-item pair.
        """
        return self.model.predict(user_id, item_id).est


if __name__ == "__main__":
    ratings_file = "storage/u.data"

    # SVD Model
    print("Evaluating SVD Model")
    svd_model = CollaborativeFiltering(ratings_file, algorithm="SVD")
    svd_model.fit()
    svd_rmse = svd_model.evaluate()
    print(f"SVD Model RMSE: {svd_rmse}")

    # User-User CF
    print("\nEvaluating User-User Collaborative Filtering Model")
    user_model = CollaborativeFiltering(ratings_file, algorithm="user")
    user_model.fit()
    user_rmse = user_model.evaluate()
    print(f"User-User Model RMSE: {user_rmse}")

    # Item-Item CF
    print("\nEvaluating Item-Item Collaborative Filtering Model")
    item_model = CollaborativeFiltering(ratings_file, algorithm="item")
    item_model.fit()
    item_rmse = item_model.evaluate()
    print(f"Item-Item Model RMSE: {item_rmse}")

    # Example 
    user_id = 196  
    item_id = 242  
    predicted_rating = item_model.predict(user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")
