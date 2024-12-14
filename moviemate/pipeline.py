import pandas as pd
from sklearn.model_selection import train_test_split


class Pipeline:
    """
    A pipeline class for loading and partitioning the dataset.
    """

    def __init__(self):
        self.ratings_df = None
        self.items_df = None
        self.users_df = None

    def load_datasets(self, ratings_path, items_path=None, users_path=None):
        """
        Load the ratings dataset and optionally items and users datasets.

        Parameters:
        ----------
        ratings_path : str
            Path to the ratings dataset file.
        items_path : str, optional
            Path to the items dataset file.
        users_path : str, optional
            Path to the users dataset file.

        Returns:
        -------
        None
        """
        # ratings data
        self.ratings_df = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp']
        )
        print("Ratings Dataset Loaded:")
        print(self.ratings_df.info())
        print(self.ratings_df.describe())

        #  items data 
        if items_path:
            self.items_df = pd.read_csv(
                items_path,
                sep='|',
                encoding='latin-1',
                header=None,
                names=[
                    'item', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
                ]
            )
            print("Items Dataset Loaded:")
            print(self.items_df.info())

        # users data 
        if users_path:
            self.users_df = pd.read_csv(
                users_path,
                sep='|',
                encoding='latin-1',
                header=None,
                names=['user', 'age', 'gender', 'occupation', 'zip_code']
            )
            print("Users Dataset Loaded:")
            print(self.users_df.info())

    def partition_data(self, partition_type=None):
        """
        Partition the dataset into training and testing sets.

        Parameters:
        ----------
        partition_type : str, optional
            The type of partitioning to apply ('stratified' or 'temporal').

        Returns:
        -------
        pd.DataFrame, pd.DataFrame
            Training and testing datasets.
        """
        if self.ratings_df is None:
            raise ValueError("No ratings dataset loaded. Use `load_datasets` first.")

        if partition_type == 'stratified':
            train, test = train_test_split(
                self.ratings_df, test_size=0.2, stratify=self.ratings_df['user'], random_state=42
            )
        elif partition_type == 'temporal':
            self.ratings_df = self.ratings_df.sort_values('timestamp')
            split_index = int(len(self.ratings_df) * 0.8)
            train = self.ratings_df[:split_index]
            test = self.ratings_df[split_index:]
        else:
            raise ValueError("Invalid partition type. Choose 'stratified' or 'temporal'.")

        print(f"Training set size: {len(train)}")
        print(f"Testing set size: {len(test)}")
        return train, test


if __name__ == "__main__":
    pipeline = Pipeline()

    pipeline.load_datasets(
        ratings_path="storage/u.data",
        items_path="storage/u.item",
        users_path="storage/u.user"
    )

    train_set, test_set = pipeline.partition_data(partition_type='stratified')
