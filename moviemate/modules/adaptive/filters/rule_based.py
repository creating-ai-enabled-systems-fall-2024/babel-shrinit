import pandas as pd


class RuleBasedRecommender:
    """
    A rule-based recommender system that suggests top-rated movies.
    """

    def __init__(self, ratings_file, metadata_file):
        """
        Initialize the recommender system.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset.
        metadata_file : str
            Path to the metadata file containing movie genres.
        """
        self.ratings = pd.read_csv(ratings_file, sep='\t', names=['user', 'item', 'rating', 'timestamp'])
        self.metadata = pd.read_csv(
            metadata_file,
            sep='|',
            encoding='latin-1',
            names=[
                'item', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
        )

    def recommend_top_movies(self, top_n=10):
        """
        Recommend top-rated movies overall.

        Parameters
        ----------
        top_n : int
            Number of top movies to recommend.

        Returns
        -------
        pd.DataFrame
            Top-rated movies and their average ratings.
        """
        movie_ratings = self.ratings.groupby('item')['rating'].mean()
        top_movies = movie_ratings.nlargest(top_n).reset_index()
        return pd.merge(top_movies, self.metadata[['item', 'title']], on='item')

    def recommend_by_genre(self, genre, top_n=10):
        """
        Recommend top-rated movies in a specific genre.

        Parameters
        ----------
        genre : str
            Genre to filter movies by.
        top_n : int
            Number of top movies to recommend.

        Returns
        -------
        pd.DataFrame
            Top-rated movies in the given genre.
        """
        genre_movies = self.metadata[self.metadata[genre] == 1]
        genre_ratings = self.ratings[self.ratings['item'].isin(genre_movies['item'])]
        avg_ratings = genre_ratings.groupby('item')['rating'].mean()
        top_movies = avg_ratings.nlargest(top_n).reset_index()
        return pd.merge(top_movies, genre_movies[['item', 'title']], on='item')


if __name__ == "__main__":
    recommender = RuleBasedRecommender(ratings_file="storage/u.data", metadata_file="storage/u.item")
    print("Top Movies Overall:")
    print(recommender.recommend_top_movies())
    print("\nTop Movies in Comedy Genre:")
    print(recommender.recommend_by_genre("Comedy"))
