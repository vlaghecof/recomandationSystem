"# recomandationSystem" 
import pandas as pd
from surprise import Dataset
from surprise import Reader
import matplotlib.pyplot as plt

fifa=1
minecraft=2

ratings_dict = {
    "item": [fifa,minecraft, fifa, minecraft, fifa, minecraft, fifa,minecraft, fifa],
    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
}


# df = pd.DataFrame(ratings_dict)
# df.to_csv('csv_example')

df = pd.read_csv('csv_example')
print(df)


reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)


# movielens = Dataset.load_builtin('ml-100k')

from surprise import KNNWithMeans

# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}
algo = KNNWithMeans(sim_options=sim_options)


trainingSet = data.build_full_trainset()
algo.fit(trainingSet)

prediction = algo.predict('E', fifa)


print (prediction.est)
