### ML9 - Repository for the Two Sigma kaggle competition
This is a code competition, so a little different from the typical competition.

Get the training dataset with the following command (in an object):
```
with pd.HDFStore("../input/train.h5", "r") as train:
  # Note that the "train" dataframe is the only dataframe in the file
  self.train = train.get("train")

```

Set up the kagglegym api environement with the following:

```
self.env = kagglegym.make()        
self.observation = self.env.reset()
```

From there, self.observations.features is a dataframe for your test features (x variables)

While self.observations.target contains the test id, and what your prediction is for test response(y variable)
