# Fake News Classifier
## Data Set
The data set used to train this model can be found [here](https://www.kaggle.com/nopdev/real-and-fake-news-dataset). This data set consists of over 6,000 news articles that are all labeled either REAL or FAKE.

## Summary
We start by importing a data set of news articles into our project. We then split the data set into three parts such that 80% of the data is used for training, 10% of the data is used for testing, and 10% of the data is used for validation. Then, we clean the text data using some basic regular expression matching. After this, we preprocess all of the data for use with the BERT model. After these preparation steps, we begin to train and fine-tune the BERT model. This step is discussed further in the following section. Finally, we calculate the accuracy and performance of the model and display the statistics in a graph.

## Training
In this project, we choose to train the model for five epochs. This means we iterate over the entire training data five times. Each epoch, we split the data into batches of size 16. Then, as we process each batch, we keep track of and calculate statistics such as training loss and elapsed time. If we enable validation mode, we also keep track of validation loss and validation accuracy. For every batch, we perform both a forward pass and a backward pass. To keep track of progress (since training a model takes quite a bit of time), we print these statistics to the terminal every 100 batches.

## Statistics
Note that these statistics were generated using a random seed of 2021. After training on the data set, the BERT model attained an AUC of 0.9957, an F1 score of 0.97, and an accuracy of 97.16%. The ROC (Receiver Operating Characteristic) curve is also shown below.
![ROC curve](https://raw.githubusercontent.com/ishawng/fake-news-classifier/main/ROC.png)

## Conclusion
Based on the statistics in the previous section, we can make some inferences about our model and our data. Our AUC score is remarkably high, which implies that the data set is simply too easy for this model. The F1 score and the accuracy also support this conclusion. In general, the model does not seem to face any challenges distinguishing between the real and fake news articles in this data set. In the future, using a more challenging data set could help further evaluate and train the model.
