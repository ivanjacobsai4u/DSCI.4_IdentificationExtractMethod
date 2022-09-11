# DSCI.4_IdentificationExtractMethod

In order to successfully binary classify code fragments, we need to create a dataset consisting of positive and negative examples. Positive examples are code fragments that have undergone an Extract Method transformation, identified using RefactoringMiner tool . Negative examples are code fragments that are less likely to be extracted, identified by a ranking formula introduced by the work of Haas and Hummel [13].


The balanced dataset, consisting of equal number of positive and negative examples, is characterized by 78 metrics that have been extensively studied in previous studies. These metrics are going to be used as input attributes to the model. In order to train a binary classification model we label the positive and negative examples with “1” and “0” respectively.

![](assets/Code_duplicate_extractionCurrent.png)

The model, consisting of a Batch Normalization layer, 1D Convolutional layer, 1D Deconvolutional layer, Max Pooling layer with dropout and a Fully connected Dense layer, will consume the 78 metric values and produce a probabilistic decision about the whether the code fragment should be refactored or not.

![](assets/model.png)

In order to evaluate our work we define a set of research questions:
RQ1: How does our model perform compared to a machine learning baseline model?
RQ2: How does our model perform compared to other models?

