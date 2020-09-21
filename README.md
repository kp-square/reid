Code is in the colab notebook. A link to the copy is 
https://colab.research.google.com/drive/1kT4ve4Ft1b-EJgl9Lxo4DqPyyj5ZjJUp?usp=sharing 

To train the model, data need to be downloaded to the colab notebook from kaggle.
Link to the dataset is https://www.kaggle.com/pengcw1/market-1501

Code is implemented in tensorflow 2.3.0 and uses EfficientNetB0 as 
backend with imagenet weights.

Loss = categorical_cross_entropy + center_loss + triplet_loss + osm_caa_loss

