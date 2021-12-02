# Draft That Win
Small project on estimating matches outcomes in Dota 2 based on heroes' draft.

# Motivation

The aim of this project is to compare a series of different model on the task of predicting the outcome of a Dota2 match given the hero outline from the two teams (dire and radiant in the game).

# Features

1. Models
    * RandomEstimator: baseline producing estimates based on the observed win ratio in the training data.
    * HyperMLP: a fully tunable multilayer perceptron using embeddings for handling categorical inputs (i.e. heroes id)
    * HyperRNN: a fully tunable recurrent neural network using embeddings for handling categorical inputs (i.e. heroes id). Despite there is no temporal structure in the data we implemented this model in order to provide support for partial or sequential drafting (estimating the probability to win after each new hero is revealed).
    * HyperSKlearn: a fully tunable sklearn model, chosen during optimization from logistic regression, random forest and adaboost.

2. Script for performing data preparation.
3. Script for optimizing each of the tunable models.
4. Script for comparing each of the optimized model using a 30 fold stratified cross-validation strategy. 

# Results

<p align="center">
  <img width="600" height="600" src="https://github.com/vb690/draft_that_win/blob/main/results/figures/model_comparison.pdf">
</p>



# Credits

This project stems and is heavily influenced by another collective project that was carried out in collaboration with [Peter York](https://github.com/Pete-York), [Charlie Ringer](https://www.charlieringer.com/) and Moni Patra.

