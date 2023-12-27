# Code for the UWNCD19 Kaggle Neural Data prediction competition

Code and models used by the FeatureSpaceExplorers team to submit to the UWNDC19 competition to resulting in 4th place.

- `main.py`: main code to get features, predict on the data, and submit results.
    - Modeling consists of 3 main parts:
        1. Run stimuli through Alexnet and extract features from every layer.
        2. Take the features and fit Lasso regression on them to predict the spike rates for each neuron. (Feature selection)
        2. Then, take the features with nonzero Lasso coefficients and fit Ridge regression to predict the spike rates.

- `baseline_mask.py` and `single_baseline_mask.py`: alternate modeling attempts. 
    - Consists of a Pytorch model which takes stimulus videos that each neuron saw, pushes them through VGG16 to extract features, and then learns a spatial mask on those extracted features. The spatial mask is then averaged to a single pixel per channel. Finally, linear regression is done to predict the spike rate from the vector of feature channels for each neuron.
    - `single_` is for individual neurons and `baseline_mask` is for groups of neurons to be done at once.

These models minimize the root mean squared error between the actual and predicted V4 neuron spike rate, in response to stimulus images.