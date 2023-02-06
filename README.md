# Ion Switching

This repo contains a collections of python scripts which in turn contain the models I developed to solve the University of Liverpool - Ion Switching challenge. See link below for more info

> https://www.kaggle.com/competitions/liverpool-ion-switching/overview

The main model consist of the combination of two algorithm: Multi Gaussian Regression and Hidden Marcov Chain. I used the Multi Gaussian Model (MGM) to find the location and the width of each stripe ().
Multi Gaussian Regression alone gave a reasonable good accuracy (about 0.9000) but it does not infer any temporal correlations between envents. Therefore I included together with MGM a Hidden Marcov Chain Model (HMCM). 
I used the info from the MGM as intial conditions for HMCM which was used primarly to infer the transition probabilities between the current state and the next state



# Prerequisites
- Python 3.x
- NumPy
- Pandas
- Pomegranate (for Hidden Markov Chain)
- h5py
- bayes_opt
- matplotlib
- os
- sklearn
- scipy

