# Music Recommendation System
Our program is based on an open-sourced repository of a similar project:
https://github.com/ZwEin27/User-based-Collaborative-Filtering

## Overview
In this project, we built a music recommendation system using the Collaborative Filtering algorithm. It will predict the rating of a song for a given user and recommend songs to user based on the prediction. Currently this system has only achieved the prediction function, we will add the recommendation function later.

## Features
We implemented two versions using the user-based and item-based methods respectively. Both of the implementations would:

1. calculate the similarities between the given user/item and all other user/items
2. predict the rating of a song for a given user based on the similarities calculated in the previous step
3. calculate the accuracy and RMSE after predicting for all users in the test set

## Required library
In the implementation, we imported libraries such as numpy and matplotlib to help us build the music recommendation system.

## Command to run our system
Python {python_file} {input_file} {user_id} {song_id} {k_neighbours}
E.g Python userBasedCollabFilter.py 800015 105021 song_dataset.txt 10

## Example output:

10 nearest neighbours and their similarity values:
1600196 0.5589067288502397
1695150 0.5525568839779398
1795955 0.5394716212738989
1600149 0.4687109821048257
1600086 0.4114027364392768
1600113 0.40649681127862186
1600149 0.38980777077594553
1776214 0.3233291304112335
1795933 0.32098023210283666
1600141 0.28127847504192927


The predicted rating of song 1853 for user 1662912 is: 4.343751017141255

The RMSE of the system: 0.4920309712863019
The accuracy of the system: 0.8154059680777238