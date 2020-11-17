"""
Our program is based on an open-sourced repository of a similar porject:
https://github.com/ZwEin27/User-based-Collaborative-Filtering

In this project, we:
1. predict the rating of a song for a given user based on his/her rating history
2. after prediting for all users in the test set, calculate the accuracy and RMSE
3. compare the user-based method with the item-based method
"""
import sys
import math
import os
import numpy as np
import matplotlib.pyplot as plt

# Define the Collaborative Filter Class
class Collaborate_Filter:
    def __init__(self, file_name, user_id, music, k):
        self.file_name = file_name
        self.user_id = user_id
        self.music = music
        self.k = k
        self.dataset = None
        self.user_dataset = None
        self.item_dataset = None

    def initialize(self):
        # check file exist and if it's a file or directory
        if not os.path.isfile(self.file_name):
            self.quit("Input file doesn't exist or it's not a file")
        
        # load data
        self.dataset, self.user_dataset, self.item_dataset = self.load_data(self.file_name)

        # check if user exist
        users = self.user_dataset.keys()
        if self.user_id not in users:
            self.quit("User ID doesn't exist")

    # Pearson Correlation: Calculate similarity between users
    def pearson_correlation(self, user1, user2):
            result = 0.0
            user1_data = self.user_dataset[user1]
            user2_data = self.user_dataset[user2]

            rx_avg = self.user_average_rating(user1_data)
            ry_avg = self.user_average_rating(user2_data)
            sxy = self.common_users(user1_data, user2_data)

            top_result = 0.0
            bottom_left_result = 0.0
            bottom_right_result = 0.0
            for item in sxy:
                rxs = user1_data[item]
                rys = user2_data[item]
                top_result += (rxs - rx_avg)*(rys - ry_avg)
                bottom_left_result += pow((rxs - rx_avg), 2)
                bottom_right_result += pow((rys - ry_avg), 2)
            bottom_left_result = math.sqrt(bottom_left_result)
            bottom_right_result = math.sqrt(bottom_right_result)

            if bottom_left_result == 0 or bottom_right_result == 0:
                return 0
            result = top_result/(bottom_left_result * bottom_right_result)
            return result

    def user_average_rating(self, user_data):
        avg_rating = 0.0
        size = len(user_data)
        for (music, rating) in user_data.items():
            avg_rating += float(rating)
        avg_rating /= size * 1.0
        return avg_rating

    def common_users(self, user1_data, user2_data):
        result = []
        ht = {}
        for (music, rating) in user1_data.items():
            ht.setdefault(music, 0)
            ht[music] += 1
        for (music, rating) in user2_data.items():
            ht.setdefault(music, 0)
            ht[music] += 1
        for (k, v) in ht.items():
            if v == 2:
                result.append(k)
        return result

    # k-nearest neighbour model
    def k_nearest_neighbors(self, user, k):
        neighbors = []
        result = []
        for (user_id, data) in self.user_dataset.items():
            if user_id == user:
                continue
            pearson = self.pearson_correlation(user, user_id)
            neighbors.append([user_id, pearson])
        sorted_neighbors = sorted(neighbors, key=lambda neighbors: (neighbors[1], neighbors[0]), reverse=True)

        for i in range(k):
            if i >= len(sorted_neighbors):
                break
            result.append(sorted_neighbors[i])
        return result

    # predict the rating of the music for the given user 
    def predict(self, user, item, k_nearest_neighbors):
        valid_neighbors = self.check_neighbors_validattion(item, k_nearest_neighbors)
        if not len(valid_neighbors):
            return 0.0
        top_result = 0.0
        bottom_result = 0.0
        for neighbor in valid_neighbors:
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1]
            rating = self.user_dataset[neighbor_id][item]
            top_result += neighbor_similarity * rating
            bottom_result += abs(neighbor_similarity)
        result = top_result/bottom_result
        return result

    def check_neighbors_validattion(self, item, k_nearest_neighbors):
        result = []
        for neighbor in k_nearest_neighbors:
            neighbor_id = neighbor[0]
            # print item
            if item in self.user_dataset[neighbor_id].keys():
                result.append(neighbor)
        return result

    # load_data(self, file_name): helper function, load the data
    # from the data set
    def load_data(self, file_name):
        input_file = open(file_name, 'rU')
        dataset = []
        user_dataset = {}
        item_dataset = {}
        for line in input_file:
            row = str(line)
            row = row.split("\t")
            row[2] = row[2][:-1]
            dataset.append(row)

            user_dataset.setdefault(row[0], {})
            user_dataset[row[0]].setdefault(row[1], float(row[2]))
            item_dataset.setdefault(row[1], {})
            item_dataset[row[1]].setdefault(row[0], float(row[2]))
        return dataset, user_dataset, item_dataset

    # display(self, k_nearest_neighbors, prediction): helper function, display the output
    def display(self, k_nearest_neighbors, prediction, user_id, music, k):
        print('{k} nearest neighbours of {user_id} and their similarity values:'.format(k=k, user_id=user_id))
        print('neighbour_id,  similarity')
        for neighbor in k_nearest_neighbors:
            print(neighbor[0], neighbor[1])
        print("\n")
        print('The predicted rating of song {song_id} for user {user_id} is: {rating}'.format(song_id=music, user_id=user_id, rating=prediction))
    
    # quit(self, err_desc): quit when error occurs
    def quit(self, err_desc):
        raise SystemExit('\n'+ "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n')

def load_test_data(file_name):
    input_file = open(file_name, 'rU')
    dataset = []

    for line in input_file:
        row = str(line)
        row = row.split("\t")
        dataset.append(row)

    return dataset

def eval_RMSE(testset, collaborate_filter, k):
    RMSE = 0
    for i in range(len(testset)):
        user_id = testset[i][0]
        song_id = testset[i][2].rstrip("\n")
        rating = float(testset[i][1])
        k_nearest_neighbors = collaborate_filter.k_nearest_neighbors(user_id, k)
        predicted_rating = collaborate_filter.predict(user_id, song_id, k_nearest_neighbors) 
        RMSE += (predicted_rating - rating) ** 2

    RMSE = (RMSE / len(testset)) ** 1/2
    
    return RMSE

# evaluate using accuacty with a threhold 
def eval_acc(testset, collaborate_filter, k, threhold):
    accuracy = 0
    for i in range(len(testset)):
        user_id = testset[i][0]
        song_id = testset[i][2].rstrip("\n")
        rating = float(testset[i][1])

        k_nearest_neighbors = collaborate_filter.k_nearest_neighbors(user_id, k)
        predicted_rating = collaborate_filter.predict(user_id, song_id, k_nearest_neighbors) 
        
        min_rating = rating - threhold
        max_rating = rating + threhold
        if (predicted_rating >= min_rating) & (predicted_rating < max_rating):
            accuracy += 1

    accuracy = accuracy / len(testset)
    
    return accuracy

if __name__ == '__main__':

    # publish
    # file_name = sys.argv[1]   # ratings-dataset.tsv
    # user_id = sys.argv[2]   # user name
    # music = sys.argv[3]     # music name
    # k = int(sys.argv[4])    # k neighbors

    # test
    file_name = "song_dataset.txt"
    user_id = "1644845"
    music = '50605'
    k = 10

    cf = Collaborate_Filter(file_name, user_id, music, k)
    cf.initialize()

    k_nearest_neighbors = cf.k_nearest_neighbors(user_id, k)

    prediction = cf.predict(user_id, music, k_nearest_neighbors)
    cf.display(k_nearest_neighbors, prediction, user_id, music, k)

    #print RMSE & ACC
    dataset = load_test_data(file_name)
    RMSE = eval_RMSE(dataset, cf, k)
    acc = eval_acc(dataset, cf, k, 1)
    print()
    print('The RMSE of the system: {}'.format(RMSE))
    print('The accuracy of the system: {}'.format(acc))