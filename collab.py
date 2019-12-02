import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from collections import Counter
from sklearn.model_selection import train_test_split
from math import sqrt
import scipy.sparse as sps
import csv
import re


# return the level of the prediction
def level(rate):
    if rate >= 4.75:
        return 5.0
    elif rate >= 4.25:
        return 4.5
    elif rate >= 3.75:
        return 4.0
    elif rate >= 3.25:
        return 3.5
    elif rate >= 2.75:
        return 3.0
    elif rate >= 2.25:
        return 2.5
    elif rate >= 1.75:
        return 2.0
    elif rate >= 1.25:
        return 1.5
    elif rate >= 0.75:
        return 1.0
    elif rate >= 0.25:
        return 0.5
    else:
        return 0.0

#whether it is larger than 4.0 (i.e. worth recommending)
def binary_classify_accuracy(pred, true):
    if pred >= 4.0 and true >= 4.0:
        return True
    elif pred < 4.0 and true < 4.0:
        return True
    else:
        return False

# prescreen returns a smaller dataset with only the top popular movies within the group, without affecting the input movies
def prescreen(dataset, id_list, upp):  # upper bound and lower bound
    todo = dataset[~dataset['movie_id'].isin(id_list)]
    reserve = dataset[dataset['movie_id'].isin(id_list)]
    #top = todo.groupby('movie_id')['movie_id'].value_counts().nlargest(upp)  # not workable
    top = todo.groupby('movie_id').size().nlargest(upp).reset_index(name='count')
    return reserve.append(todo[todo['movie_id'].isin(top['movie_id'])], ignore_index=True)

# create the NM after screening and the dictionaries for bidirectional hashing movie_id and 
def create_NM(dataset):
    n = dataset.user_id.unique().shape[0]
    m = dataset.movie_id.unique().shape[0]
    mat = np.zeros((n, m))
    user = dict()  # map 0,1,2,... to user_id
    mv = dict()  # map 0,1,2,... to movie_id
    user_re = dict()  # map user_id to 0,1,2,...
    mv_re = dict()  # map movie_id to 0,1,2,...
    
    count = 0
    for i in dataset.user_id.unique():
        user[count] = i
        user_re[i] = count
        count += 1
    
    count = 0
    for i in dataset.movie_id.unique():
        mv[count] = i
        mv_re[i] = count
        count += 1
    
    for row in dataset.itertuples():
        mat[user_re[row[1]], mv_re[row[2]]] = row[3]
#    for id, rate in zip(id_list, rate_list):
#        mat[-1, mv_re[id]] = rate
    return mat, user, mv, user_re, mv_re

# predict:
def predict_new_input(mat, mv, mv_re, id_list, rate_list):  # mv is the hashing dictionary
    n, m = mat.shape
    new_row = np.zeros(m).reshape((1, m))
    for id, rate in zip(id_list, rate_list):
        new_row[0, mv_re[id]] = rate
    mat = np.append(mat, new_row, axis=0)
    n = n+1
    # check whether to discard the two lines below:
    known = [x for x in range(m) if mat[-1,x]!=0]
    desired = [x for x in range(m) if mat[-1,x]==0]  # stores all the index of movies in the matrix that is to predict ratings
    
    predicts = dict()  # key is movie_id and value is the predicted ratings
    
    T1 = mat[:, known]
    model = NMF(n_components=3, tol=0.005)  # give a 5-component model if not specify
    model.fit(T1)
    W1 = model.fit_transform(T1)
    H1 = model.components_
    #print model.reconstruction_err_
    #print 'W1', W1[-1,:]
    
    for i in desired:  # each time focus on a even smaller matrix with only one desired entry to predict
        cols = known
        cols.append(i)
        T2 = mat[:-1, cols]
        model2 = NMF(n_components=3, tol=0.005)  # if do not specify n_components, there will be 23846 components
        model2.fit(T2)
        W2 = model2.fit_transform(T2)
        H2 = model2.components_
        #print model2.reconstruction_err_
        #print 'H2', H2[:,-1]
        predicts[mv[i]] = np.dot(H2[:,-1],W1[-1,:])
    #print predicts
    return predicts


# this is the workflow of this file:
# note that the three columns name must be 'movie_id', 'user_id', and 'rating'
# for now the input number is 5, which should be change to any number(probably larger than 3)
# upp is the number of movies to predict, num is the number of result to be shown
def collab_recom(dataset, id_list, rate_list, upp, num):
    #filter = (dataset['movie_id'] == id_list[0]) | (dataset['movie_id'] == id_list[1]) | (dataset['movie_id'] == id_list[2]) | (dataset['movie_id'] == id_list[3]) | (dataset['movie_id'] == id_list[4])
    #filter = set_filter(dataset, id_list)
    #set_usr = dataset[filter]
    set_usr = dataset[dataset['movie_id'].isin(id_list)]
    new_set = dataset[dataset['user_id'].isin(set_usr['user_id'])]  # new set with similar users
    
    sm_set = prescreen(new_set, id_list, upp)  # smaller set with potential movies
    mat, user, mv, user_re, mv_re = create_NM(sm_set)
    prediction = predict_new_input(mat, mv, mv_re, id_list, rate_list)  # prediction is a dictionary of all movie rating prediction with key as movie_id
    # next is to get the num highest ratings in prediction
    return dict(Counter(prediction).most_common(num))

# Testing accuracy:

def predict_test(mat, mv, mv_re, id_list, rate_list, n_com):
    n, m = mat.shape
    for id, rate in zip(id_list, rate_list):
        mat[-1, mv_re[id]] = rate
    
    #mat = mat.tocsr()
    #print mat.shape
    T1 = mat[:-1, :]
    print T1.shape
    model = NMF(n_components=n_com, tol=0.01)
    model.fit(T1)
    W1 = model.fit_transform(T1)
    H1 = model.components_
    #H1[:,-1]
    T2 = mat[:,:-1]
    model = NMF(n_components=n_com, tol=0.01)
    model.fit(T2)
    W2 = model.fit_transform(T2)
    H2 = model.components_
    #W2[-1,:]
    return np.dot(H1[:,-1],W2[-1,:])
    


# testing accuracy based on the test dataset is different from predicting from new input. The thought here is to pick a non-zero rating for a test user randomly and use the other ratings of him as input to predict
# test_usr is the np array for user_id for testing
def test_accuracy(dataset, train_usr, test_usr, upp, n_com):
    train_df = dataset[dataset['user_id'].isin(train_usr)]  # training dataset
    error = 0
    num = 0
    correct_time = 0
    bin_correct_time = 0
    predicted_record = []
    for usr in test_usr.tolist():
        print usr
        set = dataset[dataset['user_id']==usr]
        mv_arr = set['movie_id'].values
        rate_arr = set['rating'].values
        print mv_arr.shape[0]
        if mv_arr.shape[0] > n_com:
            # pick a random movie_id:
            #tar = np.random.choice(mv_arr)  # tar is the movie_id to predict
            ran = np.random.randint(mv_arr.shape[0])
            tar = mv_arr[ran]  # a random movie_id as target
            truth = rate_arr[ran]  # its true rating
            id_list = [mv_arr[i] for i in range(mv_arr.shape[0]) if i != ran]  # the movie_id list for predicting tar
            #truth = dataset[(dataset['user_id']==usr) & (dataset['movie_id']==tar)]['rating']
            rate_list = [rate_arr[i] for i in range(rate_arr.shape[0]) if i != ran]
        #        arr = np.zeros(m)
        #        for row in set.itertuples():
        #            mat[mv_re[row[2]]] = row[3]
            entire_list = mv_arr.tolist()
            entire_set = train_df[train_df['movie_id'].isin(entire_list)]
            top = entire_set.groupby('user_id').size().nlargest(upp).reset_index(name='count')
            smaller_set = entire_set[entire_set['user_id'].isin(top['user_id'])]
            #pred_set = smaller_set.append(train_df[train_df['movie_id']==tar], ignore_index=True)
            #mat, user, mv, user_re, mv_re = create_NM(pred_set)  # mat is (95887L, 18L)
            
            n = mv_arr.shape[0]
            
            #mat = sps.lil_matrix((upp+1, n))
            mat = np.zeros((upp+1, n))
            user = dict()  # map 0,1,2,... to user_id
            mv = dict()  # map 0,1,2,... to movie_id
            user_re = dict()  # map user_id to 0,1,2,...
            mv_re = dict()  # map movie_id to 0,1,2,...
            count = 0
            for i in smaller_set.user_id.unique():
                user[count] = i
                user_re[i] = count
                count += 1
            
            count = 0
            for i in id_list:
                mv[count] = i
                mv_re[i] = count
                count += 1
            mv[count] = tar
            mv_re[tar] = count
            
            for row in smaller_set.itertuples():
                mat[user_re[row[1]], mv_re[row[2]]] = row[3]
            
            pred = predict_test(mat, mv, mv_re, id_list, rate_list, n_com)
            predicted_record.append(pred)
            error = error + (pred - truth)**2
            num += 1
            print pred, truth
            print 'err:', error
            print 'num: ', num
            # test accuracy by rate level:
            if level(pred) == truth:
                correct_time += 1
            # test binary classification:
            if binary_classify_accuracy(pred, truth):
                bin_correct_time += 1
    RMSE = sqrt(error/float(num))
    print 'test result:'
    print 'num:', num
    print 'RMSE:', RMSE
    print 'accuracy rate: ', correct_time/float(num)
    print 'binary classification rate: ', bin_correct_time/float(num)
    return RMSE
    #print predicted_record
    #return predicted_record

# this is to do a real scenario recommendation based on some inputs with ratings        
def fire_recom(name_list, rating_list):
    r_cols = ['user_id', 'movie_id', 'rating']
    dataset = pd.read_csv('ratings.csv', names=r_cols, usecols=[0, 1, 2], dtype={'user_id': np.int, 'movie_id': np.int, 'rating': np.float}, skiprows=[0], encoding='latin-1')
    cols_name = ['movie_id', 'title', 'genres']
    nameset = pd.read_csv('movies.csv', names=cols_name, usecols=[0, 1, 2], dtype={'movie_id': np.int, 'title': str, 'genres': str}, skiprows=[0], encoding='latin-1')
#    if id_list == None:
#        id_list = []
#        rating_list = []
#        name = ''
#        while True:
#            name = raw_input('Please input a movie name. Input "start" to start recommendation filtering:\n')
#            if name == 'start':
#                print id_list, rating_list
#                break
#            input_id = nameset[nameset['title'].str.contains(r'^'+re.escape(name)+r'$')]['movie_id']  # for now the name must match the name appear in the dataset rigorously, and it can not deal with same names in the dataset
#            input_id = int(input_id)
#            id_list.append(input_id)
#            rate = raw_input('Please give a rating on this movie on a 5-star basis:\n')
#            rate = int(rate)
#            rating_list.append(rate)
#    else:
#        rating_list = []
#        for i in range(len(id_list)):
#            rate = raw_input('Please give a rating on the corresonding movie on a 5-star basis:\n')
#            rating_list.append(rate)
    id_list = []
    for name in name_list:
        input_id = nameset[nameset['title'].str.contains(r'^'+re.escape(name)+r'$')].values[0][0]
        id_list.append(input_id)
#    id_list = [1721, 1682, 179819, 7022, 103228]
#    rating_list = [5.0, 5.0, 3.0, 4.5, 2.0]
    top5 = collab_recom(dataset, id_list, rating_list, 50, 5)
    names = []
    for key in top5:
        names.append(nameset[nameset['movie_id']==key]['title'].values[0])
    for name in names:
        print name



def main():
    r_cols = ['user_id', 'movie_id', 'rating']
    dataset = pd.read_csv('ratings.csv', names=r_cols, usecols=[0, 1, 2], dtype={'user_id': np.int, 'movie_id': np.int, 'rating': np.float}, skiprows=[0], encoding='latin-1')
    cols_name = ['movie_id', 'title', 'genres']
    nameset = pd.read_csv('movies.csv', names=cols_name, usecols=[0, 1, 2], dtype={'movie_id': np.int, 'title': str, 'genres': str}, skiprows=[0], encoding='latin-1')
    
    
    # do partition here:
    # randomly take 20% users out
    train_usr, rest = train_test_split(dataset.user_id.unique(), test_size=0.01)
    validation_usr, test_usr = train_test_split(rest, test_size=0.5)
    
    # there are 283228 unique users
    # train_usr: a np array with size 226582
    # rest: a np array with size 56646 (20%)
    
    # choosing the best n_component:
    #test_accuracy(dataset, train_usr, validation_usr, 50, 2)
    #RMSE: 0.234633649161
    
    #test_accuracy(dataset, train_usr, validation_usr, 50, 3)
    #RMSE: 0.25147865397
    
    #test_accuracy(dataset, train_usr, validation_usr, 50, 4)
    #RMSE: 0.213990666342
    
#    test_accuracy(dataset, train_usr, validation_usr, 50, 5)
#    RMSE: 0.243800306348
#    RMSE1 = []
#    for n_com in range(5)[2:]:
#        RMSE1.append(test_accuracy(dataset, train_usr, validation_usr, 50, n_com))
#    #list of RMSE: [10.033782494209843, 1.2651045229517777, 1.7882098776206223]
#    print "list of RMSE1 :", RMSE1
#    
#    RMSE2 = []
#    for n_com in range(11)[5:]:
#        RMSE2.append(test_accuracy(dataset, train_usr, validation_usr, 50, n_com))
#    csvfile = "RMSE2.csv"
#    with open(csvfile, "w") as output:
#        writer = csv.writer(output, lineterminator='\n')
#        for val in RMSE2:
#            writer.writerow([val])    
#    print "list of RMSE2 :", RMSE2
#    1.514188667
#    1.507183865
#    1.580100949
#    1.596076259
#    1.456105944
#    1.804974522

#    
#    RMSE3 = []
#    for n_com in range(16)[11:]:
#        RMSE3.append(test_accuracy(dataset, train_usr, validation_usr, 50, n_com))
#    csvfile = "RMSE3.csv"
#    with open(csvfile, "w") as output:
#        writer = csv.writer(output, lineterminator='\n')
#        for val in RMSE3:
#            writer.writerow([val])
#    print "list of RMSE3 :", RMSE3
#    2.637895672
#    1.927740318
#    1.773440781
#    1.619293198
#    1.669712171


#    RMSE4 = []
#    for n_com in range(21)[16:]:
#        RMSE4.append(test_accuracy(dataset, train_usr, validation_usr, 50, n_com))
#    csvfile = "RMSE4.csv"
#    with open(csvfile, "w") as output:
#        writer = csv.writer(output, lineterminator='\n')
#        for val in RMSE4:
#            writer.writerow([val])
#    print "list of RMSE4 :", RMSE
    #1.615912454
    #1.823558923
    #2.428847848
    #2.22904435
    #1.912016852


#    RMSE5 = []
#    for n_com in range(26)[21:]:
#        RMSE5.append(test_accuracy(dataset, train_usr, validation_usr, 50, n_com))
#    csvfile = "RMSE5.csv"
#    with open(csvfile, "w") as output:
#        writer = csv.writer(output, lineterminator='\n')
#        for val in RMSE5:
#            writer.writerow([val])
#    print "list of RMSE5 :", RMSE5
#    1.918571168
#    1.939745818
#    1.652844851
#    2.096331417
#    2.173080651

#    RMSE6 = []
#    for n_com in range(31)[26:]:
#        RMSE6.append(test_accuracy(dataset, train_usr, validation_usr, 50, n_com))
#    csvfile = "RMSE6.csv"
#    with open(csvfile, "w") as output:
#        writer = csv.writer(output, lineterminator='\n')
#        for val in RMSE6:
#            writer.writerow([val])
#    print "list of RMSE6 :", RMSE6
#    1.765141839
#    1.981529545
#    2.490697779
#    2.378056013
#    1.919740085

    
    
    
#    RMSE: 
    
    # accuracy on test set: n_components as 4
    #test_accuracy(dataset, train_usr, test_usr, 50, 4)
#    RMSE: 2.6514523911152685
    
    test_accuracy(dataset, train_usr, test_usr, 50, 3)
#result:
#RMSE: 2.43977460206
#accuracy rate:  0.238060249816
#binary classification rate:  0.636296840558
    
#    id_list = [1721, 1682, 179819, 7022, 103228]
#    rating_list = [5.0, 5.0, 3.0, 4.5, 2.0]
#    top5 = collab_recom(dataset, id_list, rating_list, 20, 5)
#    names = []
#    for key in top5:
#        names.append(nameset[nameset['movie_id']==key]['title'])
#    for name in names:
#        print name
    
    x_para = range(2,31)
    y_num = [10.033782494209843,
             1.2651045229517777,
             1.7882098776206223,
             1.514188667,
             1.507183865,
             1.580100949,
             1.596076259,
             1.456105944,
             1.804974522,
             2.637895672,
             1.927740318,
             1.773440781,
             1.619293198,
             1.669712171,
             1.615912454,
             1.823558923,
             2.428847848,
             2.22904435,
             1.912016852,
             1.918571168,
             1.939745818,
             1.652844851,
             2.096331417,
             2.173080651,
             1.765141839,
             1.981529545,
             2.490697779,
             2.378056013,
             1.919740085
             ]
    # plot:
    plt.plot(x_para, y_num)
    plt.xlabel('value of n_components')
    plt.ylabel('RMSE on validation dataset')
    plt.xticks(x_para)
    plt.title('RMSE on Validation Dataset with Different n_components')
    plt.show()
    

if __name__=="__main__":
    main()


