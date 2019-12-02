import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from collections import Counter
from sklearn.model_selection import train_test_split
import scipy.sparse as sps
from sklearn.metrics import explained_variance_score


# get the explained variance
def get_var(model, data):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return explained_variance_score(data.toarray(), prediction)
# reference from url: https://stackoverflow.com/questions/48148689/how-to-compare-predictive-power-of-pca-and-nmf


# test the optimal n_components value:
def validate(train_m, val_m):
    validation = []
    max = 0
    K = [300, 400, 500, 600]  # from 60 to 1000 with step=10
    for k in K:
        nmf = NMF(n_components=k, random_state=0, tol=0.01).fit(train_m)
        var = get_var(nmf, val_m)
        validation.append(var)
        print var
        if var > max:
            max = var
            n_com = k
    print validation
    print k, var
    return k, var


def main():
    r_cols = ['user_id', 'movie_id', 'rating']
    dataset = pd.read_csv('ratings.csv', names=r_cols, usecols=[0, 1, 2], dtype={'user_id': np.int, 'movie_id': np.int, 'rating': np.float}, skiprows=[0], encoding='latin-1')
    
    #n = dataset.user_id.max
    #m = dataset.movie_id.max
    #cm = csr_matrix((dataset['rating'], (dataset['user_id'], dataset['movie_id'])), dtype=np.float32)
    
    #cm.data = np.nan_to_num(cm.data)
    #cm.eliminate_zeros()
    
    # directly creating a csr_matrix will result in error later when doing NMF, the better way is to create a lil_matrix row by row, afterwards convert it into csr:
    n = dataset.user_id.unique().shape[0]
    m = dataset.movie_id.unique().shape[0]
    mat = sps.lil_matrix((n, m))
    
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
    #for id, rate in zip(id_list, rate_list):
    #    mat[-1, mv_re[id]] = rate
    
    cm = mat.tocsr()
    
    train_m, rest = train_test_split(cm, test_size=0.02)
    val_m, test_m = train_test_split(rest, test_size=0.5)
    
    model = NMF(n_components=50, random_state=0, verbose=True)
    model.fit(train_m)
    W = model.fit_transform(train_m)
    print model.reconstruction_err_
    
    # plot result:
    x_para = range(2,11)
    y_var = [0.011148820087655631,
        0.012289894027255313,
        0.013387386796954979,
        0.015236649392704244,
        0.015889526247372773,
        0.01678371376687657,
        0.01711572272804447,
        0.017423276607785323,
        0.018254795671530002
        ]
    plt.plot(x_para, y_var)
    plt.xlabel('value of n_components')
    plt.ylabel('explained variance on test dataset')
    plt.xticks(x_para)
    plt.title('Explained Variance on Test Dataset with Different n_components (within 10)')
    plt.show()
    
    x_para2 = x_para + range(50, 54)
    y_var2 = y_var + [0.027330455218776983,
                      0.028385167668193538,
                      0.028100367653178478,
                      0.029776391723281604
                      ]
    plt.plot(x_para2, y_var2)
    plt.xlabel('value of n_components')
    plt.ylabel('explained variance on test dataset')
    #plt.xticks(x_para2)
    plt.title('Explained Variance on Test Dataset with Different n_components (within 60)')
    plt.show()
    
    x_para3 = [i*10 for i in range(6,33)]
    y_var3 = [0.03012005445858233,
              0.030156254455530565,
              0.03117303309593157,
              0.030874502247799675,
              0.03182358560034558,
              0.03393135085081179,
              0.032347745166297154,
              0.03330484704438651,
              0.03342732100699637,
              0.03678242762051066,
              0.04244988810906529,
              0.04783528295834701,
              0.048681026247524295,
              0.04765110335269021,
              0.04877476069237645,
              0.03620720654212584,
              0.048149617627192505,
              0.049006507894698285,
              0.05087528543807484,
              0.05025886897183912,
              0.038922453038700545,
              0.05058462800812159,
              0.05020918395350984,
              0.05140515650588571,
              0.05070266630661625,
              0.05059631236082883,
              0.05158765392504946,
              ]
    plt.plot(x_para3, y_var3)
    plt.xlabel('value of n_components')
    plt.ylabel('explained variance on test dataset')
    #plt.xticks(x_para3)
    plt.title('Explained Variance on Test Dataset with Different n_components (within 320)')
    plt.show()
# Test results:    
# explained variance on val_m, starting from k=2:
#0.011148820087655631
#0.012289894027255313
#0.013387386796954979
#0.015236649392704244
#0.015889526247372773
#0.01678371376687657
#0.01711572272804447
#0.017423276607785323
#0.018254795671530002

# k=50:
#0.027330455218776983
#0.028385167668193538
#0.028100367653178478
#0.029776391723281604
    
# k = 60: step=10
#0.03012005445858233
#0.030156254455530565
#0.03117303309593157
#0.030874502247799675
#0.03182358560034558
#0.03393135085081179
#0.032347745166297154
#0.03330484704438651
#0.03342732100699637
#0.03678242762051066
#0.04244988810906529
#0.04783528295834701
#0.048681026247524295
#0.04765110335269021
#0.04877476069237645
#0.03620720654212584
#0.048149617627192505
#0.049006507894698285
#0.05087528543807484
#0.05025886897183912
#0.038922453038700545
#0.05058462800812159
#0.05020918395350984
#0.05140515650588571
#0.05070266630661625
#0.05059631236082883
#0.05158765392504946

# k = 700: Memory Error!

# k = 300: 0.05070266630661625
# k = 400: 0.054403726407786325

if __name__=="__main__":
    main()