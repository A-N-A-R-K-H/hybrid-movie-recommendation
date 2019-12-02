# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:50:00 2019

@author: Jason


Completed
    -Fix unicode issue (mosty done)
    -Change topic analysis to exclude names, not all capitalized words
Next steps:
    -Check to make sure all parts work as intended (no skipping entires, etc)
    -Find optimal # of topics
        -Find imdb genres or movielens genres to test against topics
    -Ask about plagiarism
    -Replace websbite's code
    -Find recommendations given user's input
Steps that might give better results:
    -Metric should find how many words of that category it has, not just how many instances it has
        -AKA Rate based on number of unique words in each topic
    -Resolve parts of speech
        -Consider: https://www.clips.uantwerpen.be/pages/pattern-en#conjugation
        -Also https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
        -And https://stackoverflow.com/questions/17669952/finding-proper-nouns-using-nltk-wordnet
    -Test different metrics
        -Euclidean
        -score^4
        -Weighted
        -Rank
Optinal steps:
    -Ability to read scores / topics from movie_scores.csv so we don't have to do it repeatedly
    -Potential time fix: when building movie_wfm, add by word in movie_dictionary,
      then binary search in dictionary (which is a sorted list)
      (The current implementation is in n, while this could be in m*log(n))
    -Optimize find_closest_to()
    
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tag import pos_tag
#import re
#import csv
import time

class GenreAnalysis(object):
    def __init__(self,dataset_filename,cutoff_year,topics_file=None,
                 num_topics=100,num_words=8):
        #This function initializes the GenreAnalysis object
        #  If no topics_file specified, number of topics and max words will be 
        #  given by parameters
        
        self.descs = pd.read_csv(dataset_filename)
        self.descs = self.descs[(self.descs['Release Year'] >= cutoff_year) & (self.descs['Origin/Ethnicity'] == 'American')]
        
        self.num_topics = num_topics
        self.num_top_words = num_words
        self.topics = {}
        #topics is a dictionary of lists
        
        self.filename_suffix = str(self.num_topics)+" topics "+str(self.num_top_words)+" top words"
        
        self.genres = set()
        self.dictionary = []
        self.movie_scores = []
        
        if topics_file == None:
            self._find_topics()
        else:
            self._find_topics_from_csv(topics_file)
        
        self.generate_scores()
        #self.movie_wfm = [] #wfm = word frequency matrix
        #stores dictionaries for each entry
        #print len(self.descs)
    
    def _find_topics_from_csv(self, topics_filename):
        #This function is called by __init__() if a file is specified
        #  It takes the topics from the file
        
        #start = time.time()
        topics_df = pd.read_csv(topics_filename,index_col=0)
        #   print topics_df
        #topics_df.drop(columns=0)
        #print topics_df
        self.num_topics = len(topics_df)
        self.num_top_words = len(topics_df.iloc[0])
        self.filename_suffix = str(self.num_topics)+" topics "+str(self.num_top_words)+" top words"
        
        for topic_id in range(self.num_topics):
            for word in range(self.num_top_words):
                #print topic_id, word, topics_df.iloc[topic_id][word]
                feature_name = topics_df.iloc[topic_id][word].decode('utf8')
                
                if feature_name in self.topics:                    
                    self.topics[feature_name].append(topic_id)
                else:
                    self.topics[feature_name] = [topic_id]
        #print "_find_topics_from_csv() total time:", time.time() - start
        #print self.topics
        
    def _find_topics(self):
        #This function is called by __init__() if no topics file is specified
        #  It runs a topic analysis on the plot descriptions, then calls 
        #  _display_topics(), which also processes the data
        
        #NOTE: This function was heavily inspired by Aneesha Bakharia's code
        #  at https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
        
        #start = time.time()
        #shamelessly copied from the internet
        no_features = len(self.descs)
        
        descs_features = []
        for i in range(no_features):
            i_plot = str(self.descs.iloc[i]['Plot']).split()
            i_plot = [word for word,pos in pos_tag(i_plot) if pos != 'NNP']
            i_plot = " ".join(i_plot)
            descs_features.append(i_plot)
            #descs_features.append(str(self.descs.iloc[i]['Plot']))
            #descs_features[i] = re.sub(r'[A-Z][a-z]*','',descs_features[i])
            
        # NMF is able to use tf-idf
        # the codes is from url "
        tfidf_vectorizer = TfidfVectorizer(max_df=0.99, min_df=10, max_features=no_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(descs_features)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        
        nmf = NMF(n_components=self.num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        
        #print "_find_topics total time:",time.time() - start
        
        self._display_topics(nmf, tfidf_feature_names, self.num_top_words)
        
    def _display_topics(self, model, feature_names, num_top_words):
        #This function is called by _find_topics() - in addition to printing
        #  the topics, it also processes them and stores them in self.topics
        #  It also stores the topics in a csv for future use.
        #  _display_topics() does a lot more than just display the topics!
        
        #NOTE: This function was heavily inspired by Aneesha Bakharia's code
        #  at https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
        
        #start = time.time()
        print "_display_topics called"
        #shamelessly copied from the internet
        for topic_idx, topic in enumerate(model.components_):
            #print "Topic %d:" % (topic_idx)
            #print "|".join([feature_names[i]
            #for i in topic.argsort()[:-no_top_words - 1:-1]])
            
            for i in topic.argsort()[:-num_top_words - 1:-1]:
                feature_name = feature_names[i].encode('utf8')
                if feature_name in self.topics:                    
                    self.topics[feature_name].append(topic_idx)
                else:
                    self.topics[feature_name] = [topic_idx]
                #print feature_names[i],"assigned to topic",topic_idx
            """
            for topic in self.topics.keys():
                if topic_idx in self.topics[topic]:
                    print topic_idx,":",topic
            """
        #print self.topics
        
        words_by_topic = [[] for i in range(self.num_topics)]
        for i in range(self.num_topics):
            #print "Topic",i
            for topic in self.topics.keys():
                if i in self.topics[topic]:
                    words_by_topic[i].append(topic)
                    
        for i in range(self.num_topics):
            print "Topic",i
            print words_by_topic[i]
        print ""
                    
        df = pd.DataFrame(words_by_topic, columns=range(self.num_top_words))
        df.to_csv('test data/topics - '+self.filename_suffix+'.csv')
        
        #print "_display_topics total time:",time.time()-start
    
    def generate_scores(self):
        #This function generates the topic scores for each movie's plot
        #  and stores them in self.movie_scores
        #  It is called regardless of whether a topics file is specified, since
        #  there is currently no way to import scores from csv
        
        #start = time.time()
        num_errors = 0
        
        tokenize_time = 0
        topic_time = 0
        
        for i in range(len(self.descs)):
        #for i in range(1000):
            try:
                t_start = time.time()
                i_plot = nltk.word_tokenize(self.descs.iloc[i]['Plot'].lower().decode('utf8'))
                i_scores = [0 for i in range(self.num_topics)]
                tokenize_time += time.time() - t_start
                
                #normalized value for each instance of a word
                norm_value = 1/float(len(i_plot))
                #norm_value = 1
                
                t_start  = time.time()
                for word in i_plot:
                    if word in self.topics:
                        for topic in self.topics[word]:
                            i_scores[topic] += norm_value
                            #print "score for",word,"added to topic",topic
                    #else:
                        #print word,"not found"
                self.movie_scores.append(i_scores)
                topic_time += time.time() - t_start
            except UnicodeDecodeError:
                num_errors += 1
                self.movie_scores.append([-10]*self.num_topics)
                print "Error with description for", self.descs.iloc[i]['Title'], "(",i,")"
        
#        print "Number of errors:",num_errors
            
        df = pd.DataFrame(self.movie_scores, columns=range(self.num_topics))
        titles = self.descs['Title'].copy()
        titles.index = range(len(self.descs))
        df.insert(0, 'Title', titles)
        df.to_csv('movie_scores - '+self.filename_suffix+'.csv')
        #print df
        #print "generate_scores total time:",time.time()-start
        #print "tokenize time:",tokenize_time
        #print "topic time:",topic_time
    
    def find_closest_to(self, title, num_closest=1):
        #This function returns a list of the top (num_closest) closest movies
        #  to the specified movie
        
        #start  = time.time()
        closest_titles = []
        title_id = self.find_movie_id(title)
            
        if title_id >= 0:
            closest = [(999999,-1) for i in range(num_closest)]
            max_dist = 999999
            for i in range(len(self.movie_scores)):
                if i != title_id:
                    dist = self.eucl_distance(title_id, i)
                    if dist < max_dist:
                        closest.append((dist,i))
                        closest.sort(key = lambda d:d[0])#, reverse=True)
                        closest.pop()
                        max_dist = closest[-1][0]
            #print "Closest movies to \'",title,"\':"
            for i in range(num_closest):
                closest_titles.append(self.descs.iloc[closest[i][1]]['Title'])
                #print "    ",i,":",self.descs.iloc[closest[i][1]]['Title'],self.descs.iloc[closest[i][1]]['Release Year'],"(",closest[i][0],")"
        else:
            return -1
            #print "Movie not found"
        
        return closest_titles
        #print "find_closest_to total time:",time.time()-start
    
    def eucl_distance(self, m1, m2):
        # This function calculates the Euclidean distance from m1 and m2
        dist = 0
        
        for topic in range(self.num_topics):
            dist += (self.movie_scores[m1][topic] - self.movie_scores[m2][topic])**2
        
        dist = (dist)**0.5
        return dist
    
    """
    def weight_distance(self,m1,m2):
        dist = 0
        
        for topic in range(self.num_topics):
            score1 = 1000*self.movie_scores[m1][topic]
            score2 = 1000*self.movie_scores[m2][topic]
            
            if score1==0 or score2==0:
                if score1==score2:
                    dist += 1
                else:
                    dist += 1-np.log(score1 + score2)
            else:
                dist += (score1)*(1-abs(score1-score2)/(2*(score1+score2)))
        dist = (dist)**0.5
        return dist
    
    def rank_distance(self,m1,m2):
        return -1
    """ 
    
    def find_movie_id(self,title):
        #This function finds the movie id for the given title
        #  It can also tell whether the movie is in the dataset, returns -1 if not
        for i in range(len(self.descs)):
            if self.descs.iloc[i]['Title'] == title:
                return i
        return -1

def main():
    #ga = GenreAnalysis("wiki_movie_plots_deduped.csv", 1960)
    ga = GenreAnalysis("wiki_movie_plots_deduped.csv", 1960) 
                       #topics_file="topics - 35 topics 10 top words.csv")
    
    print ga.find_closest_to('Avatar',5)
    ga.find_closest_to('The Lion King',5)
    ga.find_closest_to('Star Wars Episode IV: A New Hope (aka Star Wars)',5)
    ga.find_closest_to('Titanic',5)
    ga.find_closest_to('Toy Story',5)
    ga.find_closest_to('Mulan',5)
    ga.find_closest_to('The Princess Bride',5)
    ga.find_closest_to('The Shawshank Redemption',5)
    ga.find_closest_to('Harry Potter and the Order of the Phoenix',5)  
    ga.find_closest_to('Inception',5)
    ga.find_closest_to('Avengers, TheThe Avengers',5)
    ga.find_closest_to('Paddington',5)
    ga.find_closest_to('The Martian',5)
    ga.find_closest_to('The Lord of the Rings: The Fellowship of the Ring',5)
    ga.find_closest_to('Twilight',5)

if __name__ == '__main__':
    main()

#ga = GenreAnalysis("wiki_movie_plots_deduped.csv", 1960, "test data/topics - 50 topics 10 top words.csv")
#print sorted(ga.genres)
#print ga.dictionary
#print ga.movie_wfm[0]

#print(read_data)

