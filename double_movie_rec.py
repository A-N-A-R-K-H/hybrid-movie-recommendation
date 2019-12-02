# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:12:06 2019

@author: Jason
"""

import genre_analysis as gen_an
import collab

ga = gen_an.GenreAnalysis("wiki_movie_plots_deduped.csv", 1960) 
                       #topics_file="topics - 100 topics 8 top words.csv")

def recommend():
#This function prompts the user for 5 movies and their ratings, then gives
# recommendations.    
    
    movies = []
    ratings = []
    
    for i in range(5):
        movie_title = raw_input("Enter movie "+str(i)+": ")
        movie_id = ga.find_movie_id(movie_title)
        while (movie_id == -1):
            print "Movie not found, please try again"
            movie_title = raw_input("Enter movie "+str(i)+": ")
            movie_id = ga.find_movie_id(movie_title)
        
        movie_year = raw_input("What year was "+movie_title+" released? ")
        movie_title += " ("+movie_year+")"
        #Thprint movie_title
        movies.append(movie_title)
        
        movie_score = float(raw_input("Enter rating for "+movie_title+", from 0.5 to 5: "))
        while not (2*movie_score == int(2*movie_score) and 0.5 <= movie_score and 5.0 >= movie_score):
            print "Score invalid, please try again"
            movie_score = float(raw_input("Enter rating for "+movie_title+", from 0.5 to 5: "))
        ratings.append(movie_score)
        
        print("----------")
     
    print movies
    print ratings
    
    collab.fire_recom(movies,ratings)

def test():
    movies = ['Mulan', 'Harry Potter and the Order of the Phoenix', 'Three Billboards Outside Ebbing, Missouri', '2001: A Space Odyssey', 'Blade Runner 2049']
    movies_with_year = ['Mulan (1998)', 'Harry Potter and the Order of the Phoenix (2007)', 'Three Billboards Outside Ebbing, Missouri (2017)', '2001: A Space Odyssey (1968)', 'Blade Runner 2049 (2017)']
    scores = [4.5, 3.5, 4.5,     5.0, 4.5]
    
    print "Your ratings:"
    for i in range(5):
        print movies_with_year[i]+": "+str(scores[i])
    
    print "\nRecommended movies given your ratings:"
    collab.fire_recom(movies_with_year,scores)
    
    print "\nMovies closest to your highest rated movie ("+movies[get_highest_index(scores)]+"):"
    nearest_movies = ga.find_closest_to(movies[get_highest_index(scores)],5)
    for movie in nearest_movies:
        print movie
    
def get_highest_index(my_list):
    highest = 0
    for i in range(1,len(my_list)):
        if my_list[i] > my_list[highest]:
            highest = i
    return highest

if __name__ == '__main__':
    test()