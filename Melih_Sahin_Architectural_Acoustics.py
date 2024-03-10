# Google Server Link: https://colab.research.google.com/drive/1TLhIFcfBY22xS6SxXSd1rPHcSadqzeOp?usp=sharing#scrollTo=EaVGKRDIRUvW
#Room Geometry Impulse Response Generator:
#Melih Şahin Koç University Summer 2022 Research Internship under the guiadnce of Professor Dr. Engin Erzin
#Except the part to get the audio file, and the definition to graph a sound file, all code completely belongs to the author, Melih Şahin.
#Average Running Time: 30 seconds on the provided Google Server

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[+0.0, +5.5],
                        [+0.0, +0.0],
                        [+0.0, -5.5]])

group_covariances = np.array([[[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+3.2, +2.8],
                               [+2.8, +3.2]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw06_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 3

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    data_set1 = np.genfromtxt("hw06_initial_centroids.csv", delimiter = ",")
    means = data_set1[:, [0, 1]]
    


    covariances=np.zeros((3,2,2))
    numbers=np.zeros((3))
    priors=np.zeros((3))

    for a in range(1200):
        category=0
        distance1=(X[a][0]-means[0][0])**2+(X[a][1]-means[0][1])**2

        for b in range(2):
            distance = (X[a][0] - means[b+1][0]) ** 2 + (X[a][1] - means[b+1][1]) ** 2
            if distance<distance1:
                category=b+1
                distance1=distance
        priors[category]+=1/1200
        numbers[category]+=1

        a1=X[a][0]-means[category][0]
        a2=X[a][1]-means[category][1]

        covariances[category][0][0]+=a1**2
        covariances[category][0][1] +=a1*a2
        covariances[category][1][0] +=a1*a2
        covariances[category][1][1] +=a2**2
    for c in range(3):
        covariances[c][0][0] /= numbers[c]
        covariances[c][0][1] /=numbers[c]
        covariances[c][1][0] /= numbers[c]
        covariances[c][1][1] /= numbers[c]
    
    # your implementation ends above
    
   
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    def multivariate_gaussian(meanvector, gaussianmatrix, data_point):
        det=gaussianmatrix[0][0]*gaussianmatrix[1][1]-gaussianmatrix[1][0]*gaussianmatrix[0][1]
        det=np.linalg.det(gaussianmatrix)

        if det==0:
            det=0.000001
            print("dire error")
            return 0
        firstconstant=1/((((2*np.pi)**3)*det)**(1/2))
        inverse=np.linalg.inv(gaussianmatrix)
        
        a2=(meanvector-data_point)
        a1=np.array([[a2[0]],[a2[1]]])

        
        result=np.matmul(a2,inverse)
        
      

        result=np.matmul(result,a1)
      

          
        number=np.exp((-1/2)*result)
        #print(number*firstconstant)


        return number*firstconstant
        
        



    for a in range(100):




        c = np.zeros((1200,3))
        for k in range(1200):
            for l in range(3):
                c[k][l]=multivariate_gaussian(means[l],covariances[l],X[k])*priors[l]
                h=0
                for q in range(3):
                    h+=multivariate_gaussian(means[q],covariances[q],X[k])*priors[q]
                c[k][l]/=h
                    
        
        

        for t in range(3):
            priors[t]=0
            for p in range(1200):
                priors[t]+=c[p][t]
            priors[t]/=1200
        

        
        for t1 in range(3):
            means[t1]=[0,0]
            for p in range(1200):
                means[t1]+=c[p][t1]*(X[p])
            yyy=0
            
            for p in range(1200):
                yyy+=c[p][t1]
            means[t1]/=yyy
            
        for t2 in range(3):
            covariances[t2]=[[0,0],[0,0]]
            for p in range(1200):
                a1=(X[p]-means[t2])
                result=np.array([[a1[0]*a1[0],a1[0]*a1[1]],[a1[0]*a1[1],a1[1]*a1[1]]])



                

                
                covariances[t2]+=c[p][t2]*result
            yyy=0
            
            for p in range(1200):
                
                yyy+=c[p][t2]
            covariances[t2]/=yyy
    assignments=np.zeros((1200))        
    for a in range(1200):
        
        category=0
        distance1=(X[a][0]-means[0][0])**2+(X[a][1]-means[0][1])**2

        for b in range(2):
            distance = (X[a][0] - means[b+1][0]) ** 2 + (X[a][1] - means[b+1][1]) ** 2
            if distance<distance1:
                category=b+1
                distance1=distance
        assignments[a]=category
         
            
            
        
            
            
                
                
            
            
    
                


    
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    redx=[]
    redy=[]
        
    bluex=[]
    bluey=[]
        
    greenx=[]
    greeny=[]
    for a in range(1200):
        
        
        if assignments[a]==0:
            bluex.append(X[a][0])
            bluey.append(X[a][1])
            
        if assignments[a]==1:
            redx.append(X[a][0])
            redy.append(X[a][1])
        if assignments[a]==2:
            greenx.append(X[a][0])
            greeny.append(X[a][1])

    plt.scatter(greenx, greeny, c='green')
    plt.scatter(redx, redy, c='red')
    plt.scatter(bluex, bluey, c='blue')
    
    
        
            
        
    plt.axis([-8, 8, -8, 8])
    plt.show()

    
    
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)

