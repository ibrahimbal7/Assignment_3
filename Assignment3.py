# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:26:19 2023

This is the code for Applied Data Science 1 lecture Assignment 3

@author: ibrah
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import seaborn as sns
import warnings
import scipy.optimize as opt


def Barchart(df,figname):
    """
    Produces barcharts as subplot with data.
    df: data to plot
    fig_name: name of the png file
    """
    b,i=1,0
    a = len(features)
    plt.figure(figsize=(15,40))
    #loop to make subplots for each attribute
    for i in range(a):
        plt.subplot(a,2,b)                 
        temp = df.drop(columns="Year").groupby("Country").agg("mean")
        temp = temp[features[i]].sort_values(ascending = True)
        temp.plot(kind="bar")    
        plt.title(f"Average {features[i]} between 1983-2021")
        plt.xlabel("Countries")
        b=b+1
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()
    
def Lineplot(df,figname):
    """
    Produces lineplots as subplot with data.
    df: data to plot
    fig_name: name of the png file
    """
    a = len(features)
    c = len(countries)
    b,i,j = 1,0,0
    plt.figure(figsize=(15,40))
    #Loops to make subplot for each attribute with all countries
    for i in range(a):
        plt.subplot(a,2,b)        
        for j in range(c):            
            temp2 = df[j].groupby("Year").sum()
            temp2 = temp2.replace(0, np.nan)
            plt.plot(temp2[features[i]].iloc[-15:],label=countries[j])
        b=b+1
        plt.title(f"{features[i]} for last 15 Years")    
        plt.xlim(2007,2021)
        plt.xlabel('Years')  
    plt.legend(bbox_to_anchor=(2, 1),fontsize = "xx-large")
    plt.savefig(figname) 
    plt.show()
    
def norm(arr):
    """
    Array normalisation function to use for data
    arr: array to normalise
    """
    min_ = np.min(arr)
    max_ = np.max(arr)
    result = (arr-min_) / (max_-min_)
    return result

def norm_df(df):
    """Normalisating the data by making it for each array 
    using norm(arr) function
    df: dataframe to normalise
    """
    #Loop to normalise each array in data
    for i in df.columns[0:]:
        df[i] = norm(df[i])
    return df

def func(x,k,l,m):
    """Function to use for finding error ranges
    """
    k,x,l,m=0,0,0,0
    return k * np.exp(-(x-l)**2/m)

def err_ranges(x, func, param, sigma):
    """Function to find error ranges for fitted data
    x: x array of the data
    func: defined function above
    param: parameter found by fitted data
    sigma: sigma found by fitted data
    """
    import itertools as iter
    
    low = func(x, *param)
    up = low
    
    uplow = []
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        low = np.minimum(low, y)
        up = np.maximum(up, y)
        
    return low, up
   
def K_Means_Score(df,a,b):
    """Finding K_Means Score of clustered data
    df: data to cluster
    a,b : number of clusters to be find the scores
    """
    i=0
    #Loop to find b number of cluster's error starting from a
    for i in range(a, b):
    
        kmeans = cluster.KMeans(n_clusters=i)
        kmeans.fit(df)
        labels = kmeans.labels_
        print (i, "number of cluster gives:",
               skmet.silhouette_score(df, labels), "score")
       
       
def K_Means(df,n,figname):
    """Making plot from chosen number of clusters for the data
    df: data the cluster
    n: number of clusters
    figname: name of the png file to be saved
    """
    kmeans = cluster.KMeans(n_clusters=n)
    kmeans.fit(df)
    kmeans2 = cluster.KMeans(n_clusters=n)
    kmeans2.fit(df_un)
    #centers of each cluster    
    cen = kmeans2.cluster_centers_
    labels = kmeans.labels_
    df["cluster"]= labels
    print("Centers are located at", cen)
    plt.figure(figsize=(6, 6))
    #Making scatter plot with the real values(not normalised) by using 
    #normalised clusters
    sns.scatterplot(df_un["Population"],df_un["GDP"],
                    hue='cluster',data=df,cmap="Accent")
    #Loop to mark centers of each cluster
    for i in range(n):
        x, y = cen[i,:]
        plt.plot(x, y, "dk", markersize=10,marker="o",
                 color="black")
    plt.xlabel("Population")
    plt.ylabel("GDP ($)")
    plt.title(f"{n} Clusters")
    plt.savefig(figname)
    plt.show()
    
    return df,x,y,kmeans

def Expo(t, scale, growth):
    """ Exponential fitting formula function
    """
    f = scale * np.exp(growth * (t-1990))
    return f

    
data = pd.read_csv("data.csv",sep=';') 
warnings.filterwarnings("ignore")
#Defining each countries' names and making as a list 
countries = data["Country"].to_list()
countries=list(dict.fromkeys(countries))
#Making a list to take each attribute
features= data.drop(columns=["Year","Country"]).columns.to_list()
df_country = list(range(10))
#Loop to make dataframe for each country with the all attributes
for i in range(len(countries)):   
    df_country[i] = data[data["Country"]=="{}".format(countries[i])]   
#Calling Barchart function to make barchart for the data
Barchart(data,"barchart.png")
#Calling Lineplot function to make barchart for the data
Lineplot(df_country,"lineplot.png")
#Plotting scatter matrix plot to compare each attribute with others
plt.figure()
pd.plotting.scatter_matrix(data.drop(columns="Year"), figsize=(9, 9))
plt.tight_layout()
plt.savefig("scatter_matrix.png")
plt.show()
#Making a heatmap to see the correlation of attributes
corr = data.drop(columns="Year").corr()
plt.figure(figsize=(14,8))
sns.heatmap(corr)
plt.savefig("heatmap.png")
plt.show()
#making normalisation for Population and GDP attributes to use for clustering
df_norm = data[["Population","GDP"]].copy().dropna(axis=0)
df_un = data[["Population","GDP"]].copy().dropna(axis=0)
df_norm=norm_df(df_norm)
#Finding the scores of K_means clustering for 2 to 6 numbers and printing the
#result. 3 and 4 clusters are founded as best ones
K_Means_Score(df_norm,2,6)
#Plotting clusters for 3 and 4 clusters
K_Means(df_norm,3,"k_means3.png")
K_Means(df_norm,4,"k_means4.png")
#Making a dataframe for fitting Years and Inflation Rates in Turkey
df3 =pd.DataFrame(data,columns=["Year","Country","Inflation"])
df3 = df3[(df3["Country"] == 'Turkiye')]
#Making a scatter plot for the infltation rates for last 32 years in Turkey          
plt.figure()
sns.scatterplot(df3["Year"][-32:],df3["Inflation"],
                hue='Country',data=df3,cmap="Accent")
plt.title('Scatter Plot between 1990-2021 before fitting')
plt.ylabel('Inflation')
plt.xlabel('Year')
plt.xlim(1990,2021)
plt.savefig("Scatter_fit.png")
plt.show()
#Finding the necessery values for fitting data by using exponential method
popt, pcov = opt.curve_fit(Expo, df3["Year"],df3["Inflation"], p0=[1000, 0.02])
df3["Pop"] = Expo(df3["Year"], *popt)
sigma = np.sqrt(np.diag(pcov))
low, up = err_ranges(df3["Year"],Expo,popt,sigma)
#Plotting the fitted and real data by showing confidence range
plt.figure()
plt.title("Plot After Fitting")
plt.plot(df3["Year"], df3["Inflation"], label="data")
plt.plot(df3["Year"], df3["Pop"], label="fit")
plt.fill_between(df3["Year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.savefig("Fitted.png")
plt.show()
#Predicting future values
low, up = err_ranges(2030,Expo,popt,sigma)
print("Forecasted inflation in 2030 is ", low, "and", up)
low, up = err_ranges(2040,Expo,popt,sigma)
print("Forecasted inflation in 2040 is ", low, "and", up)
