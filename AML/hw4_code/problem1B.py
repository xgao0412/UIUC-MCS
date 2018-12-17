from sklearn.cluster import KMeans
sd=[]
for i in range(1,20):
    model = KMeans(n_clusters=i).fit(df)
    sd.append(model.inertia_)
    
plt.plot(range(1,20),sd)
plt.xlabel('number of clusters')
plt.ylabel('Sum of squared distances')
plt.xticks(range(1,20))
plt.title('Sum of Squared distances for different number of clusters')