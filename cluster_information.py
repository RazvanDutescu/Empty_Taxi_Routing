import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN,  KMeans
import random
from scipy import stats
import gc


class cluster_information():

	def read_mat(self,remove_outliers = True):
		dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
		train_df = pd.read_csv("./train.csv",
                     parse_dates=[ 'pickup_datetime'],
                     date_parser=dateparse)
		#adding weekday and hour fields for easy grouping
		train_df['pickup_day'] = train_df['pickup_datetime'].apply(lambda x: x.weekday())
		train_df['pickup_hour'] = train_df['pickup_datetime'].apply(lambda x: x.hour)

		train_df["pickup_longitude"] = (train_df["pickup_longitude"]  -train_df["pickup_longitude"].mean())/train_df["pickup_longitude"].std()
		train_df["pickup_latitude"] = (train_df["pickup_latitude"]  -train_df["pickup_latitude"].mean())/train_df["pickup_latitude"].std()

		#expand
		train_df["pickup_longitude"] *= 100
		train_df["pickup_latitude"] *= 100

		#get only what you need
		new_df = train_df.filter(["pickup_latitude", "pickup_longitude", "pickup_hour", "pickup_day"], axis = 1)
		if remove_outliers:
			#remove outliers
			new_df = new_df[(np.abs(stats.zscore(new_df)) < 3).all(axis = 1)]

		data_friday = new_df["pickup_day"] == 5 # can be changed for other day
		friday_df = new_df[data_friday]
		a = friday_df.values
		a = np.delete(a, np.s_[3], axis = 1)
	
		np.random.shuffle(a)
		a = a[1:10000]
		return a
		
		
	def DBSCANCluster(self, a):
		db = DBSCAN()
		db.fit(a)
		
		return db
		
	def MeanShiftCluster(self,a):

		bandwidth = estimate_bandwidth(a, quantile=0.5, n_samples = 1000)
		ms = MeanShift(bandwidth= bandwidth)
		ms.fit(a)
		
		return ms
		
	def KMeansCluster(self, a, num_of_clusters):
		km = KMeans(n_clusters = num_of_clusters)
		km.fit(a)
		
		return km
		

	def generate_state_action_dictionary(self, start, stop):
		
		dictionary = {}
		
		for i in range(start,stop+1):
			b = self.data[self.data[:,2] == i,:]
			results = self.model.predict(b)
			clusters_for_hour = np.unique(results)
			for j in range(0,self.num_of_clusters):#clusters_for_hour:
				for k in range(0,self.num_of_clusters):#clusters_for_hour:
					for h_till_stop in range(0,9):
						for profit in range(0,2):
							key = str(j) + str(i) + str(profit) + str(h_till_stop) + str(k)
							dictionary[key] = {a: 0.0 for a in range(0,self.num_of_clusters)}
		return dictionary

	
	def __init__(self, cluster_type, num_of_clusters = 1):
		#based on cluster_type, cluster data
		#self.number_of_clusters = 1
		self.time = 0
		
		if(cluster_type == "meanshift"):
			self.data = self.read_mat()
			self.model = self.MeanShiftCluster(self.data)
		elif(cluster_type == "dbscan"):
			self.data = self.read_mat(False)
			self.model = self.MeanShiftCluster(self.data)
		elif(cluster_type == "kmeans"):
			self.data = self.read_mat()
			self.model = self.KMeansCluster(self.data, num_of_clusters)
		elif(cluster_type == "agglomerative"):
			self.data = self.read_mat()
			self.model = self.AggCluster(self.data, num_of_clusters)
		print("clustering done")
		
		#elif an so on
		#print(self.model)
		self.cluster_labels = np.unique(self.model.labels_)
		self.num_of_clusters = len(self.cluster_labels)
		print("there are " + str(self.num_of_clusters) + " clusters")
		#self.state_action_dictionary = self.generate_state_action_dictionary()
		print("state action dictionary done")
		self.state_action_dict = self.generate_state_action_dictionary(0,8)
		
	def get_cluster_in(self, position):
		return self.model.predict([[position[0], position[1], self.time]])
		
	def get_num_of_clusters(self):
		return self.num_of_clusters
					
		
	def get_state_action_dictionary(self):
		return self.state_action_dict
	
	def update(self):
		#gc.collect()
		if self.time == 1440:
			self.time = 0
			return True
		self.time += 1
		return True
			
	def get_highest_cluster(self, hails):
		results = self.model.predict(hails)
		if len(results) == 1:
			return results
		unique, counts = np.unique(results, return_counts=True)
		result_dict = dict(zip(unique, counts))
		max_values = 0
		max_cluster = 0
		for key, value in result_dict.items():#np.unique(results, return_counts = True):
			if max_values < value:
				max_cluster = key
		return max_cluster
		
	def get_destination(self, cluster):
		return self.model.cluster_centers_[cluster][0:2]
		
		
		
	def get_state(self, time, time_till_stop, position, is_profit, hails):
		cluster_in = self.model.predict([[position[0], position[1], self.time]])
		
		max_values = 0
		max_cluster = 0
		for key, value in hails.items():#np.unique(results, return_counts = True):
			if max_values < value:
				max_cluster = key
				
		return str(cluster_in[0]) + str(int(self.time/60)) + str(is_profit) + str(int(time_till_stop/60)) + str(max_cluster)