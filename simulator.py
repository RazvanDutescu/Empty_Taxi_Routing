import pandas as pd
import taxi_agent as taxi
import passager  as pas
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import cluster_information as cl_info
import matplotlib.pyplot as plt
import gc
import time
import sys


num_of_taxis = 20
cluster_type = "kmeans"
num_of_clusters = 1

if len(sys.argv) == 4:
	num_of_taxis = sys.argv[1]
	cluster_type = str(sys.argv[2])
	num_of_clusters = sys.argv[3]
else:
	print("using defaults")
	num_of_taxis = 20
	cluster_type = "kmeans"
	num_of_clusters = 1

#generate probabilities

prob_mat = np.zeros((601, 600, 24))
f = open('prob_file.txt', 'r+')
lines = [line for line in f.readlines()]
len(lines)
np.random.seed(1)


i = 0
j = 0
k = 0
index = 0
while k < 24:
    while j < 600:
        while i < 601:
            prob_mat[i,j,k] = lines[index]
            index += 1
            i += 1
        i = 0
        j += 1
    j = 0
    k += 1
	
	
threshold = 0.40
new_data = [[0,0,0]]
for i in range(0,24):
    new_mat = np.random.randint(100, size = (600, 601))
    new_mat = np.dot(prob_mat[:,:,i], new_mat)
    for j in range(0,601):
        for k in range(0,600):
            if prob_mat[j,k,i] != 0:
                if new_mat[j,k] > threshold:
                    new_data = np.append( new_data,[[j,k,i]], axis = 0)
					


np.random.shuffle(new_data)
number_of_taxis = num_of_taxis # can be changed TODO: add as parameter
taxis = []


cluster_info = cl_info.cluster_information(cluster_type, num_of_clusters)
for i in range(0,number_of_taxis):
	taxis.append(taxi.taxi(8,0,[new_data[i][0],new_data[i][1]], cluster_info))
passagers = []
	
passagers_who_got_bored_and_cancelled = 0

daily_profit = []
total_passengers_served = []
total_distance_travelled = []
total_distance_travelled_with_passenger = []
total_pass_wait_time = []

for h in range(0, 30):
	np.random.seed(h)

	i = 0
	j = 0
	k = 0
	index = 0
	while k < 24:
	    while j < 600:
	        while i < 601:
	            prob_mat[i,j,k] = lines[index]
	            index += 1
	            i += 1
	        i = 0
	        j += 1
	    j = 0
	    k += 1
		
		
	threshold = 0.35
	new_data = [[0,0,0]]
	for i in range(0,24):
	    new_mat = np.random.randint(80, size = (600, 601))
	    new_mat = np.dot(prob_mat[:,:,i], new_mat)
	    for j in range(0,601):
	        for k in range(0,600):
	            if prob_mat[j,k,i] != 0:
	                if new_mat[j,k] > threshold:
	                    new_data = np.append( new_data,[[j,k,i]], axis = 0)

	pass_wait_time = 0
	for i in range(0,24):
		print("current hour: " + str(i))
		passagers_this_hour = new_data[new_data[:,2]==i,:]
		#passagers_this_hour = np.random.shuffle(passagers_this_hour)
		np.random.shuffle(passagers_this_hour)
		number_of_passagers_this_hour = int(len(passagers_this_hour)/2)
		print("number of passagers this hour: " + str(number_of_passagers_this_hour))
		new_passagers_per_minute = int(number_of_passagers_this_hour / 60)
		t = time.time()
		for j in range(0,60):
			#print(j)
			passagers_this_minute = passagers_this_hour[new_passagers_per_minute*j: new_passagers_per_minute*(j+1)]
			passagers_destinations_this_minute = passagers_this_hour[new_passagers_per_minute*j+number_of_passagers_this_hour: new_passagers_per_minute*(j+1)+number_of_passagers_this_hour]
			#for each new passager, add to list of waiting passagers
			
			for current_passager in range(0,len(passagers_this_minute)):
			#for current_passager in passagers_this_minute:
				#print(current_passager)
				passagers.append(pas.passager([passagers_this_minute[current_passager][0], passagers_this_minute[current_passager][1]],[passagers_destinations_this_minute[current_passager][0],passagers_destinations_this_minute[current_passager][1]],cluster_info))
			for passager in passagers:
				if not(passager.tick()):
					#del passager
					passagers.remove(passager)
					passagers_who_got_bored_and_cancelled += 1
					continue
				for taxi in taxis:
					if taxi.hail(passager.get_location(), passager.get_destination(),passager.part_of_cluster):
						#remove from list
						#del passager
						pass_wait_time += passager.get_waiting_time()
						passagers.remove(passager)
						break
			for taxi in taxis:
				taxi.tick()
			cluster_info.update()
		#gc.collect()
		print("elapsed time: " + str(time.time()-t))
			
		#for taxi in taxis:
		#	taxi.clean()
	total_profit = 0
	distance_travelled = 0
	distance_travelled_with_passenger = 0
	for taxi in taxis:
		a = taxi.get_stats()
		print(a)
		total_profit += a[2]
		
		distance_travelled += a[0]
		distance_travelled_with_passenger = a[1]
	daily_profit.append(total_profit)
	total_distance_travelled.append(distance_travelled)
	total_distance_travelled_with_passenger.append(distance_travelled_with_passenger)
	total_pass_wait_time.append(pass_wait_time)
	print("total profit for day " + str(h) + " is " + str(total_profit))
	print(len(passagers))
	print(passagers_who_got_bored_and_cancelled)
	total_passengers_served.append(len(new_data)-passagers_who_got_bored_and_cancelled)
	passagers = []
	passagers_who_got_bored_and_cancelled = 0
	
	
plt.figure(1)
plt.plot(daily_profit)
plt.xlabel("Days")
plt.ylabel("Profit")
plt.title("Daily profit over " + str(h+1) + " days using K-means clustering with "  +str(cluster_info.get_num_of_clusters()) + "clusters ")
#plt.show()

plt.figure(2)
plt.plot(total_distance_travelled)
plt.xlabel("Days")
plt.ylabel("Distance travelled")
plt.title("Distance travelled over " + str(h+1) + " days using K-means clustering with "  +str(cluster_info.get_num_of_clusters()) + " clusters")
#plt.show()

plt.figure(3)
plt.plot(total_distance_travelled_with_passenger)
plt.xlabel("Days")
plt.ylabel("Distance travelled with passengers")
plt.title("Distance travelled with passenger over " + str(h+1) + " days using K-means clustering with "  +str(cluster_info.get_num_of_clusters()) + " clusters")
#plt.show()

plt.figure(4)
plt.plot(total_passengers_served)
plt.xlabel("Days")
plt.ylabel("Passengers picked up")
plt.title("Total number of passenger picked up over " + str(h+1) + " days using K-means clustering with "  +str(cluster_info.get_num_of_clusters()) + " clusters")
plt.show()

plt.figure(5)
plt.plot(total_pass_wait_time)
plt.xlabel("Days")
plt.ylabel("Passengers waiting time")
plt.title("Total passenger waiting time over " + str(h+1) + " days using K-means clustering with "  +str(cluster_info.get_num_of_clusters()) + " clusters")
plt.show()

