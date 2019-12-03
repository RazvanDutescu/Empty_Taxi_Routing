import numpy as np
import math
import random
import gc

class taxi():
	def __init__(self, workday_length, current_time, origin, cluster_info):
		
		self.empty = True
		self.currently_at = origin
		self.origin = origin
		self.hailed = False
		self.time_to_start = True
		
		self.time = current_time
		self.time_to_retreat = False
		self.retired_for_the_day = False
		self.need_waiting_spot = True
		self.at_location = False
		
		
		self.today_profit = 0 # current profit for today
		self.gas = 200 # not used for now, but in the future assume everyone has full tank at the beginning
		self.idle_time = 0 # time taxi has waited for passager
		self.speed = 8
		self.profit = 0
		self.origin = origin
		self.destination = [0,0]
		self.destination_of_passager = [0,0]
		self.distance_travelled = 0
		self.distance_travelled_with_passager = 0
		self.time_remaining_until_retreat = workday_length * 60
		self.time_to_retreat_at = self.time + self.time_remaining_until_retreat
		if self.time_to_retreat_at > 1439:
			self.time_to_retreat_at -= 1439
		#print(self.time_to_retreat_at)
		self.time_to_start_at = self.time_to_retreat_at + (24-workday_length) * 60
		if self.time_to_start_at > 1439:
			self.time_to_start_at -= 1439
		#print(self.time_to_start_at)
		self.number_of_hails = 0
		self.waiting_time = 0
		
		self.num_of_episodes = 1
		self.profit_during_episode = 0
		self.distance_travelled_during_episode = 0
		self.waiting_time_during_episode = 0
		
		
		self.cluster_information = cluster_info
		self.state_action_dict = cluster_info.generate_state_action_dictionary(int(current_time/60),  int(current_time/60) + 8)
		
		self.num_of_clusters = self.cluster_information.get_num_of_clusters()
		self.hails = {a: 0 for a in range(0,self.num_of_clusters)}
		
			
		self.action = 0
		self.state = "00070"
		
		
		self.reward = 0
		self.exploration_rate = 0.01
		self.step_size = 0.01
		try:
		
			self.find_waiting_spot()
		except KeyError:
			print(self.state)
	def get_cluster_in(self):
		#print("cluster in " + str(self.cluster_information.get_cluster_in(self.currently_at)[0]))
		return str(self.cluster_information.get_cluster_in(self.currently_at)[0])
	def is_profit(self):
		if self.profit_during_episode > 0:
			return "1"
		else:
			return "0"
	def get_time_until_stop(self):
		#print("time until stop " +  str(int(self.time_remaining_until_retreat / 60)))
		return str(int(self.time_remaining_until_retreat / 60))
	def get_cluster_with_most_hails(self):
		
		
		max_values = 0
		max_cluster = 0
		for key, value in self.hails.items():#np.unique(results, return_counts = True):
			if max_values < value:
				max_cluster = key
		return str(max_cluster)
		
		
	
	def get_state(self):
		
		
		
		return str(self.get_cluster_in()) + str(int(self.time/60)) +str(self.is_profit()) + str(self.get_time_until_stop()) + str(self.get_cluster_with_most_hails())	
	
	#Code taken from https://github.com/jknthn/learning-rl/blob/master/td-learning.ipynb
	#Original author : Jaremi Kaczmarczyk
	
	def argmax_state_action(self):
		state_action_list = list(map(lambda x:x[1], self.state_action_dict[self.state].items()))
		indices = [i for i, x in enumerate(state_action_list) if x == max(state_action_list)]
		max_state_action = random.choice(indices)
		return max_state_action
	
	def greedy_policy(self):
		policy = {}
		for state in self.state_action_dict.keys():
			policy[state] = self.argmax_state_action()
		return policy
		
	def calculate_reward(self):
	
		self.reward = 0 
		
		#self.distance_travelled_during_episode
		average_distance_travelled_during_episode = self.distance_travelled_with_passager / self.num_of_episodes
		difference_in_distance = average_distance_travelled_during_episode - self.distance_travelled_during_episode
		
		average_waiting_time = self.waiting_time / self.num_of_episodes
		difference_in_waiting_time = average_waiting_time - self.waiting_time_during_episode
		
		#profit * difference_waiting/difference_distance
		
		average_profit = self.profit / self.num_of_episodes
		difference_in_profit =average_profit - self.profit_during_episode  
		
		return difference_in_profit * ((abs(difference_in_distance)+1) /(abs(difference_in_waiting_time)+1) )
		
		
		
		
	def find_waiting_spot(self):# r-learning algo goes here
		
		self.state = self.cluster_information.get_state(self.time, self.time_remaining_until_retreat, self.currently_at, self.is_profit(), self.hails)
		
		
		if random.uniform(0,1) < 0.05:
			self.action = random.randint(0, len(self.state_action_dict[self.state]) - 1)#random action/ exploration
		else:
			self.action = max(self.state_action_dict[self.state])# exploitaition
		self.num_of_episodes += 1		
		
		self.need_waiting_spot = False
		
		self.destination = self.cluster_information.get_destination(self.action)
		
		return True
		
		
	def pickup_passager(self):
		self.at_location = False
		self.hailed = False
		self.empty = False
		self.destination = self.destination_of_passager
		self.need_waiting_spot = False
		return True
		
	def drop_passager(self):
		self.at_location = False
		self.empty = True
		self.need_waiting_spot = True
		self.reward = self.calculate_reward()
		try:
			self.state_action_dict[self.state][self.action] = (1-self.step_size)*self.state_action_dict[self.state][self.action] + self.step_size*(self.reward + self.exploration_rate * max(self.state_action_dict[self.get_state()].values()))
		except KeyError:
			print(self.state)
			print(self.state_action_dict[self.state])
			print(self.action)
		self.profit += self.profit_during_episode
		self.profit_during_episode = 0
		self.distance_travelled_during_episode = 0
		self.waiting_time_during_episode = 0
		self.state = self.cluster_information.get_state(self.time, self.time_remaining_until_retreat, self.currently_at, self.is_profit(), self.hails)
		self.hails = {a: 0 for a in range(0,self.num_of_clusters)}
		self.find_waiting_spot()
		return True
		
	def get_position():
		return self.currently_at
		
	def get_distance(self, position):
		return math.hypot(position[0] - self.currently_at[0], position[1] - self.currently_at[1])
		
				
	def go_to_location(self):
		 
		distance_to_location = math.hypot(self.destination[0] - self.currently_at[0], self.destination[1] - self.currently_at[1])
		
		if distance_to_location < self.speed:
			self.currently_at[0] = self.destination[0]
			self.currently_at[1] = self.destination[1]
			self.at_location = True
			self.distance_travelled += distance_to_location
			self.distance_travelled_during_episode += distance_to_location 
			if not(self.empty):
				self.distance_travelled_with_passager += distance_to_location
			
			if not(self.empty):	
				self.profit_during_episode += distance_to_location * 1.5# rate is 1.5 currency for 1 uom 
			self.gas -= distance_to_location * 0.5
			if self.gas == 0:
				self.gas = 200
				self.profit_during_episode -= 300
			return True
		else:
			# taken from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
			v0 = np.array(self.destination) - np.array(self.currently_at)
			v1 = np.array([self.currently_at[0], self.currently_at[1] + 1]) - np.array(self.currently_at)
			angle_between_v = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
			self.distance_travelled += distance_to_location #% self.speed
			self.distance_travelled_during_episode += distance_to_location
			if not(self.empty):
				self.distance_travelled_with_passager += distance_to_location #% self.speed
			
			self.currently_at[0] += math.sin(angle_between_v)* self.speed 
			
			self.currently_at[1] += math.cos(angle_between_v)* self.speed
			if not(self.empty):	
				self.profit_during_episode += 7.5
			
			self.gas -= self.speed * 0.5
			if self.gas == 0:
				self.gas = 200
				self.profit_during_episode -= 300
			
			return True
				
		

	def make_available(self):
		self.available = True
		self.destination = self.find_wait_spot()
		return True
		
	def hail(self, destination, destination_of_passager, cluster_in):
		self.number_of_hails += 1
		self.hails[cluster_in] += 1
		if self.empty and not(self.hailed): #and self.get_distance(destination) < 100:#self.action == cluster_in:
			self.at_location = False
			self.destination = destination
			self.need_waiting_spot = False
			self.destination_of_passager = destination_of_passager
			self.hailed = True
			return True
		return False
		
			
	def get_stats(self):
		
		dist = self.distance_travelled
		self.distance_travelled = 0
		
		dist_pass = self.distance_travelled_with_passager
		self.distance_travelled_with_passager = 0
		
		prof = self.profit
		self.profit = 0
		
		wait = self.waiting_time
		self.waiting_time = 0
		
		
		return [dist, dist_pass, prof, wait]
		
	def check_if_time_to_start(self):
		if self.time == self.time_to_start_at:
			self.retired_for_the_day = False
			self.time_remaining_until_retreat = self.time_to_retreat_at
			self.find_waiting_spot()
		return True
	
	def clean(self):
		gc.collect()
		
		
		
	def tick(self):
		
		
		
		
		if self.time + 1 > 1440:
			self.time = 0
		else:
			self.time += 1
			
		if self.retired_for_the_day:
			self.check_if_time_to_start()
			return True
		if not(self.at_location) and self.retired_for_the_day:
			self.go_to_location()
			return True
	
		if self.time_remaining_until_retreat > 0:
			self.time_remaining_until_retreat -= 1
		else:
			if self.empty:
				self.retired_for_the_day = True
				self.at_location = False
				self.destination = self.origin
				return True

		
			
			
		
		if self.need_waiting_spot and not(self.hailed):
			#print("finding waiting spot " + str(self.need_waiting_spot))
			self.find_waiting_spot()
			return True
			
		
		if not(self.at_location):#or self.hailed: # means it's moving
			self.go_to_location()
			return True
		
		if self.at_location and not(self.hailed) and self.empty:
			if self.time_to_retreat:
				self.retreat_for_the_day()
				return True
							
		if self.at_location and self.hailed:
			self.pickup_passager()
			return True
		elif self.at_location and not(self.empty):
			self.drop_passager()
			return True
		
		self.waiting_time += 1
		self.waiting_time_during_episode += 1
		
	