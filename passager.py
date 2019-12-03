class passager():
	def __init__(self, location, destination, cl_info ):
		self.location = location
		self.destination = destination
		self.been_picked_up = False
		self.time_waiting = 0
		
		self.cl_info = cl_info
		self.part_of_cluster =  self.get_part_of_cluster()
	def tick(self):
		self.time_waiting += 1
		if self.time_waiting == 20:
			return False
		return True
		
	def get_waiting_time(self):
		return self.time_waiting
	def get_location(self):
		return self.location
	def get_destination(self):
		return self.destination
	def get_part_of_cluster(self):
		return self.cl_info.get_cluster_in(self.destination)[0]
	