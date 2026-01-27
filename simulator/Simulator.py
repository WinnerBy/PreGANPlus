from simulator.host.Host import *
from simulator.container.Container import *

class Simulator():
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, TotalPower, RouterBw, Scheduler, Recovery, Stats, ContainerLimit, IntervalTime, hostinit):
		self.totalpower = TotalPower
		self.totalbw = RouterBw
		self.hostlimit = len(hostinit)
		self.scheduler = Scheduler
		self.scheduler.setEnvironment(self)
		self.recovery = Recovery
		self.recovery.setEnvironment(self)
		self.containerlimit = ContainerLimit
		self.hostlist = []
		self.containerlist = []
		self.intervaltime = IntervalTime
		self.interval = 0
		self.inactiveContainers = []
		self.stats = Stats
		self.stats.setEnvironment(self)
		self.addHostlistInit(hostinit)
		
		# 故障检测配置（基于论文ADE定义）
		self.fault_config = {
			'cpu_threshold': 90.0,      # CPU过载阈值（%）
			'ram_threshold': 90.0,       # RAM过载阈值（%）
			'network_threshold': 90.0,   # 网络过载阈值（%）
			'fault_duration_required': 60.0,  # 持续60秒
		}
		
		# 故障持续时间跟踪器：{host_id: {'cpu': duration_seconds, 'ram': duration_seconds, ...}}
		self.fault_duration_tracker = {}
		
		# 故障历史记录：{interval: {host_id: fault_type}}
		self.fault_history = {}

	def addHostInit(self, IPS, RAM, Disk, Bw, Latency, Powermodel):
		assert len(self.hostlist) < self.hostlimit
		host = Host(len(self.hostlist), IPS, RAM, Disk, Bw, Latency, Powermodel, self)
		self.hostlist.append(host)

	def addHostlistInit(self, hostList):
		assert len(hostList) == self.hostlimit
		for IPS, RAM, Disk, Bw, Latency, Powermodel in hostList:
			self.addHostInit(IPS, RAM, Disk, Bw, Latency, Powermodel)

	def addContainerInit(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
		container = Container(len(self.containerlist), CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID = -1)
		self.containerlist.append(container)
		return container

	def addContainerListInit(self, containerInfoList):
		deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
		deployedContainers = []
		for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
			dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
			deployedContainers.append(dep)
		self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
		return [container.id for container in deployedContainers]

	def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
		for i,c in enumerate(self.containerlist):
			if c == None or not c.active:
				container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID = -1)
				self.containerlist[i] = container
				return container

	def addContainerList(self, containerInfoList):
		deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
		deployedContainers = []
		for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
			dep = self.addContainer(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
			deployedContainers.append(dep)
		return [container.id for container in deployedContainers]

	def getContainersOfHost(self, hostID):
		containers = []
		for container in self.containerlist:
			if container and container.hostid == hostID:
				containers.append(container.id)
		return containers

	def getContainerByID(self, containerID):
		return self.containerlist[containerID]

	def getContainerByCID(self, creationID):
		for c in self.containerlist + self.inactiveContainers:
			if c and c.creationID == creationID:
				return c

	def getHostByID(self, hostID):
		return self.hostlist[hostID]

	def getCreationIDs(self, migrations, containerIDs):
		creationIDs = []
		for decision in migrations:
			if decision[0] in containerIDs: creationIDs.append(self.containerlist[decision[0]].creationID)
		return creationIDs

	def getPlacementPossible(self, containerID, hostID):
		container = self.containerlist[containerID]
		host = self.hostlist[hostID]
		ipsreq = container.getBaseIPS()
		ramsizereq, ramreadreq, ramwritereq = container.getRAM()
		disksizereq, diskreadreq, diskwritereq = container.getDisk()
		ipsavailable = host.getIPSAvailable()
		ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()
		disksizeav, diskreadav, diskwriteav = host.getDiskAvailable()
		return (ipsreq <= ipsavailable and \
				ramsizereq <= ramsizeav and \
				# ramreadreq <= ramreadav and \
				# ramwritereq <= ramwriteav and \
				disksizereq <= disksizeav \
				# diskreadreq <= diskreadav and \
				# diskwritereq <= diskwriteav
				)

	def addContainersInit(self, containerInfoListInit):
		self.interval += 1
		deployed = self.addContainerListInit(containerInfoListInit)
		return deployed

	def allocateInit(self, decision):
		migrations = []
		routerBwToEach = self.totalbw / len(decision)
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			assert container.getHostID() == -1
			numberAllocToHost = len(self.scheduler.getMigrationToHost(hid, decision))
			allocbw = min(self.getHostByID(hid).bwCap.downlink / numberAllocToHost, routerBwToEach)
			if self.getPlacementPossible(cid, hid):
				if container.getHostID() != hid:
					migrations.append((cid, hid))
				container.allocateAndExecute(hid, allocbw)
			# destroy pointer to this unallocated container as book-keeping is done by workload model
			else: 
				self.containerlist[cid] = None
		return migrations

	def destroyCompletedContainers(self):
		destroyed = []
		for i,container in enumerate(self.containerlist):
			if container and container.getBaseIPS() == 0:
				container.destroy()
				self.containerlist[i] = None
				self.inactiveContainers.append(container)
				destroyed.append(container)
		return destroyed

	def getNumActiveContainers(self):
		num = 0 
		for container in self.containerlist:
			if container and container.active: num += 1
		return num

	def getSelectableContainers(self):
		selectable = []
		for container in self.containerlist:
			if container and container.active and container.getHostID() != -1:
				selectable.append(container.id)
		return selectable

	def addContainers(self, newContainerList):
		self.interval += 1
		destroyed = self.destroyCompletedContainers()
		deployed = self.addContainerList(newContainerList)
		return deployed, destroyed

	def getActiveContainerList(self):
		return [c.getHostID() if c and c.active else -1 for c in self.containerlist]

	def getContainersInHosts(self):
		return [len(self.getContainersOfHost(host)) for host in range(self.hostlimit)]

	def detect_faults(self):
		"""
		检测当前interval的故障状态（基于ADE逻辑）
		返回：{host_id: fault_type}，fault_type可以是'cpu', 'ram', 'network', 或None
		"""
		current_faults = {}
		
		for host in self.hostlist:
			host_id = host.id
			
			# 初始化跟踪器
			if host_id not in self.fault_duration_tracker:
				self.fault_duration_tracker[host_id] = {
					'cpu': 0.0,
					'ram': 0.0,
					'network': 0.0
				}
			
			# 获取当前资源使用情况
			cpu_usage = host.getCPU()  # 百分比
			ram_size, _, _ = host.getCurrentRAM()
			ram_usage = 100.0 * (ram_size / host.ramCap.size)  # 百分比
			
			# 检查CPU过载（持续60秒）
			if cpu_usage > self.fault_config['cpu_threshold']:
				self.fault_duration_tracker[host_id]['cpu'] += self.intervaltime
			else:
				self.fault_duration_tracker[host_id]['cpu'] = 0.0
			
			# 检查RAM过载（持续60秒）
			if ram_usage > self.fault_config['ram_threshold']:
				self.fault_duration_tracker[host_id]['ram'] += self.intervaltime
			else:
				self.fault_duration_tracker[host_id]['ram'] = 0.0
			
			# 判断是否发生故障（持续60秒）
			if self.fault_duration_tracker[host_id]['cpu'] >= self.fault_config['fault_duration_required']:
				current_faults[host_id] = 'cpu'
			elif self.fault_duration_tracker[host_id]['ram'] >= self.fault_config['fault_duration_required']:
				current_faults[host_id] = 'ram'
			# TODO: 网络过载检测（需要根据实际情况实现）
			# elif self.fault_duration_tracker[host_id]['network'] >= self.fault_config['fault_duration_required']:
			#     current_faults[host_id] = 'network'
		
		# 记录故障历史
		self.fault_history[self.interval] = current_faults
		
		return current_faults
	
	def get_fault_history(self):
		"""获取故障历史记录"""
		return self.fault_history
	
	def simulationStep(self, decision):
		routerBwToEach = self.totalbw / len(decision) if len(decision) > 0 else self.totalbw
		migrations = []
		containerIDsAllocated = []
		for (cid, hid) in decision:
			container = self.getContainerByID(cid)
			currentHostID = self.getContainerByID(cid).getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			migrateFromNum = len(self.scheduler.getMigrationFromHost(currentHostID, decision))
			migrateToNum = len(self.scheduler.getMigrationToHost(hid, decision))
			allocbw = min(targetHost.bwCap.downlink / migrateToNum, currentHost.bwCap.uplink / migrateFromNum, routerBwToEach)
			if hid != self.containerlist[cid].hostid and self.getPlacementPossible(cid, hid):
				migrations.append((cid, hid))
				container.allocateAndExecute(hid, allocbw)
				containerIDsAllocated.append(cid)
		# destroy pointer to unallocated containers as book-keeping is done by workload model
		for (cid, hid) in decision:
			if self.containerlist[cid].hostid == -1: self.containerlist[cid] = None
		for i,container in enumerate(self.containerlist):
			if container and i not in containerIDsAllocated:
				container.execute(0)
		return migrations