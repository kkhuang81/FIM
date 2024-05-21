import numpy as np
import scipy.special as sp
from sklearn.metrics import f1_score
from os.path import exists
import heapq
import math
import random
import time
from model import MyModel
import glob, os
import torch
import numba

TimeHorizon = 20

def BCELoss(pred_y, true_y):
    cross_loss=torch.mean(-true_y*torch.log(torch.clip(pred_y,1e-10,1.0))-(1.0-true_y)*torch.log(torch.clip((1.0-pred_y),1e-10,1.0)))
    return cross_loss

def MAE(pred_y, true_y):
	return torch.mean(torch.abs(pred_y-true_y)).item()

def parse(cascade, adjlist, v):
	vtime=cascade[v]
	if vtime==np.inf:
		return None, 0.0
	InNodeList=[]
	Ttime=0.0
	for node in adjlist:
		if cascade[node]<vtime:
			InNodeList.append(node)
			if cascade[node]>Ttime:
				Ttime=cascade[node]
	if len(InNodeList)==0:
		return None, 0.0
	Ttime=(vtime-Ttime)/len(InNodeList)
	return InNodeList, Ttime 

def InitialWeight(cascades_time, weights, adj, N):	
	nsample=cascades_time.shape[0]	

	adjlists=[]  
	for i in range(N):
		adjlists.append(np.nonzero(adj[i])[0])	
	
	cnt=np.full((N,N),0, dtype=float)

	for s in range(nsample):
		for v in range(N):			
			InNodeList, time=parse(cascades_time[s], adjlists[v], v)
			if InNodeList==None:
				continue
			for node in InNodeList:				
				cnt[v][node]+=1
				weights[v][node]=weights[v][node]/cnt[v][node]*(cnt[v][node]-1)+time/cnt[v][node]
	for v in range(N):
		for node in adjlists[v]:
			if weights[v][node]>0:
				weights[v][node]=1.0/weights[v][node] 	

def CreateFile(folder, tau):
	cascades_time=np.load(folder+'/cascades_time.npy')
	nsamples=cascades_time.shape[0]
	nNodes=cascades_time.shape[1]
	ncut=int(TimeHorizon/tau)+1  
	cascades=np.empty([nsamples,ncut,nNodes])	
	t=np.arange(0,TimeHorizon+tau,tau)
	for i in range(ncut):
		cascades[:,i,:]=np.multiply((cascades_time<=t[i]), 1)
	np.save(folder+'/cascades-'+str(tau)+'.npy', cascades)
	return

def LoadGraph(folder, tau, train_per, val_per):
	DataFile=folder+'/cascades-'+str(tau)+'.npy'
	data_exists=exists(DataFile)
	if data_exists==False:
		CreateFile(folder, tau)
		print('File Created!')
	Data=np.load(DataFile)
	Tsize=Data.shape[0]
	TrainData=Data[:int(Tsize*train_per)]
	ValData=Data[int(Tsize*train_per):int(Tsize*(train_per+val_per))]
	TestData=Data[int(Tsize*(train_per+val_per)):]			
	return TrainData, ValData, TestData

import Diffusion
class InfEst:
	def __init__(self, weightfile, threhold, T, SeedSet):
		self.T=T
		self.SeedSet=SeedSet		
		self.edges, self.weights, self.nNodes=self.Preprocess(weightfile, threhold)

	def Preprocess(self, weightfile, threshold):
		weight=np.load(weightfile)
		nNodes=weight.shape[0]			
		pos=np.argwhere(weight>=threshold)		
		newweight=[]
		for (row, col) in pos:		
			newweight.append(weight[row, col])  
		return pos, np.array(newweight), nNodes		

	def Est(self, theta):
		t1=time.time()
		edges, weights, SeedSet = Diffusion.uintmat(self.edges.tolist()), Diffusion.doublevector(self.weights.tolist()), Diffusion.uintvector(self.SeedSet) 
		cnt=Diffusion.Diffusion(edges, weights, SeedSet, self.T, theta, self.nNodes, 1)		
		return cnt/theta

class InfMax:
	def __init__(self, weightfile, threshold, T, K):		
		self.T=T
		self.K=K
		self.graph, self.nNodes=self.Preprocess(weightfile, threshold)
		self.RIG=[[] for i in range(self.nNodes)]
		self.RIGT=[]
		self.RIG2=[[] for i in range(self.nNodes)]
		self.RIGT2=[]
		self.SeedSet=[]

	def Preprocess(self, weightfile, threshold):
		weight=np.load(weightfile)
		nNodes=weight.shape[0]
		graph=[[] for i in range(nNodes)]	
		Pos=np.argwhere(weight>=threshold)		
		for (row, col) in Pos:		
			graph[col].append((row, weight[row, col]))  # construct reverse graph
		return graph, nNodes
	@numba.jit(forceobj=True, parallel=True)
	def GenRIS(self, RInum):
		idx=len(self.RIGT)
		for i in numba.prange(idx, RInum):
			starting_vertex=random.randint(0, self.nNodes-1)
			self.GenRI(starting_vertex, i)

	def GenRI(self, starting_vertex, iid):
		distances = np.array([float('infinity')]*self.nNodes)
		distances[starting_vertex] = 0.0
		pq = [(0, starting_vertex)]
		RI=[]		

		while len(pq) > 0:
			
			current_distance, current_vertex = heapq.heappop(pq)
			if current_distance > self.T:
				break
			if current_distance > distances[current_vertex]:
				continue		
			
			RI.append(current_vertex)
			self.RIG[current_vertex].append(iid)

			for neighbor, weight in self.graph[current_vertex]:
				weight=-np.log(np.random.uniform(0.0,1.0))/weight
				distance = current_distance + weight

				if distance < distances[neighbor] and distance <= self.T:
					distances[neighbor] = distance
					heapq.heappush(pq, (distance, neighbor))
		
		self.RIGT.append(RI)
		return
	
	def GenRI2(self, starting_vertex, iid):
		distances = np.array([float('infinity')]*self.nNodes)
		distances[starting_vertex] = 0.0
		pq = [(0, starting_vertex)]
		RI=[]		   
		while len(pq) > 0:
			current_distance, current_vertex = heapq.heappop(pq)
			if current_distance > self.T:
				break
			if current_distance > distances[current_vertex]:				
				continue		

			RI.append(current_vertex)
			self.RIG2[current_vertex].append(iid)

			for neighbor, weight in self.graph[current_vertex]:
				weight=-np.log(np.random.uniform(0.0,1.0))/weight
				distance = current_distance + weight

				if distance < distances[neighbor] and distance <= self.T:
					distances[neighbor] = distance
					heapq.heappush(pq, (distance, neighbor))
		
		self.RIGT2.append(RI)
		return

	def SelectSeedSet(self):
		self.SeedSet.clear()
		coverage=np.array([0]*self.nNodes)
		maxDeg=0
		for i in range(self.nNodes-1, -1, -1):  
			deg=len(self.RIG[i])
			coverage[i]=deg
			maxDeg=max(deg, maxDeg)
		degMap=[[] for i in range(maxDeg+1)]
		for i in range(self.nNodes-1, -1, -1):  
			degMap[coverage[i]].append(i)
		sortedNode=[0]*self.nNodes
		nodePosition=np.array([0]*self.nNodes)
		degreePosition=np.array([0]*(maxDeg+2))
		idxSort=0
		idxDegree=0
		for nodes in degMap:
			degreePosition[idxDegree + 1] = degreePosition[idxDegree]+len(nodes)
			idxDegree+=1
			for node in nodes:
				nodePosition[node] = idxSort
				sortedNode[idxSort] = node
				idxSort+=1
		edgeMark=np.array([False]*len(self.RIGT))
		sumTopk=0
		for deg in range(maxDeg,-1,-1):
			if degreePosition[deg]<=self.nNodes-self.K:
				sumTopk += deg * (degreePosition[deg + 1] - (self.nNodes - self.K))
				break
			sumTopk+=deg * (degreePosition[deg + 1] - degreePosition[deg])
		boundMin=1.0*sumTopk
		sumInf=0
		for k in range(self.K-1, -1, -1):
			seed=sortedNode.pop()
			newNumV = len(sortedNode)
			sumTopk += coverage[sortedNode[newNumV - self.K]] - coverage[seed]
			sumInf += coverage[seed]
			self.SeedSet.append(seed)			
			coverage[seed] = 0
			for edgeIdx in self.RIG[seed]:
				if len(self.RIGT[edgeIdx])==0 or edgeMark[edgeIdx]:
					continue
				edgeMark[edgeIdx] = True
				for nodeIdx in self.RIGT[edgeIdx]:
					if coverage[nodeIdx] == 0:
						continue
					currPos = nodePosition[nodeIdx]
					currDeg = coverage[nodeIdx]
					startPos = degreePosition[currDeg]
					startNode = sortedNode[startPos]
					sortedNode[currPos], sortedNode[startPos]=sortedNode[startPos], sortedNode[currPos]
					nodePosition[nodeIdx] = startPos
					nodePosition[startNode] = currPos
					degreePosition[currDeg]+=1
					coverage[nodeIdx]-=1
					if startPos >= newNumV - self.K: 
						sumTopk-=1
			boundLast=1.0*(sumInf + sumTopk)
			if boundMin > boundLast: 
				boundMin = boundLast
		return boundMin
	@numba.jit(forceobj=True, parallel=True)
	def cover(self, sample):
		len2=len(self.RIGT2)
		for idx in numba.prange(len2, sample):
			starting_vertex=random.randint(0, self.nNodes-1)
			self.GenRI2(starting_vertex, idx)
			idx+=1
		overlap=np.array([False]*idx)
		for seed in self.SeedSet:			
			overlap[self.RIG2[seed]]=True
		return overlap.sum()

		
	def InfMax(self, delta, epsilon, IniSample, factor=1.0-1.0/np.e):
    		
		alpha = np.sqrt(np.log(6. / delta))
		beta = np.sqrt(factor*(sp.comb(self.nNodes, self.K) + np.log(6.0 / delta)))

		theta_max = 2 * sqr(factor*alpha + beta)*self.nNodes/self.K / epsilon / epsilon

		i_max = math.ceil(np.log(theta_max / IniSample) / np.log(2.0)) + 1
		ai = np.log(3 * i_max / delta)
		sample = int(IniSample)
		for i in range(i_max):
			t1=time.time()
			self.GenRIS(sample)						
			t1=time.time()
			self.SeedSet.clear()
			influence=self.SelectSeedSet()
			t1=time.time()
			cover=self.cover(sample)
			lower = sqr(np.sqrt(cover + 2. * ai / 9.) - np.sqrt(ai / 2.)) - ai / 18
			upper = sqr(np.sqrt(influence + ai / 2.) + np.sqrt(ai / 2.))

			if lower / upper > 1.0 - 1.0/np.e - epsilon:
				return self.SeedSet, cover/sample*self.nNodes
			sample*=2	

		self.SeedSet.clear()
		self.SelectSeedSet()
		return self.SeedSet, cover/sample*self.nNodes

def sqr(t):
	return t*t

def InfEstbyModel(nNodes, SeedID):
	model=MyModel(nNodes=nNodes)	
	checkpt_file=glob.glob('pretrained/*.pt')[0]
	model.load_state_dict(torch.load(checkpt_file))
	model.eval()
	output=np.zeros([1,nNodes])
	output[0,SeedID]=1.0
	output = torch.FloatTensor(output)
	with torch.no_grad():
		for slot in range(1,SlotNum):
			output = model(output, args.tau)
	return output.sum().item()
