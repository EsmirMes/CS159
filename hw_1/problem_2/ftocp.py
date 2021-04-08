import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from dataclasses import dataclass, field


class FTOCP(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0 and terminal contraints
		- buildNonlinearProgram: builds the ftocp program solved by the above solve method
		- model: given x_t and u_t computes x_{t+1} = f( x_t, u_t )

	"""

	def __init__(self, N, A, B, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, printLevel):
		# Define variables
		self.printLevel = printLevel

		self.A  = A
		self.B  = B
		self.N  = N
		self.n  = A.shape[1]
		self.d  = B.shape[1]
		self.Fx = Fx
		self.bx = bx
		self.Fu = Fu
		self.bu = bu
		self.Ff = Ff
		self.bf = bf
		self.Q  = Q
		self.Qf = Qf
		self.R  = R

		print("Initializing FTOCP")
		ftocp.buildCost()
		ftocp.buildIneqConstr()
		ftocp.buildEqConstr()
		print("Done initializing FTOCP")

		self.time = 0


	def solve(self, x0):
		"""Computes control action
		Arguments:
		    x0: current state
		"""

		# Solve QP
		startTimer = datetime.datetime.now()
		self.osqp_solve_qp(self.H, self.q, self.G_in, np.add(self.w_in, np.dot(self.E_in,x0)), self.G_eq, np.dot(self.E_eq,x0) )
		endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
		self.solverTime = deltaTimer
		
		# Unpack Solution
		self.unpackSolution(x0)

		self.time += 1

		return self.uPred[0,:]

	def unpackSolution(self, x0):
		# Extract predicted state and predicted input trajectories
		self.xPred = np.vstack((x0, np.reshape((self.Solution[np.arange(self.n*(self.N))]),(self.N,self.n))))
		self.uPred = np.reshape((self.Solution[self.n*(self.N)+np.arange(self.d*self.N)]),(self.N, self.d))

		if self.printLevel >= 2:
			print("Optimal State Trajectory: ")
			print(self.xPred)

			print("Optimal Input Trajectory: ")
			print(self.uPred)

		if self.printLevel >= 1: print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")

	def buildIneqConstr(self):
		# Hint 1: consider building submatrices and then stack them together
		# Hint 2: most likely you will need to use auxiliary variables 
		...
		G_in = ...

		...
		E_in = ...
		
		...
		w_in = ...

		if self.printLevel >= 2:
			print("G_in: ")
			print(G_in)
			print("E_in: ")
			print(E_in)
			print("w_in: ", w_in)			

		self.G_in = sparse.csc_matrix(G_in)
		self.E_in = E_in
		self.w_in = w_in.T

	def buildCost(self):
		# Hint: you could use the function "linalg.block_diag"
		"""
		A = [[1, 1], [0, 1]]
		B = [[0], [1]]
		U_0 has dim: N x 1
			goes from u_1 to u_N
		X_0 has dim: 2N x 1
			goes from x_1 to x_N
		x(0) has dim: 2 x 1
		[[X_0], [U_0]]] has dim 3N x 1
		G_{0, eq} has dim: 2N x 3N
			creates dynamics constraints from time 1 to N
		E_{0, eq} has dim: 2N x 2
		"""
		barQ = ...
		barR = ...

		H = linalg.block_diag(barQ, barR)
		q = np.zeros(H.shape[0]) 

		if self.printLevel >= 2:
			print("H: ")
			print(H)
			print("q: ", q)
		
		self.q = q
		self.H = sparse.csc_matrix(2 * H)  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

	def buildEqConstr(self):
		"""
		A = [[1, 1], [0, 1]]
		B = [[0], [1]]
		Fx = [[1, 0], [0, 1],[-1, 0], [0, -1]] has dim: 4 x 2
		bx = [15, 15, 15, 15] has dim: 4
		Fu = [[1], [-1]] has dim: 2 x 1
		bu = [5, 5] has dim: 2
		Ff = [[1, 0], [0, 1],[-1, 0], [0, -1]] has dim: 4 x 2
		bf = [15, 15, 15, 15] has dim: 4
		U_0 has dim: N x 1
			goes from u_1 to u_N
		X_0 has dim: 2N x 1
			goes from x_1 to x_N
		x(0) has dim: 2 x 1
		[[X_0], [U_0]]] has dim 3N x 1
		x(0) has dim: 2 x 1

		G_{0, in} has dim: (4 + 4N + 2N) x 3N
		E_{0, in} has dim: (4N + 4 + 2N) x 2
		w_{0, in} has dim: (4N + 4 + 2N) x 1
		"""
		# Hint 1: consider building submatrices and then stack them together
		# Hint 2: most likely you will need to use auxiliary variables 
		...
		G_eq = ...
		
		...
		E_eq = ...

		if self.printLevel >= 2:
			print("G_eq: ")
			print(G_eq)
			print("E_eq: ")
			print(E_eq)

		self.G_eq = sparse.csc_matrix(G_eq)
		self.E_eq = E_eq

	def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
		""" 
		Solve a Quadratic Program defined as:
		minimize
			(1/2) * x.T * P * x + q.T * x
		subject to
			G * x <= h
			A * x == b
		using OSQP <https://github.com/oxfordcontrol/osqp>.
		"""  
		
		qp_A = vstack([G, A]).tocsc()
		l = -inf * ones(len(h))
		qp_l = hstack([l, b])
		qp_u = hstack([h, b])

		self.osqp = OSQP()
		self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)

		if initvals is not None:
			self.osqp.warm_start(x=initvals)
		res = self.osqp.solve()
		if res.info.status_val == 1:
			self.feasible = 1
		else:
			self.feasible = 0
			print("The FTOCP is not feasible at time t = ", self.time)

		self.Solution = res.x

