import numpy as np
from utils import dlqr, system
import pdb
import matplotlib.pyplot as plt
from matplotlib import rc
import pypoman
from polytope import polytope
from ftocp import FTOCP
# =============================
# Initialize system parameters
A = np.array([[1.2, 1],
	          [0,   1]]);
B = np.array([[0], 
			  [1]]);
n = 2; d = 1;
x0      = np.array([13, -5.5])   # initial condition
# x0      = np.array([2, -1])   # initial condition
sys     = system(A, B, x0)
maxTime = 25
N       = 3
Q       = np.eye(n)
R       = np.eye(d)

# State constraint set X = \{ x : F_x x \leq b_x \}
Fx = np.vstack((np.eye(n), -np.eye(n)))
bx = np.array([15,15]*(2))

# Input constraint set U = \{ u : F_u u \leq b_u \}
Fu = np.vstack((np.eye(d), -np.eye(d)))
bu = np.array([1]*2)


# # =======================================================================================
# # ============== Approach 1 =============================================================
# # =======================================================================================
# Hint: the terminal set is X_f =\{x | F_f x <= b_f\}
"""
x is dim: 2 x 1

F_f is dim: 2 x 2
	just the identity, cause we are directly limiting what x is

b_f is dim: 2 x 1
	is just the terminal set
"""
# Ff = np.eye(n)
# bf = np.zeros(n)
## NEED TO BOUND FROM ABOVE AND BELOW!
Ff = np.vstack((np.eye(n), -np.eye(n)))
bf = np.array([0, 0]*(2))
# Doesn't matter
# Qf = np.eye(n)
Qf = np.zeros((n, n))

printLevel = 1
mpcApproach1 = FTOCP(N, A, B, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, printLevel)

# Run a closed-loop simulation
sys.reset_IC() # Reset initial conditions
xPredApp1 = []
for t in range(0,maxTime): # Time loop
	xt = sys.x[-1]
	ut = mpcApproach1.solve(xt)
	if mpcApproach1.feasible == 0:
		print("============ The MPC problem is not feasible")
		break
	xPredApp1.append(mpcApproach1.xPred)
	sys.applyInput(ut)

x_cl_1 = np.array(sys.x)

# Plot the results if the MPC problem was feasible
if mpcApproach1.feasible == 1:
	plt.figure()
	plt.plot(x_cl_1[:,0], x_cl_1[:,1], '-ob', label = "Closed Loop")
	for i in range(0, maxTime):
		if i == 0:
			plt.plot(xPredApp1[i][:,0], xPredApp1[i][:,1], '--.r', label="Predicted Trajectoires")	
		else:
			plt.plot(xPredApp1[i][:,0], xPredApp1[i][:,1], '--.r')	

	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	plt.legend()
plt.show()

# # =======================================================================================
# # ============== Approach 2 =============================================================
# # =======================================================================================
# Hint: the dlqr function return: i) P which is the solution to the DARE, ii) the optimal feedback gain K and iii) the closed-loop system matrix Acl = (A-BK)
P, K, Acl = dlqr(A, B, Q, R)
Ftot = np.vstack((Fx, np.dot(Fu, -K)))
btot = np.hstack((bx, bu ))
Qf = P
printLevel = 0

poli = polytope(Ftot, btot)
F, b = poli.computeO_inf(Acl) # Hint: this function returns F and b so that compute O_inf = \{ x | Fx <= b\}

# Hint: the terminal set is X_f =\{x | F_f x <= b_f\}
Ff = F
bf = b

terminalSetApproach2 = polytope(Ff, bf)
mpcApproach2 = FTOCP(N, A, B, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, printLevel)

# Simulate the closed-loop system
sys.reset_IC() # Reset initial conditions
xPredApp2 = []
for t in range(0,maxTime): # Time loop
	xt = sys.x[-1]
	ut = mpcApproach2.solve(xt)
	if mpcApproach2.feasible == 0:
		print("============ The MPC problem is not feasible")
		break

	xPredApp2.append(mpcApproach2.xPred)
	sys.applyInput(ut)

x_cl_2 = np.array(sys.x)

# Plot the results if the MPC problem was feasible

if mpcApproach2.feasible == 1:
	plt.figure()
	plt.plot(x_cl_2[:,0], x_cl_2[:,1], '-ob', label = "Closed Loop")
	for i in range(0, maxTime):
		if i == 0:
			plt.plot(xPredApp2[i][:,0], xPredApp2[i][:,1], '--.r', label="Predicted Trajectoires")	
		else:
			plt.plot(xPredApp2[i][:,0], xPredApp2[i][:,1], '--.r')	

	terminalSetApproach2.plot2DPolytope('k','Terminal Set')
	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	plt.legend()

plt.show()

# # =======================================================================================
# # ============== Compute the Region of Attraction =======================================
# # =======================================================================================

# First we over-approximate the terminal set used for approach 1 (We do so to have a set which is full dimension)
Ff = np.vstack((np.eye(n), -np.eye(n)))
bf = np.hstack((np.ones(n), np.ones(n)))*0.01
terminalSetApproach1 = polytope(Ff, bf)

# Compute the N-Step Controllable sets for approach 1 and approach 2
NStepControllable = []
for terminalSet in [terminalSetApproach1, terminalSetApproach2]:
	F, b = terminalSet.NStepPreAB(A, B, Fu, bu, N)
	NStepControllable.append(polytope(F, b))

# Plot the results
plt.figure()
terminalSetApproach2.plot2DPolytope('r', '$\mathcal{O}_\infty$')
NStepControllable[0].plot2DPolytope('b', '$\mathcal{K}_3(\{0\})$')
NStepControllable[1].plot2DPolytope('k', '$\mathcal{K}_3(\mathcal{O}_\infty)$')
plt.plot([2, -1], 'or', label='Initial condition part 1')
plt.plot([13, -5.5], 'sb', label='Initial condition part 2')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

plt.show()
