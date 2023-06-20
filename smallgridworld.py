
import numpy as np
import mdptoolbox


# define the rewards for each (state, action) pair


# define the transition probabilities for each (state, action, next_state) triple
P = np.zeros((4, 9, 9))
R = np.zeros((9,4))
import time


#UP


P[0, 0, 0] = 1
P[0, 1, 1] = 1
P[0, 2, 2] = 1
P[0, 3, 0] = 1
P[0, 4, 1] = 1
P[0, 5, 5] = 1
P[0, 6, 3] = 1 
P[0, 7, 4] = 1
P[0, 8, 5] = 1



#DOWN

P[1, 0, 3] = 1
P[1, 1, 4] = 1
P[1, 2, 2] = 1
P[1, 3, 6] = 1
P[1, 4, 7] = 1
P[1, 5, 5] = 1
P[1, 6, 6] = 1 
P[1, 7, 7] = 1
P[1, 8, 8] = 1



#LEFT
P[2, 0, 0] = 1
P[2, 1, 0] = 1
P[2, 2, 2] = 1
P[2, 3, 3] = 1
P[2, 4, 3] = 1
P[2, 5, 5] = 1
P[2, 6, 6] = 1 
P[2, 7, 6] = 1
P[2, 8, 7] = 1

#RIGHT
P[3, 0, 1] = 1
P[3, 1, 2] = 1
P[3, 2, 2] = 1
P[3, 3, 4] = 1
P[3, 4, 5] = 1
P[3, 5, 5] = 1
P[3, 6, 7] = 1 
P[3, 7, 8] = 1
P[3, 8, 8] = 1



R[2, :] = 1
R[5, :] = -1
R[0, :] = -.04
R[1, :] = -.04
R[3, :] = -.04
R[4, :] = -.04
R[6, :] = -.04
R[7, :] = -.04
R[8, :] = -.04





mdp = mdptoolbox.mdp.PolicyIteration(P, R, discount=0.9)
mdpe = mdptoolbox.mdp.ValueIteration(P, R, discount=0.9)
start = time.time()
mdp.run()
end = time.time()
start1 = time.time()
mdpe.run()
end1 = time.time()

print("Optimal Policy for policy iteration " + str(mdp.policy))
print("NUmber of iterations to converge for policy iteration " + str(mdp.iter))

print("Optimal Policy for value iteration " + str(mdpe.policy))
print("NUmber of iterations to converge for value iteration " + str(mdpe.iter))

print("Time to converge for the policy iteration " + str(end - start))
print("Time to converge for the value iteration " + str(end1 - start1))