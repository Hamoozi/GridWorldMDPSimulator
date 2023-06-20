import numpy as np
import mdptoolbox
import time

# define the rewards for each (state, action) pair


# define the transition probabilities for each (state, action, next_state) triple
P = np.zeros((4, 20, 20))
R = np.zeros((20,4))



#UP


P[0, 0, 0] = 1
P[0, 1, 1] = 1
P[0, 2, 2] = 1
P[0, 3, 3] = 1
P[0, 4, 4] = 1
P[0, 5, 0] = 1
P[0, 6, 1] = 1 
P[0, 7, 2] = 1
P[0, 8, 3] = 1
P[0, 9, 9] = 1
P[0, 10, 5] = 1
P[0, 11, 6] = 1
P[0, 12, 7] = 1
P[0, 13, 8] = 1
P[0, 14, 9] = 1
P[0, 15, 10] = 1
P[0, 16, 11] = 1
P[0, 17, 12] = 1
P[0, 18, 13] = 1
P[0, 19, 14] = 1



#DOWN

P[1, 0, 5] = 1
P[1, 1, 6] = 1
P[1, 2, 7] = 1
P[1, 3, 8] = 1
P[1, 4, 4] = 1
P[1, 5, 10] = 1
P[1, 6, 11] = 1 
P[1, 7, 12] = 1
P[1, 8, 13] = 1
P[1, 9, 9] = 1
P[1, 10, 15] = 1
P[1, 11, 16] = 1
P[1, 12, 17] = 1
P[1, 13, 18] = 1
P[1, 14, 19] = 1
P[1, 15, 15] = 1
P[1, 16, 16] = 1
P[1, 17, 17] = 1
P[1, 18, 18] = 1
P[1, 19, 19] = 1

#LEFT
P[2, 0, 0] = 1
P[2, 1, 0] = 1
P[2, 2, 1] = 1
P[2, 3, 2] = 1
P[2, 4, 4] = 1
P[2, 5, 5] = 1
P[2, 6, 5] = 1 
P[2, 7, 6] = 1
P[2, 8, 7] = 1
P[2, 9, 9] = 1
P[2, 10, 10] = 1
P[2, 11, 10] = 1
P[2, 12, 11] = 1
P[2, 13, 12] = 1
P[2, 14, 13] = 1
P[2, 15, 15] = 1
P[2, 16, 15] = 1
P[2, 17, 16] = 1
P[2, 18, 17] = 1
P[2, 19, 18] = 1

#RIGHT
P[3, 0, 1] = 1
P[3, 1, 2] = 1
P[3, 2, 3] = 1
P[3, 3, 4] = 1
P[3, 4, 4] = 1
P[3, 5, 6] = 1
P[3, 6, 7] = 1 
P[3, 7, 8] = 1
P[3, 8, 9] = 1
P[3, 9, 9] = 1
P[3, 10, 11] = 1
P[3, 11, 12] = 1
P[3, 12, 13] = 1
P[3, 13, 14] = 1
P[3, 14, 14] = 1
P[3, 15, 16] = 1
P[3, 16, 17] = 1
P[3, 17, 18] = 1
P[3, 18, 19] = 1
P[3, 19, 19] = 1



R[4, :] = 1
R[9, :] = -1
R[0, :] = -.04
R[1, :] = -.04
R[2, :] = -.04
R[3, :] = -.04
R[5, :] = -.04
R[6, :] = -.04
R[7, :] = -.04
R[8, :] = -.04
R[10, :] = -.04
R[11, :] = -.04
R[12, :] = -.04
R[13, :] = -.04
R[14, :] = -.04
R[15, :] = -.04
R[16, :] = -.04
R[17, :] = -.04
R[18, :] = -.04
R[19, :] = -.04




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