# partial-policy
Partial policy-based reinforcement learning (PPRL) for anatomical landmark localization

This is a simplified implementation example of the following work of ours:

W. Abdullah Al and I. D. Yun, "Partial Policy-Based Reinforcement Learning for Anatomical Landmark Localization in 3D Medical Images," in IEEE Transactions on Medical Imaging, vol. 39, no. 4, pp. 1245-1255, April 2020, doi: 10.1109/TMI.2019.2946345.

## Major benefits: 
Faster and better learning comparing with the other RL-based localization approaches


## Approach summary:
An agent initialized at a random point inside a 3D medical image, moves to a neighborhood point using 6 actions (left-right-up-down-sliceForeward-sliceBackward). By taking an episode of such moves, it attempts to converge to the target landmark.

State, S: ROI centered at the current point

Action, A: x+, x-, y+, y-, z+, z-

Reward, R (for taking action a) : +1, if the agent move closer to the target; or -1, if the agent moves farther

Policy, pi(s,a): for a given state s, pi gives the optimal action distribution for all the actions: {x+, x-, y+, y-, z+, z-}

In PPRL, we learn partial policies on the axial projections of the actual action space.
i.e., we have three partial policies: pi_x, pi_y, pi_z. 

For example, pi_x: S -> {x+, x-} ; only decides between the two actions along the X-axis. 

These three partial policies are periodically applied during exploration

## Implementation summary:
In this implementation example, an agent optimizes its partial policies to localize a target in a SINGLE volume, from a FIXED initial point.

Policy optimization technique: Proximal policy optimization (PPO)

Reward itself is used as the advantage. NO CRITIC is used.



