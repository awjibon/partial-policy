# Partial policy-based reinforcement learning (PPRL) for anatomical landmark localization

This is a simplified implementation example of the following work:

W. Abdullah Al and I. D. Yun, "Partial Policy-Based Reinforcement Learning for Anatomical Landmark Localization in 3D Medical Images," in IEEE Transactions on Medical Imaging, vol. 39, no. 4, pp. 1245-1255, April 2020, doi: 10.1109/TMI.2019.2946345.

## Major benefits: 
Faster and better learning comparing with the other RL-based localization approaches


## Approach summary:
An agent initialized at a random point inside a 3D medical image, moves to a neighborhood point using 6 actions (left-right-up-down-sliceForeward-sliceBackward). By taking an episode of such moves, it attempts to converge to the target landmark.

State, `S`: ROI centered at the current point

Action, `A`: `{x+, x-, y+, y-, z+, z-}`

Reward, `R` (for taking action `a`) : `+1`, if the agent moves closer to the target; or `-1`, if the agent moves farther

Policy, `pi(s,a)`: for a given state `s`, `pi` gives the optimal action distribution for all the actions: `{x+, x-, y+, y-, z+, z-}`

In PPRL, we learn partial policies on the axial projections of the actual action space.
i.e., we have three partial policies: `pi_x`, `pi_y`, `pi_z`. 

For example, `pi_x: S -> {x+, x-}` ; only decides between the two actions along X-axis. 

These three partial policies are periodically applied during exploration.

## Implementation summary:
In this implementation example, an agent optimizes its partial policies to localize a target in a **SINGLE** 3D volume.

Policy optimization technique: Proximal policy optimization (PPO)

Reward itself is used as the advantage. NO CRITIC is used.

And, a reward of convergence (`+2`) is added for the agent positions within a radius of 4 voxels

## Running the code
**Example**

`pprl_simple.py -mode "train" -volume_path "example.mat" -network_path "net/policy_best" -init_pos_center [200, 200, 200]`

**Parameters**
`-mode` : `"train"` / `"test"`

`-volume_path` : `"*/*.mat"`: this mat is formatted as follows: `{'vol': 3D array, 'gt':[[x,y,z]]}`. During test, a fake GT can be provided

`-network_path` : `"path/to/net"` : (default: `"net/policy_best"`)

`-init_pos_center`: Center of the sample space for the random initial position. (default: center of the input volume)

`-init_pos_radii`: Radii of the sample space for the random initial position. (default: `5`). Use `0`, for exactly using the center constantly, as the initial position.

`-init_pos_radii_multiplier`: To extend the radii but with stride. (default: `1` for zero stride)

`-max_episode`: number of episodes to explore at each epoch. (default: `5`)

`-max_step`: maximum number of steps per episode. (default: `30`). Keep in mind, the step-size is `2` in the current code.

`-max_epoch`: total number of epochs for training. (default: `300`)

`-epsilon`: initial epsilon value for training. (default: `0.7`). epsilon value is increased over the epochs at a rate of `1/max_epoch`. The higher the `epsilon` value, the greedier the policy.

`-alpha`: learning rate. (default: `1e-6`)

`-batch_size`: batch size for stochastic gradient descent. (default: `20.0`)

`-max_ppo_epoch`: the `K`-value in PPO. (default: `2`)

## Troubleshooting:
**If the reward does not improve**

- Try a smaller space for the initial position first (by reducing `init_pos_radii`), if possible.
- Increase `max_episode`, so that agent get more experienes to effectively learn from.
- Increase `max_epoch`.
- Lower the input volume resolution. ROI size for the state is `[32, 32, 32]`, which may not be large enough for the agent to decide the actions. Lowering the resolution while keeping the state-size same may help in this case. Gradually, train higher resolution agents like in different multi-scale approaches.







