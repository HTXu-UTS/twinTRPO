# twinTRPO
twinTRPO-Continuous: A self-contained implementation of the algorithm proposed in the following papers

[1] Haotian Xu, Junyu Xuan, Guangquan Zhang, Jie Lu. Twin Trust Region Policy Optimization.
submitted to IEEE Transactions on Transactions on Systems, Man and Cybernetics: Systems, 2024.

[2] Haotian Xu, Junyu Xuan, Guangquan Zhang, Jie Lu. Reciprocal trust region policy optimization.
In: World Scientific Proceedings Series on Computer Engineering and Information Science --
Intelligent Management of Data and Information in Decision Making, pp. 187-194,2024.

This twinTRPO aggregates TRPO and rTRPO, which has an upper bound and a lower bound of step size,
    and a least objective increments.

## Contribution

- A novel trust region policy optimization: rTRPO, based on the reciprocal optimization technique, which minimizes the KL divergence, subject to a least surrogate objective constraint, via exchanging the surrogate objective and the KL divergence constraint in the primary TRPO. This rTRPO induces a lower bound for the linear search of the step size and achieves a least objective increment.

- Aggregate TRPO and rTRPO to construct a novel TRPO variant: twinTRPO, which has both an upper bound and a lower bound for the purely linear search of the step size. This effectively limits the range of step size to facilitate the policy optimization, and obtains a least objective improvement simultaneously.

## Setup
Ensure that you have Python 3.8 and PyTorch 2.4

## Dataset
You can find the datasets [here](https://pypi.org/project/gym/)

## Usage

Run "twinTRPO_Continuous.py"

Please consider citing if you find this helpful or use this code for your research.
#==========================================================================

