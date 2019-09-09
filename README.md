# reinforcementlearning-Udemy
This repository is dedicated to my re-implementations of the programs in the Reinforcement Learning Course on Udemy. The read
will be updated as I cover more topics.

__Explore Exploit Problem__
Too much exploration and less exploitation will waste resources and increase the percentage of time exploring sub-optimal options
Too less exploration and more exploitation will result in not exploring optimal options

To solve the Explore Exploit problem we will be looking at 3 methods

__First Method: Greedy Epsilon Method__</br></br>
Start off with predicted mean as 0 and choose a very low initial epsilon value such that
```
P = random.randn()
If p is < eps
 Explore
Else
 Exploit
```
It ensures that each bandit is explored an infinite number of times in the long run
Eventually we will discover the true best since this allows us to update every bandit’s estimate
The problem with this strategy is that we are exploring even when we don’t need to . 
For example, for eps = 10% we are exploring sub-optimal options for 10% of our time. </br></br>
The first program **"comapre_epsilon.py"** will look at the results for the different epsilon values 0.1, 0.05, 0.01
The results of the experiment are

![](images/Screen%20Shot%202019-07-22%20at%201.04.28%20PM.png)

__Second Method (Explore Exploit Problem):Optimal Initial Values__ </br></br>
Here, we choose estimated mean for the bandits such that Estimated Mean >> True Mean.In contrast we choose the initial estimated mean to be 0 in the greedy epsilon method
This helps exploring more bandits as the collection of more data will force the estimated mean to only go down (since it is very high in comparison to true mean) while converging to the true mean.
</br></br> The second program **"Optimistic.py"** explores the Optimal Initial Values method. The results of the experiment are

![](images/Screen%20Shot%202019-07-22%20at%201.05.03%20PM.png)
