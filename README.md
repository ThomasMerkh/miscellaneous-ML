# miscellaneous-ML
Contains scripts written for embodied reinforcement learning, and soon others.

Files:
hexapod.py : This is the controller file for an embodied reinforcement learning task utilizing the YARS "Yet Another Robot Simulator" physics simulator.  The policy model being used is a conditional restricted Boltzmann machine, which learns a stochastic policy based on the GPOMDP algorithm.  Note that one must have YARS installed in order to use the hexapod.py controller file. See "Zahedi, Keyan, Arndt von Twickel, and Frank Pasemann. 'Yars: A physical 3d simulator for evolving controllers for real robots.' International Conference on Simulation, Modeling, and Programming for Autonomous Robots. Springer, Berlin, Heidelberg, 2008." for more information on YARS.
