# q_learner_test
  -  Q_nn class is a single layer neural net: Input(4) -> Hidden(32) -> ReLU -> Output(2)
  
      - graph structure, predict, and update step are held in separate methods
  
  - Eventually we will have a Q_learner base class and our different architectures will extend it by overriding the build_graph method
  
      - loss and optimizer should be a separate method as well
  
  - NetworkCopier keeps track of the graphs in the estimator and target networks and has a function to copy the estimator to the target
  
  - ReplayBuffer is a rolling experience replay buffer (without priority for now)
  
  We run for 1000 episodes, each time executing a policy on the environment and storing the outputs in the replay buffer.
  Then we do 10 iterations of sgd on random batches of up to 100 from the buffer.
  The first 250 episodes are purely random, then we start annealing epsilon from 1 down to 0.1
  Last cell tests random agent and learned agent, looking at best of 100.
  
  Outputs are from running on my local machine (Surface Book, i5 + GeForce)

# Current issues

  - best action is argmin, not argmax (I'm not sure if this is actually the case right now)
  
  - sometimes the loss explodes into 10e5-10e6 range, easier to see when testing on a hpc node (xeon + tesla p100) after 5000+ episodes 
  
  - still need to convert into a script
