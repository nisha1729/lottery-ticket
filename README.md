# lottery-ticket
Experiments on the Lottery Ticket Hypothesis. 
Base: Implement Frankle and Carbin's Lottery Ticket hypothesis for CNNs with CIFAR10 dataset.
Experiment1: Train the network on half of the dataset and find a winning ticket and then re-train on the other half and check the test accuracy. (generalisation)
Experiment2: Repeat Experiment1 with 10% of the dataset. This gives a winning ticket much faster with the same accuracy.
