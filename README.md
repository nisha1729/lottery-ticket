# lottery-ticket
Experiments on the Lottery Ticket Hypothesis. 
Base: Implement Frankle and Carbin's Lottery Ticket hypothesis for CNNs with CIFAR10 dataset.
Experiment1: Train the network on half of the dataset and then prune and re-train on the other half. (generalisation)
Experiment2: Repeat Experiment1 with 5% of the dataset and then prune and re-train on the other half. Significant reduction in training time since only 5% of the dataset is involved.
