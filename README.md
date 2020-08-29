# Experiments on the Lottery Ticket Hypothesis. 
- Base: Implement Frankle and Carbin's Lottery Ticket hypothesis for CNNs with CIFAR10 dataset (**conv2_base.py**)
- Experiment 1: Train the network on half of the dataset and find a winning ticket and then re-train on the other half and check the test accuracy. (**conv2_split_iter.py**). Repeat for 4,10,40,80% splits.

To do: make the code more organised
