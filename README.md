# Experiments on the Lottery Ticket Hypothesis. 
We first implement from scratch the Frankle and Carbin's Lottery Ticket hypothesis for CNNs with CIFAR10 dataset. We then make a few interesting observations about lottery tickets.

**Lottery ticket hypothesis**: _A randomly-initialized, dense neural network contains a sub-network that is initialized such that when trained in isolation it can match the test accuracy of the original network after training for at most the same number of iterations._
These sub-networks are also referred to as _winning tickets_.
- **Trainability on unseen data:** Train the network on half of the dataset and find a winning ticket and then re-train on the other half and check the test accuracy. Repeat for 4,10,40,80% splits. We observe that winning tickets identified on the first half exhibit lottery ticket pattern on the second unseen half.
  
  <img src="/assets/unseen_data.png?raw=true" height=350>

- **Faster winning tickets:** Using subsets of training data will provide fast and efficient ways to identify winning tickets which continue to exhibit the lottery ticket behavior that matches the behavior for a winning ticket discovered using the full dataset. Furthermore, a winning ticket found from a subset of dataset will converge faster than that found from the entire dataset.

  <img src="/assets/faster_ticket.png?raw=true" height=350><img src="/assets/subset_expt.png?raw=true" height=350>

The full report is [here](https://github.com/nisha1729/lottery-ticket/files/6454066/lottery_ticket_experiments_report.pdf).
