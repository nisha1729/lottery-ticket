import matplotlib.pyplot as plt
from model_functions import *

if __name__ == "__main__":
    percent_weights_remaining = []
    test_accuracies = []
    iter_history = []

    model = ConvNet(input_size, num_classes, mask=None).to(device)
    model.apply(weights_init)
    torch.save(model.state_dict(), 'model_initial.ckpt')

    iter_history.append(train(model))

    test_accuracies.append(test(model))
    initial_mask = get_initial_mask(model)
    percent_weights_remaining.append(get_weights_remaining(initial_mask))

    new_mask = prune(prune_percent, model, initial_mask)

    for i in range(1, prune_iter):
        try:
            initial_model = torch.load("model_initial.ckpt")
            model.load_state_dict(initial_model)
            model.mask = new_mask
            iter_history.append(train(model))
            test_accuracies.append(test(model))
            percent_weights_remaining.append(get_weights_remaining(new_mask))
            new_mask = prune(prune_percent, model, new_mask)
        except IndexError:
            break

    print(test_accuracies)
    print(iter_history)

    f1 = plt.figure(1)
    plt.plot(percent_weights_remaining[:len(test_accuracies)], test_accuracies, 'o--')
    plt.xlabel("Percentage of Weights Remaining")
    plt.ylabel("Early Stopping Test Accuracy")
    plt.title('Base Experiment')
    plt.grid()
    f1.gca().invert_xaxis()
    plt.savefig("acc_base.png")

    plt.clf()

    f2 = plt.figure(2)
    plt.plot(percent_weights_remaining[:len(iter_history)], iter_history, 'o--')
    plt.xlabel("Percentage of Weights Remaining")
    plt.ylabel("Early Stopping Iteration")
    plt.title('Base Experiment')
    plt.grid()
    f2.gca().invert_xaxis()
    plt.savefig("iter_base.png")
