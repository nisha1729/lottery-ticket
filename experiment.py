import torch
import dataset
import network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prune_percent = 20
prune_iter = 15


def base_experiment():
    percent_weights_remaining = []
    test_accuracies = []
    iter_history = []

    model = network.FCnet(mask=None).to(device)
    model.apply(network.weights_init)
    torch.save(model.state_dict(), 'model_initial.ckpt')

    train_loader, val_loader, test_loader = dataset.data_base_mnist()

    iter_history.append(network.train(model, train_loader, val_loader))

    test_accuracies.append(network.test(model, test_loader))
    initial_mask = network.get_initial_mask(model)
    percent_weights_remaining.append(network.get_weights_remaining(initial_mask))

    new_mask = network.prune(prune_percent, model, initial_mask)

    for i in range(1, prune_iter):
        try:
            initial_model = torch.load("model_initial.ckpt")
            model.load_state_dict(initial_model)
            model.mask = new_mask
            iter_history.append(network.train(model, train_loader, val_loader))
            test_accuracies.append(network.test(model, test_loader))
            percent_weights_remaining.append(network.get_weights_remaining(new_mask))
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


if __name__ == "__main__":
    base_experiment()