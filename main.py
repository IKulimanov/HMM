import tushare
import numpy as np
import matplotlib.pyplot as plt
from GaussianHMM import GaussianHMM

if __name__ == "__main__":
    data = tushare.get_hist_data("hs300")["close"]
    n_days = len(data)
    train_data, test_data = (
        data[: int(n_days * 0.8)].to_numpy(),
        data[int(n_days * 0.8) :].to_numpy(),
    )
    print("length of training data: ", len(train_data))
    print("length of test data: ", len(test_data))

    with open("dataset/array2.txt", 'r') as f:
        dataset = np.genfromtxt(f)

    rt = train_data.reshape(-1, 1)

    plt.figure()
    plt.plot(dataset, label="actual values")
    plt.show()

    num_states = [4, 8, 16, 32]
    for n in num_states:
        hmm = GaussianHMM(n_state=n, x_size=1, iter=50)
        hmm.train(dataset.reshape(-1, 1))
        predictions, _ = hmm.generate_seq(len(dataset))
        plt.figure()
        plt.plot(predictions, label="predictions")
        plt.plot(dataset, label="actual values")
        plt.legend()
        plt.title("{} hidden states".format(n))
        plt.xlabel("t")
        plt.ylabel("price")
        plt.savefig("{}_hidden_states_predictions".format(n))