import matplotlib.pyplot as plt
import numpy as np
from neuralnet import args2data, NN, parser, random_init
from neuralnet_2 import NN2

if __name__ == "__main__":
    args = parser.parse_args()
    (X_train, y_train, X_test, y_test, out_train, out_test, _, _, _, init_flag, _) = args2data(args)


    if (init_flag == 1):
        lr = 0.001
        num_epoch = 500

        train_cross_ent = np.array([], dtype=float)
        val_cross_ent = np.array([], dtype=float)
        step_size = 1
        hidden_units = np.array([5, 20, 50, 100, 200 ])

        for num_units in hidden_units:
            
            nn = NN(
                input_size=X_train.shape[-1],
                hidden_size=num_units,
                output_size=len(y_train),
                weight_init_fn=random_init,
                learning_rate=lr
            )
            train_losses, val_losses = nn.train(X_train, y_train, X_test, y_test, num_epoch)
            train_cross_ent = np.append(train_cross_ent, train_losses[-1])
            val_cross_ent = np.append(val_cross_ent, val_losses[-1])

        fig, ax = plt.subplots()
        ax.plot(hidden_units, train_cross_ent, label='cross entropy (train)')
        ax.plot(hidden_units, val_cross_ent, label='cross entropy (val)')
        ax.set_xlabel('hidden units')  # Add an x-label to the Axes.
        ax.set_ylabel('cross entropy')  # Add a y-label to the Axes.
        ax.set_title("Cross entropy vs hidden units")  # Add a title to the Axes.
        ax.legend()  # Add a legend.
        plt.show()

    elif (init_flag == 2):
        for lr in [0.03, 0.003, 0.0003]:

            num_epoch = 100
            
            nn = NN(
                input_size=X_train.shape[-1],
                hidden_size=50,
                output_size=len(y_train),
                weight_init_fn=random_init,
                learning_rate=lr
            )
            train_losses, val_losses = nn.train(X_train, y_train, X_test, y_test, num_epoch)

            fig, ax = plt.subplots()
            ax.plot(range(0, num_epoch), train_losses, label='cross entropy (train)')
            ax.plot(range(0, num_epoch), val_losses, label='cross entropy (val)')
            ax.set_xlabel('epoch number')  # Add an x-label to the Axes.
            ax.set_ylabel('cross entropy')  # Add a y-label to the Axes.
            ax.set_title(f"Cross entropy vs epochs with {lr} learning rate")  # Add a title to the Axes.
            ax.legend()  # Add a legend.
            plt.show()
    else:
        num_epoch = 50
            
        nn = NN(
            input_size=X_train.shape[-1],
            hidden_size=50,
            output_size=len(y_train),
            weight_init_fn=random_init,
            learning_rate=0.003
        )
        train_losses, val_losses = nn.train(X_train, y_train, X_test, y_test, num_epoch)

        nn2 = NN2(
            input_size=X_train.shape[-1],
            hidden_size=50,
            output_size=len(y_train),
            weight_init_fn=random_init,
            learning_rate=0.003
        )
        train_losses2, val_losses2 = nn.train(X_train, y_train, X_test, y_test, num_epoch)

        fig, ax = plt.subplots()
        ax.plot(range(0, num_epoch), train_losses, label='cross entropy 1 hidden layers (train)')
        ax.plot(range(0, num_epoch), val_losses, label='cross entropy 1 hidden layers (val)')
        ax.plot(range(0, num_epoch), train_losses2, label='cross entropy 2 hidden layers (train)')
        ax.plot(range(0, num_epoch), val_losses2, label='cross entropy 2 hidden layers(val)')
        ax.set_xlabel('epoch number')  # Add an x-label to the Axes.
        ax.set_ylabel('cross entropy')  # Add a y-label to the Axes.
        ax.set_title("Cross entropy in 1 vs 2 hidden layers")  # Add a title to the Axes.
        ax.legend()  # Add a legend.
        plt.show()