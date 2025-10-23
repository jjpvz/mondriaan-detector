import numpy as np
import os
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from scipy.stats import entropy


class Timer:
    """
    Een klasse om de tijd bij te houden over een deel van de code.
    
    Gebruik:
        timer = Timer()
        timer.start()
        # ... code hier ...
        timer.stop()
        timer.print()
        
    Of met milliseconden:
        timer = Timer(unit='ms')
        timer.start()
        # ... code hier ...
        timer.stop()
    """
    
    def __init__(self, unit='s'):
        """
        Initialiseer de timer.
        
        Args:
            unit (str): 's' voor seconden, 'ms' voor milliseconden
        """
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.unit = unit.lower()
        if self.unit not in ['s', 'ms']:
            print(f"Waarschuwing: onbekende unit '{unit}', gebruik 's' of 'ms'. Standaard 's' wordt gebruikt.")
            self.unit = 's'
    
    def start(self):
        """Start de timer."""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = None
        print("Timer gestart.")
    
    def stop(self):
        """Stop de timer."""
        if self.start_time is None:
            print("Fout: Timer is nog niet gestart!")
            return
        
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        
        if self.unit == 'ms':
            print(f"Timer gestopt. Verstreken tijd: {self.elapsed_time * 1000:.2f} milliseconden")
        else:
            print(f"Timer gestopt. Verstreken tijd: {self.elapsed_time:.4f} seconden")
    
    def print(self):
        """Print de verstreken tijd."""
        if self.elapsed_time is None:
            if self.start_time is None:
                print("Timer is nog niet gestart.")
            else:
                print("Timer is nog niet gestopt.")
            return
        
        if self.unit == 'ms':
            elapsed_ms = self.elapsed_time * 1000
            if elapsed_ms >= 1000:
                seconds = elapsed_ms / 1000
                print(f"Verstreken tijd: {elapsed_ms:.2f} milliseconden ({seconds:.4f} seconden)")
            else:
                print(f"Verstreken tijd: {elapsed_ms:.2f} milliseconden")
        else:
            minutes = int(self.elapsed_time // 60)
            seconds = self.elapsed_time % 60
            
            if minutes > 0:
                print(f"Verstreken tijd: {minutes} minuten en {seconds:.2f} seconden ({self.elapsed_time:.4f} seconden totaal)")
            else:
                print(f"Verstreken tijd: {self.elapsed_time:.4f} seconden")
    
    def reset(self):
        """Reset de timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        print("Timer gereset.")
    
    def get_elapsed_time(self):
        """
        Geef de verstreken tijd terug.
        
        Returns:
            float: Tijd in seconden als unit='s', tijd in milliseconden als unit='ms'
        """
        if self.elapsed_time is None:
            return None
        
        if self.unit == 'ms':
            return self.elapsed_time * 1000
        else:
            return self.elapsed_time


def scatter_features_raw(df, x_feat, y_feat):
    plt.figure(figsize=(8,6))

    for label in df["label"].unique():
        mask = df["label"] == label
        df_label = df.loc[mask, [x_feat, y_feat]]
        
        plt.scatter(
            df_label[x_feat],
            df_label[y_feat],
            alpha=0.6,
            label=label
        )

    plt.xlabel(x_feat)
    plt.ylabel(y_feat)
    plt.title(f"Scatterplot: {x_feat} vs {y_feat} (individual points)")
    plt.legend()
    plt.show()

def plot_learning_curve(estimator, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.set_title("Learning Curves (SVM, RBF Kernel)")
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt

def save_plot(plt):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    filename = f"{script_name}_learning_curve.png"
    plt.savefig(filename)
    print(f"Learning curve plot saved as: {filename}")

