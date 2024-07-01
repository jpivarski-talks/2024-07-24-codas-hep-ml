# Plan

## Introduction (1.5 hours)

Organized as a PDF talk, followed by a TensorFlow Playground exercise of at least 0.5 hours (with a clear objective, not just tinkering).

- Now we have two ways to make algorithms: craftsmanship (programming by hand) and farming (throw random numbers, optimize them for a desired loss, let the solution grow/evolve).
  - There's nothing magical or automatic about farming. Even though (metaphorical) plants solve more intricate problems than we could ever engineer, they have to be provided the right conditions to grow.
  - Even if you never studied machine learning, your physics background prepared you for the concepts.
- History: HEP has always needed ML, it's just becoming possible now.
  - Started by employing humans to find tracks, automated with heuristics and now ML as rates increased (plot).
  - GOFAI (Good Old Fashioned AI) versus connectivism.
  - Early neural networks, alternatives like decision trees, clustering, SVNs, etc., and the deep learning explosion.
  - Several winters and the current spring (plots from Google Trends, CHEP abstracts): vanishing gradients, web-scale datasets, and GPUs.

Now switch to a Jupyter notebook (after a few LaTeX slides taken from my SciPy talk).

- Deep learning is just an extension of curve fitting. (Like thermodynamics is just an ensemble of classical or quantum systems.)
  - Fit an arbitrary curve to Taylor or Fourier basis functions (example from [this blog](https://www.oranlooney.com/post/adaptive-basis-functions/)).
  - Fit it to adaptively parameterized sigmoids (maybe also ReLUs): it's a better fit.
  - This is a linear fit with a hidden layer and a demonstration of the Universal Approximation Theorem.
- XOR example (responsible for the first Winter) in Scikit-Learn and PyTorch. (Justify PyTorch with GitHub-physicists and Google Trends.)
- Outline of the rest of the sessions.

Half-hour exercise classifying data in TensorFlow Playground with as few parameters as possible.

## Issues in practice (2 hours)

Organized as a Jupyter notebook talk with small challenge problems "on rails" (hand-holding). Each of the issues listed below will be addressed in purely linear fits (i.e. one layer NNs) if possible because the focus is _not_ on the architecture. The datasets are either artificial or standard (e.g. MNIST), too.

- Regression, binary classification, multi-classification: what problem are you trying to solve?
  - Loss functions: OLS & LAD for regression, NLL (define softmax & one-hot), and cross-entropy. There are others.
- Optimizers, starting with analytic (for OLS) and Minuit, then ML standards like Adam.
  - Learning rate, batches, and epochs (which are all trivial for analytic fits, non-trivial for ML optimizers).
- Feature selection and the "kernel trick"; reminder that polynomials and sin/cos are Taylor and Fourier.
- Under & overfitting: polynomial that goes through all points.
  - "Effective" dimensionality due to correlations among features.
- Parameters vs hyperparameters (what is the optimizer choosing? what are you choosing?).
- Partitioning data into train-test (no hyperparameters) and train-test-validate (with hyperparameters).
  - What if a time-series is self-correlated?
- Goodness of fit: true/false positives/negatives, ROC, AUC.
- Regularization: L1, L2, dropout (yes, for a linear fit!).

## Survey of architectures (1.5 hours)

Organized as a Jupyter notebook talk with small challenge problems "on rails" (hand-holding). Some of the architectures will be demonstrated with code and exercises while others are merely described, especially if they're an extension of one that was demonstrated.

- What are all the things we can do with multidimensional arrays? (Present as Lego bricks.)
  - Linear transformation, NumPy ufunc (map), sum/mean/min/max (reduce), concatenate, sampling (VAE), ...
- Standard MLP.
- Autoencoder (demo) and variational autoencoder (only discuss).
- CNN (demo) and other static neighborhoods, such as hexagonal detector elements (only discuss).
- DeepSet, reducing over ragged arrays (demo) and GNNs (only discuss).

Maybe switch to a PDF talk, since the rest is "only discuss."

- GANs (only discuss).
- RNNs, attention, transformers, ChatGPT (only discuss).

End by describing the challenge exercise in the next section, after the coffee break.

## Challenge exercise (2 hours)

Last year's jet classification problem, using any method at all ("open world, no rails"). Students work in groups of 4‒5 and name their teams.

I'll provide functions that split the data into train-test-validate, compute a ROC curve on the validation set, and submit it to me with the team name, so that I can present a leaderboard. The team with the best AUC wins.
