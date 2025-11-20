# Hi, I'm Horzie!

I'm an experimental neural network designed to predict horse race outcomes, or at least try.
What started as a small hobby project quickly turned into a fascinating study about whether machine learning models can make sense of inherently unpredictable or random events, such as horse races and gambling scenarios.

This repository documents my journey — from raw data to classification models, visualizations, and ongoing research.

## 🎯 What Do I Do?

I learn from historical horse racing data (Kincsem Park, 2003–2025) and try to predict the finishing position of each horse.

Currently, I extract a large number of input features — probably too many, honestly — and feed them into a classification neural network that tries to learn patterns from:

- Horse metadata
- Race conditions
- Jockey, trainer, and stable information
- Weather and race characteristics
- Historical performance

The point is not just prediction accuracy — but exploring how ML behaves when faced with chaotic or noisy domains.

## 📊 Dataset & Inputs

I currently use data from Kincsem Park, spanning 2003–2025.
Here’s how the features correlate with each other:

![Correlation matrix](./docs/corr.png "Correlation matrix")

As you can see… not very helpful.
Lots of weak correlations → lots of noise → tough problem.

But that’s what makes this project fun!

## 🧠 Model Architecture

My current "brain" is a simple fully connected neural network:

```python
self.model = nn.Sequential(
  nn.Linear(in_features, 32),
  nn.ReLU(),
  nn.Linear(32, 16),
  nn.ReLU(),
  nn.Linear(16, output_size)
)
```

Yes, this probably needs improvement, but for now this is the baseline I’m experimenting with.

## 📉 Training Performance

Here’s how the loss developed during my first major training run:

![Loss over epochs](./docs/loss.png "Loss over epochs")

Training and validation loss steadily decreased, for a limited time...
Not perfect, but significantly better than random guessing
This means the model is learning, but the problem remains highly unpredictable — which is exactly what makes it interesting.

## 🚀 Plans for the Future

Horzie is growing!
Here’s what’s coming next:

- ❤️ CNN pipeline to incorporate image data (horse photos, posture, “mindset”)
- 📡 Fully connected backend + frontend to make Horzie accessible as an app
- 🧹 Feature optimization (reduce noise, engineer better features, handle missing data)
- 📊 Experimenting with ranking models instead of simple classification
- 🧪 Publishing an article about predicting random events with ML

And of course, lots more experiments along the way.

## 🎬 Final Words

I hope I caught your attention!
Whether you're here for machine learning, horse racing, or just for fun — Horzie and I welcome you aboard.

See you at the next race! 🏇🔥
