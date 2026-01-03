# Hi, I'm Horzie!

I'm an experimental neural network designed to predict horse race outcomes, or at least try.
What started as a small hobby project quickly turned into a fascinating study about whether machine learning models can make sense of inherently unpredictable or random events, such as horse races and gambling scenarios.

This repository documents my journey — from raw data to classification models, visualizations, and ongoing research.

## 🎯 What Do I Do?

I learn from historical horse racing data (Kincsem Park, 2003–2025) and try to predict whether a horse will finish in the top 5, and if so, at which position.

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

To increase the accuracy of my predictions, I will need a more robust dataset of the horses.
You will see later in this document, but it's in plan to give me as up to date data as possible, with on sight analyzations.

## 🧠 Model Architecture

My "brain" is a **sequence-based classification model built on an LSTM**.

The idea is to analyze each horse’s historical performance **race by race, as a sequence**, and predict whether that horse will finish in the **top 5** in its next race.

For every horse:

- Past races are fed sequentially into an LSTM
- The **last hidden state** represents the horse’s current form
- A fully connected layer produces **class logits** for finishing positions

Classes are defined as:

- `0` → not in top 5
- `1–5` → exact finishing position

```python
class CustomModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, device):
    super(CustomModel, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
```

To make sequence learning possible, the dataset is structured by **horse**, not by race.

- The dataset is a list of horses
- Each horse is represented by a sequence of its past races
- Each race is a feature vector describing race conditions and metadata

So the structure looks like this:

```
[ # horse 1
  [ race_1_feature1, race_1_feature2, ..., race_1_featureK, place ],
  [ race_2_feature1, race_2_feature2, ..., race_2_featureK, place ],
  ...
]
```

During training:

- **Inputs**: all race features except `place`
- **Target**: the finishing position (`place`) of the **last race**

However, horses have **different numbers of past races**, which means their
input sequences are not the same length.  
Neural networks work best with fixed-size tensors, so to handle this, I use
**sequence padding**:

- shorter race histories are padded with zeros,
- longer histories stay as they are,
- and I keep track of the **original sequence lengths**.

Using these lengths, the model applies `pack_padded_sequence`, which tells the LSTM
to ignore the padded timesteps and only process the real race data.
This allows the model to learn effectively even when horses have very different
racing histories.

## 📉 Training Performance

Here’s how the loss developed during my first major training run:

![First loss over epochs with regression](./docs/loss.png "First loss over epochs with regression")

When the above training was run, I was using regression instead of classification.

The model is trained using **CrossEntropyLoss**, appropriate for multi-class classification.
This loss measures how confident the model is about the correct finishing class.

Here’s how the loss evolved during training:

![Latest loss over epochs](./docs/loss_latest.png "Latest loss over epochs")

Key observations:

- Training and validation loss track each other closely → no obvious overfitting
- Final loss settles around `~1.0–1.2`

For a 6-class problem (0–5), random guessing would yield:
`Loss ≈ ln(6) ≈ 1.79`

So a validation loss near `1.1` indicates the model is **significantly better than random**, despite the inherent noise of horse racing.

This confirms that the model is learning **meaningful structure**, even in a highly unpredictable domain.

## 🚀 Plans for the Future

Horzie is growing!
Here’s what’s coming next:

- ❤️ CNN pipeline to incorporate image data (horse photos, posture, “mindset”)
- 📡 Fully connected backend + frontend to make Horzie accessible as an app
- 🧹 Feature optimization (reduce noise, engineer better features, handle missing data)
- 🧪 Publishing an article about predicting random events with ML

And of course, lots more experiments along the way.

## 🎬 Final Words

I hope I caught your attention!
Whether you're here for machine learning, horse racing, or just for fun — Horzie and I welcome you aboard.

See you at the next race! 🏇🔥
