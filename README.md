# Slogan Generator and Industry Classifier (LSTM Neural Networks)

A deep learning capstone project that trains two Long Short-Term Memory (LSTM) models on business slogans using TensorFlow/Keras:

1. **Slogan generator:** produces a slogan one word at a time, starting from an industry name.
2. **Industry classifier:** predicts which industry a slogan belongs to.

The notebook also chains the two models together: it generates a slogan for an industry, then asks the classifier to identify that industry.

## Project Structure

```
.
├── neural_network_task.ipynb   # Jupyter Notebook with the full project
├── slogan-valid.csv            # Dataset (5,346 rows x 12 columns)
└── README.md
```

## Dataset

`slogan-valid.csv` contains 5,346 company slogans with their industries and other metadata. Only two columns are used:

| Column | Description |
|--------|-------------|
| `output` | The slogan text |
| `industry` | The industry of the company (142 distinct industries) |

The other columns (such as `desc`, `company`, `url` and `type`) are not needed for this project. The most common industries are information technology and services (452 slogans), marketing and advertising (267), computer software (257), construction (195) and internet (187).

## What the Notebook Does

### 1. Preprocessing
- Loads the data, keeps `output` and `industry`, and drops rows with missing values
- Uses **spaCy** (`en_core_web_sm`) to lower-case each slogan, tokenise it and remove punctuation, creating `processed_slogan`
- Creates `modified_slogan` by putting the industry name in front of the slogan (for example `computer hardware taking care of small business technology`), so the generator learns industry-specific wording

### 2. Slogan generator
- Tokenises the text with Keras `Tokenizer`
- Builds progressively longer sequences from every slogan so the model learns to predict the next word
- Pads the sequences to a common length (15 tokens) and one-hot encodes the next word
- Model: `Embedding(100)` > `LSTM(150, return_sequences=True)` > `LSTM(100)` > `Dense(softmax)`
- Trains for 50 epochs with `categorical_crossentropy` and the Adam optimiser
- `generate_slogan()` picks the most likely next word repeatedly (up to 20 words)

### 3. Industry classifier
- Maps each industry to an index and one-hot encodes the labels
- Removes the 6 industries that have only one slogan, so the data can be split in a stratified way
- Splits the data 80% / 20% into training and test sets (`stratify`, `random_state=42`)
- Uses the full padded slogans as input
- Model: `Embedding(100)` > `LSTM(150, return_sequences=True)` > `LSTM(100)` > `Dense(142, softmax)`
- Trains for 50 epochs and evaluates on the test set
- `classify_slogan()` predicts the industry of any new slogan

### 4. Combining the two models
Generates a slogan for the `internet` industry and classifies it, then compares the result.

## Results

| Model | Result |
|-------|--------|
| Generator | Training accuracy of about 72% on next-word prediction after 50 epochs (loss 1.37) |
| Classifier | Test accuracy of **18.1%** (test loss 7.49) |

For context, the classifier has 142 possible industries and the most common one makes up only about 8% of the data, so 18% is better than guessing the majority class, but still low.

**Example run**

```
Generated Slogan: internet web design digital marketing agency in dubai mena usa uae capital market goods staffing agency maryland staffing an project automation
Predicted Industry: marketing and advertising
```

The generated text reads like a mix of real slogans rather than a coherent new one, and the classifier labels it as marketing and advertising instead of internet. This shows that fluent-looking text and correct industry relevance are two different things.

## Requirements

- Python 3.9 or higher (a version compatible with your TensorFlow and spaCy installs)
- Jupyter Notebook or JupyterLab (or Google Colab, which the notebook recommends)
- tensorflow (with Keras)
- spacy and its small English model `en_core_web_sm`
- pandas
- numpy
- scikit-learn

## Getting Started

```bash

# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install tensorflow spacy pandas numpy scikit-learn jupyter
python -m spacy download en_core_web_sm

# 3. Launch the notebook
jupyter notebook neural_network_task.ipynb
```

Keep `slogan-valid.csv` in the same folder as the notebook, since it is loaded with a relative path. Then run all cells in order (**Kernel > Restart & Run All**).

Training takes a while on a CPU (the generator took roughly 20 seconds per epoch in the original run), so Google Colab with a GPU can be faster. If you use Colab, mount your Google Drive and update the CSV path in the data-loading cell.

## Author

**Siboniso Zikhali**
