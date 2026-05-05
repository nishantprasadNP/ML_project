# Machine Learning GUI Starter Project

This is a clean, beginner-friendly Python starter project that combines:

- **Tkinter** for GUI
- **scikit-learn** for machine learning
- **matplotlib** for visualization

## Project Files

- `main.py`  
  Builds the GUI, handles user input, trains the selected model, and shows the chart.

- `models.py`  
  Contains machine learning logic such as model selection, training, and evaluation.

- `utils.py`  
  Contains helper functions for loading datasets and preprocessing data.

- `README.md`  
  Project documentation and usage instructions.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn matplotlib pandas
```

> `tkinter` is usually included with Python on macOS and many desktop Python installs.

## Run

```bash
python3 main.py
```

## What this starter app does

1. Lets you choose a built-in dataset (`Iris` or `Breast Cancer`)
2. Lets you choose a model (`Logistic Regression`, `KNN`, `Decision Tree`, `SVM`)
3. Trains the model on button click
4. Displays model accuracy
5. Displays a confusion matrix chart in the GUI

## Next steps you can add

- CSV upload in the GUI
- More model hyperparameter controls
- More charts (accuracy comparison, feature importance, etc.)
- Support for regression models
