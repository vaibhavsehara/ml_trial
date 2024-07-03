# IPL Score Prediction with Neural Networks

This project aims to predict the total score of a cricket team in an IPL match using neural networks. The project involves preprocessing the data, building and training a neural network model, and evaluating its performance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/ipl-score-prediction.git
    cd ipl-score-prediction
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the IPL data CSV file (`ipl_data.csv`) in the project directory.

## Usage

1. Run the script to preprocess the data, train the model, and make predictions:

    ```bash
    python ipl_score_prediction.py
    ```

2. The script will output the mean absolute error of the predictions and plot the model loss.

## Data Preprocessing

The preprocessing steps include:

- Dropping irrelevant columns.
- Encoding categorical variables using label encoding.
- Splitting the data into training and testing sets.
- Scaling the features using MinMaxScaler.

## Model Building

The neural network model is built using Keras and TensorFlow. The architecture of the model is as follows:

- Input layer
- Hidden layer with 512 units and ReLU activation
- Hidden layer with 216 units and ReLU activation
- Output layer with a single unit and linear activation

The model is compiled using the Adam optimizer and Huber loss for regression tasks.

## Evaluation

The model is trained for 50 epochs with a batch size of 64. The training and validation losses are plotted to evaluate the model's performance.

The mean absolute error (MAE) is calculated to measure the accuracy of the predictions.

## Results

The mean absolute error (MAE) of the model on the test data is displayed after training.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
