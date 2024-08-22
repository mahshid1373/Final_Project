# Twitter Sentiment Analyzer

This project is a Twitter Sentiment Analysis application built using Python. It leverages machine learning techniques, particularly XGBoost, to classify the sentiment of tweets as negative, neutral, or positive. The project also includes a web interface powered by Streamlit, making it easy to interact with the model and visualize the results.

## Project Overview

The primary goal of this project is to analyze the sentiment of tweets. The application allows users to input tweet data, which is then processed and classified into one of three sentiment categories: Negative, Neutral, or Positive. The results are displayed on the Streamlit app along with various visualizations to help understand the underlying patterns in the data.

## Features

- **Sentiment Classification:** Classifies tweets into Negative, Neutral, or Positive sentiment using an XGBoost model.
- **Data Visualization:** Visualizes the distribution of sentiments, word clouds, and other relevant statistics.
- **User Interface:** A Streamlit-based web interface that allows users to interact with the model and explore the data.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/twitter-sentiment-analyzer.git
    ```
2. Navigate to the project directory:
    ```bash
    cd twitter-sentiment-analyzer
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run streamlit.py
    ```

## Dataset

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/code/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model/notebook). It contains tweets labeled with their corresponding sentiments.

## Usage

1. **Data Loading:** Load your dataset in the `.csv` format.
2. **Data Preprocessing:** Clean and preprocess the data as needed.
3. **Model Training (if applicable):** Train the model on the dataset (already provided in this project).
4. **Sentiment Analysis:** Use the Streamlit app to classify new tweets and visualize the results.

## Project Structure

- `streamlit.py`: The main script for running the Streamlit app.
- `main.ipynb`: A Jupyter Notebook containing the exploratory data analysis, model training, and evaluation steps.
- `function.py`: Utility functions used in the notebook and Streamlit app.
- `streamlit_function.py`: Additional functions specifically for the Streamlit app.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
