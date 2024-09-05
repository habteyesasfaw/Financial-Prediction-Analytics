# Financial Data Analysis

## Overview
This repository contains the code and analysis for financial data using historical stock prices and financial news. The analysis focuses on performing exploratory data analysis (EDA), sentiment analysis, and technical indicator calculations to derive actionable investment strategies.

## Tasks

### Task 1: Exploratory Data Analysis (EDA)

#### Objective
Perform comprehensive Exploratory Data Analysis (EDA) on historical stock data to uncover insights into stock price trends, news frequency, and publisher activity.

#### Steps
1. **Setup Repository**
   - Create a GitHub repository to host all analysis code.
   - Create a new branch named `task-1`.

2. **Data Loading**
   - Load historical stock data CSV files (e.g., `NVDA_historical_data.csv`, `AAPL_historical_data.csv`, etc.).
   
3. **Exploratory Data Analysis (EDA)**
   - **Descriptive Statistics:**
     - Calculate basic statistics for textual lengths.
     - Count the number of articles per publisher.
     - Analyze publication dates for trends.
   - **Text Analysis:**
     - Perform sentiment analysis on headlines.
     - Extract topics and significant events using NLP techniques.
   - **Time Series Analysis:**
     - Examine publication frequency and time-based patterns.
   - **Publisher Analysis:**
     - Identify the most active publishers.
     - Analyze the type of news reported by different publishers.

4. **Commit Work**
   - Commit changes at least three times a day with descriptive commit messages.

5. **Documentation**
   - Document findings and insights from the EDA.

### Task 2: Technical Indicators and Quantitative Analysis

#### Objective
Perform quantitative analysis on historical stock price data using technical indicators and visualize the impact of these indicators on stock performance.

#### Steps
1. **Setup**
   - Ensure you have the necessary packages: `pandas`, `matplotlib`, `seaborn`, and `pandas_ta`.
   - Create a new branch named `task-2`.

2. **Data Preparation**
   - Load and prepare stock price data into a pandas DataFrame.
   - Columns should include Open, High, Low, Close, and Volume.

3. **Calculate Technical Indicators**
   - Use `pandas_ta` to calculate the following indicators:
     - Simple Moving Average (SMA)
     - Relative Strength Index (RSI)
     - Moving Average Convergence Divergence (MACD)

4. **Visualization**
   - Create visualizations to display the calculated indicators:
     - Plot SMA, RSI, and MACD for each stock.
     - Include charts showing the correlation between indicators and stock prices.

5. **Commit Work**
   - Merge necessary branches into the main branch using a Pull Request (PR).
   - Create a new branch named `task-3` for ongoing dashboard development.
   - Commit changes with descriptive messages.

6. **Documentation**
   - Document the process and findings, including visualizations and interpretations of technical indicators.

## Installation

To set up your environment, install the required packages:

```bash
pip install pandas matplotlib seaborn pandas_ta
```

## Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the repository directory:
   ```bash
   cd <repository-directory>
   ```

3. Open the Jupyter Notebook or Python script to start analysis.

## Contribution

1. **Branching:**
   - Create feature branches for new tasks.
   - Use descriptive names for branches (e.g., `task-1`, `task-2`).

2. **Commit Messages:**
   - Write clear and descriptive commit messages.
   - Commit changes regularly.

3. **Pull Requests:**
   - Submit Pull Requests (PRs) for merging branches into the main branch.
   - Ensure PRs include a description of changes and updates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



