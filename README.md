# Fake Social Media Account Detection

A machine learning system to detect fake social media accounts (Twitter/X & Instagram) using user behavior patterns and metadata.

## Features
- **Data Training**: Trains a Random Forest Classifier on custom datasets (`fake_dataset.xlsx`).
- **Real-time Analysis**: Fetches live profile data from Twitter/X and Instagram.
- **Interactive Dashboard**: Streamlit-based UI for easy interaction.
- **Manual Input**: Advanced mode to manually test feature combinations.

## Installation

1. **Clone the repository** (if not already done).

2. **Set up a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model
Before running the dashboard, you need to train the model using your dataset.
1. Ensure your dataset is named `fake_dataset.xlsx` and placed in the project root.
2. Run the training pipeline:
   ```bash
   python fake_account_pipeline.py
   ```
   This will save the trained model to `outputs/best_model.joblib`.

### 2. Run the Dashboard
Start the web interface:
```bash
streamlit run dashboard.py
```
Access the app at `http://localhost:8501`.

## Project Structure
- `fake_account_pipeline.py`: Script to preprocess data and train the model.
- `dashboard.py`: Main application interface.
- `twitter_fetcher.py`: Scraper for Twitter/X profiles.
- `instagram_fetcher.py`: Scraper for Instagram profiles using Instaloader.
- `fake_dataset.xlsx`: Your training dataset.
- `outputs/`: Directory where trained models are saved.
