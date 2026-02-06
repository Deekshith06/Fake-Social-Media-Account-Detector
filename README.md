# Fake Social Media Account Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-FF6F00?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen?style=flat-square)

</div>

A machine learning-powered application to detect fake social media accounts on Twitter/X and Instagram using behavioral patterns, engagement metrics, and profile metadata.

## 🎯 Overview

This system uses a Random Forest Classifier to analyze social media profiles and predict whether they are genuine or fake accounts. It features real-time profile fetching, an interactive web dashboard, and supports both automated analysis and manual feature testing.

## 🎬 Demo

### 🌐 Live Application
**Try it now:** [Fake Account Detector](https://fake-social-media-account-detector-hv3j4nrdtnwrgwhrnk7zyl.streamlit.app/)

> No installation needed! Access the live web application to analyze social media profiles instantly.

```
🔍 Analyzing Twitter Profile: @example_user
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Profile fetched successfully
✓ Features extracted
✓ Model prediction complete

📊 RESULT: Fake Account (Confidence: 87.3%)

Key Indicators:
  • Low follower/following ratio
  • Minimal post activity
  • Recent account creation
  • Suspicious engagement patterns
```

## ✨ Key Features

- **ML-Based Detection**: Random Forest Classifier trained on behavioral patterns
- **Multi-Platform Support**: Works with Twitter/X and Instagram profiles
- **Real-Time Analysis**: Fetches and analyzes live profile data
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Manual Testing Mode**: Test custom feature combinations
- **Model Persistence**: Save and reuse trained models

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Active internet connection (for profile fetching)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Deekshith06/fake-social-media-detection.git
cd fake-social-media-detection
```

### 2. Create Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
fake-social-media-detection/
│
├── fake_account_pipeline.py    # Model training pipeline
├── dashboard.py                 # Streamlit web application
├── twitter_fetcher.py          # Twitter/X profile scraper
├── instagram_fetcher.py        # Instagram profile scraper
├── fake_dataset.xlsx           # Training dataset
├── requirements.txt            # Python dependencies
│
└── outputs/
    └── best_model.joblib       # Trained model (generated)
```

## 🎓 Usage

### Step 1: Prepare Your Dataset

1. Place your training dataset as `fake_dataset.xlsx` in the project root directory
2. Ensure the dataset contains the following features:
   - Profile metadata (followers, following, posts count)
   - Engagement metrics (likes, comments, shares)
   - Account characteristics (verified status, bio length, etc.)
   - Label column indicating fake (1) or genuine (0) accounts

### Step 2: Train the Model

Run the training pipeline to build and save your model:

```bash
python fake_account_pipeline.py
```

**What this does:**
- Loads and preprocesses your dataset
- Trains a Random Forest Classifier
- Evaluates model performance
- Saves the trained model to `outputs/best_model.joblib`

### Step 3: Launch the Dashboard

Start the web interface:

```bash
streamlit run dashboard.py
```

The application will open in your browser at `http://localhost:8501`

### Step 4: Analyze Profiles

**Option A: Automated Profile Analysis**
1. Select platform (Twitter/X or Instagram)
2. Enter the username
3. Click "Analyze Profile"
4. View prediction results and confidence scores

**Option B: Manual Feature Testing**
1. Switch to "Advanced Mode"
2. Input custom feature values
3. Test different scenarios
4. Analyze model predictions

## 🔧 Configuration

### API Keys (if required)

For Twitter/X API access, create a `.env` file:
```env
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret
```

### Model Parameters

Modify hyperparameters in `fake_account_pipeline.py`:
```python
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)
```

## 📊 Dataset Format

Your `fake_dataset.xlsx` should have these columns (example):

| followers_count | following_count | posts_count | verified | fake_label |
|----------------|-----------------|-------------|----------|------------|
| 1250           | 345             | 89          | 0        | 0          |
| 45             | 5000            | 3           | 0        | 1          |

## 🛠️ Troubleshooting

**Model not found error:**
- Run `python fake_account_pipeline.py` first to train the model

**Profile fetching fails:**
- Check internet connection
- Verify the username is correct
- Ensure API credentials are valid (for Twitter/X)

**Import errors:**
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify Python version: `python --version`

## 📈 Performance Metrics

The model's performance can be viewed during training, including:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Always verify results manually and comply with platform terms of service when scraping data.

## 📧 Contact & Author

**Seelaboyina Deekshith**  
AI & ML Engineer | Building Intelligent Systems 🤖

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Deekshith06)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/deekshith03020)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/deekshith45823)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:seelaboyinadeekshith@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://deekshith06.github.io/Portfolio-website/)

For questions, suggestions, or collaboration opportunities, please:
- Open an issue on [GitHub](https://github.com/Deekshith06/fake-social-media-detection/issues)
- Reach out via email or social media
- Contribute by submitting a pull request

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by [Deekshith](https://github.com/Deekshith06)  
**Building safer social media with AI** 🚀

[![GitHub followers](https://img.shields.io/github/followers/Deekshith06?label=Follow&style=social)](https://github.com/Deekshith06)
[![GitHub stars](https://img.shields.io/github/stars/Deekshith06?style=social)](https://github.com/Deekshith06)

</div>

---

**Made with ❤️ for safer social media**
