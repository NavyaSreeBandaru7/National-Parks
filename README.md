National Park Sentiment Analysis Dashboard

Overview

This Streamlit application analyzes visitor sentiments across U.S. National Parks to identify patterns in reviews, highlighting what visitors love and areas for improvement. The dashboard provides interactive visualizations and insights based on sentiment analysis of park reviews.
Features

Scrape and analyze reviews from popular national park websites
Sentiment analysis by park and feature (hiking, camping, scenery, etc.)
Interactive visualizations with filtering capabilities
Review examples with sentiment highlighting
Recommendations for park managers based on sentiment patterns

Installation
Requirements

Python 3.8+
pip package manager

Step 1: Clone or download this repository
bashgit clone https://your-repository-url.git
cd national-park-sentiment-analysis
Step 2: Create a virtual environment (recommended)
bash# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
Step 3: Install dependencies
bashpip install -r requirements.txt
This will install all the required packages listed in the requirements.txt file.
Usage
Running the application
To start the Streamlit application, run:
bashstreamlit run app.py
The application will open in your default web browser, typically at http://localhost:8501.
Using the Dashboard

Analysis Method: Choose between loading demo data or running a new analysis.

Demo data loads instantly with synthetic data for quick exploration
New analysis scrapes and analyzes real reviews (takes longer)


Dashboard Tabs:

Overview: High-level sentiment patterns and key insights
Park Analysis: Detailed sentiment analysis for each park
Feature Analysis: Sentiment breakdown by park features
Review Examples: Sample reviews with sentiment highlighting


Filtering: Use dropdown menus to filter by specific parks or features.

Customization
Adding More Parks
To add more national parks to the analysis, edit the park_urls dictionary in the app.py file:
pythonpark_urls = {
    "Park Name": "URL to park reviews page",
    # Add more parks here
}
Extending Feature Keywords
To analyze additional park features, expand the feature_keywords dictionary:
pythonfeature_keywords = {
    "feature_name": ["keyword1", "keyword2", "keyword3"],
    # Add more features and keywords
}
Troubleshooting
Common Issues

Installation Problems: If you have issues installing dependencies, try:
bashpip install --upgrade pip
pip install -r requirements.txt

NLTK Resource Issues: If you encounter NLTK-related errors, manually download stopwords:
pythonimport nltk
nltk.download('stopwords')

Scraping Errors: Some websites may block scrapers. Consider:

Adding delays between requests (already implemented)
Rotating user agents
Using a proxy service



Credits
This dashboard was developed based on sentiment analysis techniques and natural language processing using the following technologies:

Streamlit for the web interface
Transformers library for sentiment analysis
Plotly and Matplotlib for data visualization
BeautifulSoup for web scraping

License
MIT License
