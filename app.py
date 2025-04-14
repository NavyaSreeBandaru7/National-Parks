#"""
National Park Sentiment Analysis Streamlit Application
-----------------------------------------------------
This application analyzes sentiment from national park reviews and presents the results
in an interactive dashboard using Streamlit.
"""#

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
import time
import random
import nltk
from nltk.corpus import stopwords
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="National Park Sentiment Analysis",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and intro
st.title("🏞️ National Park Sentiment Analysis Dashboard")
st.markdown("""
Analyze visitor sentiments across U.S. National Parks to identify patterns in reviews,
highlighting what visitors love and areas for improvement.
""")

# Sidebar
st.sidebar.title("Controls")

# Load or analyze option
analysis_option = st.sidebar.radio(
    "Choose Analysis Method:",
    ["Load Demo Data", "Run New Analysis"]
)

@st.cache_data
def load_demo_data():
    """Load pre-analyzed demo data"""
    # This would normally load from a saved CSV
    # For now, we'll generate synthetic data
    parks = [
        "Yellowstone", "Yosemite", "Grand Canyon", "Zion", 
        "Rocky Mountain", "Glacier", "Acadia", "Olympic", "Grand Teton"
    ]
    features = [
        "Hiking", "Camping", "Scenery", "Wildlife", 
        "Facilities", "Accessibility", "Fees", "Crowds"
    ]
    sentiments = ["POSITIVE", "NEGATIVE"]
    scores = np.random.uniform(0.5, 0.99, size=(len(parks)*len(features)))
    
    # Create dataframe
    data = []
    for park in parks:
        for feature in features:
            sentiment = np.random.choice(sentiments, p=[0.7, 0.3])  # 70% positive bias
            score = np.random.uniform(0.6, 0.95)
            example1 = f"Sample review about {feature.lower()} in {park}."
            example2 = f"Another opinion about {feature.lower()} experience at {park}."
            
            data.append({
                "Park": park,
                "Feature": feature,
                "Sentiment": sentiment,
                "Score": round(score, 3),
                "Example Sentence 1": example1,
                "Example Sentence 2": example2
            })
    
    return pd.DataFrame(data)

# Feature keywords for scraping
feature_keywords = {
    "hiking": ["hiking", "hike", "trail", "trek", "path", "walk", "climb", "summit"],
    "camping": ["camping", "campsite", "tent", "campground", "camp", "rv", "overnight"],
    "scenery": ["view", "scenic", "landscape", "beauty", "overlook", "vista", "panorama", "waterfall"],
    "wildlife": ["wildlife", "animal", "bird", "deer", "bear", "elk", "bison", "moose"],
    "facilities": ["restroom", "bathroom", "facility", "visitor center", "amenities", "lodge", "restaurant"],
    "accessibility": ["accessible", "wheelchair", "disability", "paved", "easy", "stroller", "family-friendly"],
    "fees": ["fee", "cost", "price", "pass", "expensive", "affordable", "ticket", "reservation"],
    "crowds": ["crowd", "busy", "people", "traffic", "quiet", "peaceful", "solitude", "line"]
}

# URLs for analysis
park_urls = {
    "Yellowstone": "https://www.tripadvisor.com/Attraction_Review-g143026-d108746-Reviews-Yellowstone_National_Park-Yellowstone_National_Park_Wyoming.html",
    "Yosemite": "https://www.tripadvisor.com/Attraction_Review-g60999-d108754-Reviews-Yosemite_National_Park-Yosemite_National_Park_California.html",
    "Grand Canyon": "https://www.tripadvisor.com/Attractions-g143028-Activities-Grand_Canyon_National_Park_Arizona.html",
    "Zion": "https://www.tripadvisor.com/Attractions-g143047-Activities-Zion_National_Park_Utah.html",
    "Rocky Mountain": "https://www.tripadvisor.com/Attraction_Review-g143048-d145151-Reviews-Rocky_Mountain_National_Park-Rocky_Mountain_National_Park_Colorado.html",
    "Glacier": "https://www.tripadvisor.com/Attraction_Review-g143026-d145256-Reviews-Glacier_National_Park-Glacier_National_Park_Montana.html",
    "Acadia": "https://www.tripadvisor.com/Attraction_Review-g143010-d123078-Reviews-Acadia_National_Park-Acadia_National_Park_Mount_Desert_Island_Maine.html",
    "Olympic": "https://www.tripadvisor.com/Attraction_Review-g143047-d145152-Reviews-Olympic_National_Park-Olympic_National_Park_Washington.html"
}

def scrape_article_text(url):
    """Scrape text content from a website."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        
        # Check if the request was successful
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract reviews and comments
            reviews = []
            
            # TripAdvisor reviews
            if "tripadvisor.com" in url:
                review_elements = soup.find_all(['div', 'p'], class_=lambda c: c and ('review' in str(c).lower() or 'comment' in str(c).lower()))
                reviews.extend([elem.get_text(strip=True) for elem in review_elements if elem.get_text(strip=True)])
                
                # TripAdvisor specific review elements
                review_texts = soup.find_all(class_=lambda c: c and ('partial_entry' in str(c) or 'reviewText' in str(c)))
                reviews.extend([elem.get_text(strip=True) for elem in review_texts if elem.get_text(strip=True)])
            
            # If still no reviews, try generic approach
            if not reviews:
                paragraphs = soup.find_all('p')
                reviews = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]
            
            # Combine all review text
            return " ".join(reviews)
        else:
            st.warning(f"Failed to retrieve from {url}: Status code {response.status_code}")
            return ""
    except Exception as e:
        st.warning(f"Error scraping {url}: {str(e)}")
        return ""

def extract_sentences_with_keywords(text, keywords):
    """Extract sentences that contain specific keywords."""
    if not text:
        return []
    
    # Split text into sentences more accurately
    sentences = re.split(r'(?<=[.!?])\s+', text)
    matched = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        for kw in keywords:
            if kw.lower() in sentence.lower():
                matched.append(sentence)
                break
    
    return matched

def run_sentiment_analysis():
    """Run the sentiment analysis on park reviews."""
    # Initialize sentiment analysis pipeline
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Set up progress tracking
    progress_bar = st.progress(0)
    total_parks = len(park_urls)
    
    results = []
    
    for i, (park, url) in enumerate(park_urls.items()):
        st.info(f"Analyzing reviews for {park}...")
        article_text = scrape_article_text(url)
        
        # Skip if no content found
        if not article_text:
            st.warning(f"No review content found for {park}")
            continue
            
        park_results = {"Park": park, "features": {}}
        
        # For each feature category, find relevant sentences and analyze sentiment
        for feature, keywords in feature_keywords.items():
            matched_sentences = extract_sentences_with_keywords(article_text, keywords)
            
            if matched_sentences:
                # Analyze sentiment for each matching sentence
                sentiments = []
                for j in range(0, len(matched_sentences), 3):  # Process in batches of 3
                    batch = matched_sentences[j:j+3]
                    joined_text = " ".join(batch)[:512]  # Truncate to fit model max length
                    
                    try:
                        sentiment = sentiment_model(joined_text)[0]
                        sentiments.append(sentiment)
                    except Exception as e:
                        st.warning(f"Error analyzing sentiment: {str(e)}")
                        continue
                
                # Only proceed if we have sentiment results
                if sentiments:
                    # Calculate average sentiment score
                    avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
                    # Determine overall sentiment (majority vote)
                    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0}
                    for s in sentiments:
                        sentiment_counts[s["label"]] += 1
                    
                    overall_sentiment = "POSITIVE" if sentiment_counts["POSITIVE"] >= sentiment_counts["NEGATIVE"] else "NEGATIVE"
                    
                    park_results["features"][feature] = {
                        "sentiment": overall_sentiment,
                        "score": avg_score,
                        "examples": matched_sentences[:2]  # Keep 2 example sentences
                    }
        
        results.append(park_results)
        progress_bar.progress((i + 1) / total_parks)
        time.sleep(random.uniform(0.5, 1.5))  # Respect rate limits
    
    # Flatten results for DataFrame
    flat_results = []
    
    for result in results:
        park = result["Park"]
        
        for feat, data in result["features"].items():
            flat_results.append({
                "Park": park,
                "Feature": feat.capitalize(),
                "Sentiment": data["sentiment"],
                "Score": round(data["score"], 3),
                "Example Sentence 1": data["examples"][0] if data["examples"] else "",
                "Example Sentence 2": data["examples"][1] if len(data["examples"]) > 1 else ""
            })
    
    return pd.DataFrame(flat_results)

# Load or analyze based on user selection
if analysis_option == "Load Demo Data":
    df = load_demo_data()
    st.success("Demo data loaded successfully!")
else:
    with st.spinner("Running sentiment analysis on national park reviews... This may take a few minutes."):
        df = run_sentiment_analysis()
    st.success("Analysis complete!")

# Main dashboard
st.header("Sentiment Analysis Results")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Park Analysis", "Feature Analysis", "Review Examples"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Overall Sentiment by Feature")
        
        # Create feature sentiment summary
        feature_sentiment = df.groupby('Feature')['Sentiment'].value_counts(normalize=True).unstack().fillna(0)
        feature_sentiment = feature_sentiment.sort_values(by='POSITIVE', ascending=False)
        
        # Create a bar chart with Plotly
        fig = px.bar(
            feature_sentiment.reset_index(), 
            x='Feature',
            y=['POSITIVE', 'NEGATIVE'],
            title="Sentiment Distribution by Feature",
            color_discrete_map={'POSITIVE': '#4CAF50', 'NEGATIVE': '#F44336'},
            labels={'value': 'Proportion', 'variable': 'Sentiment'},
            barmode='stack'
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Overall Sentiment Distribution")
        
        # Count sentiment distribution
        sentiment_counts = df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Create a pie chart with Plotly
        fig = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={'POSITIVE': '#4CAF50', 'NEGATIVE': '#F44336'},
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics
    st.subheader("Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_positive = df[df['Sentiment'] == 'POSITIVE'].groupby('Feature').size().sort_values(ascending=False).index[0]
        st.metric("Most Positive Feature", top_positive)
    
    with col2:
        top_negative = df[df['Sentiment'] == 'NEGATIVE'].groupby('Feature').size().sort_values(ascending=False).index[0]
        st.metric("Most Negative Feature", top_negative)
    
    with col3:
        top_park = df[df['Sentiment'] == 'POSITIVE'].groupby('Park').size().sort_values(ascending=False).index[0]
        st.metric("Highest Rated Park", top_park)
    
    # Recommendations
    st.subheader("Recommendations for Park Managers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Address Common Complaints
        - 🔍 Implement timed entry systems to reduce crowds at peak times
        - 🚽 Improve restroom and facility maintenance schedules
        - 💰 Consider more affordable fee options for families
        """)
    
    with col2:
        st.markdown("""
        #### Enhance Visitor Experience
        - 🦌 Create more guided wildlife viewing opportunities
        - 🏞️ Develop additional scenic lookout points
        - 🥾 Improve trail maintenance on popular hiking routes
        """)

with tab2:
    st.subheader("Sentiment Analysis by Park")
    
    # Filter control
    selected_park = st.selectbox("Select a Park:", ["All Parks"] + sorted(df['Park'].unique().tolist()))
    
    if selected_park != "All Parks":
        park_df = df[df['Park'] == selected_park]
    else:
        park_df = df
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create park sentiment summary
        park_sentiment = park_df.groupby('Park')['Sentiment'].value_counts().unstack().fillna(0)
        park_sentiment['Total'] = park_sentiment.sum(axis=1)
        park_sentiment['Positive %'] = (park_sentiment['POSITIVE'] / park_sentiment['Total'] * 100).round(1)
        park_sentiment = park_sentiment.sort_values(by='Positive %', ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            park_sentiment.reset_index(), 
            y='Park',
            x=['POSITIVE', 'NEGATIVE'],
            title="Sentiment Distribution by Park",
            color_discrete_map={'POSITIVE': '#4CAF50', 'NEGATIVE': '#F44336'},
            labels={'value': 'Count', 'variable': 'Sentiment'},
            orientation='h',
            barmode='stack'
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if selected_park != "All Parks":
            # Feature breakdown for selected park
            st.subheader(f"Feature Sentiment for {selected_park}")
            
            park_feature_sentiment = park_df.groupby('Feature')['Sentiment'].value_counts().unstack().fillna(0)
            park_feature_sentiment['Total'] = park_feature_sentiment.sum(axis=1)
            park_feature_sentiment['Positive %'] = (park_feature_sentiment['POSITIVE'] / park_feature_sentiment['Total'] * 100).round(1)
            park_feature_sentiment = park_feature_sentiment.sort_values(by='Positive %', ascending=False)
            
            # Display as a table
            st.dataframe(park_feature_sentiment[['POSITIVE', 'NEGATIVE', 'Positive %']])
        else:
            # Park ranking
            st.subheader("Park Ranking by Positive Sentiment")
            
            # Calculate positive sentiment percentage for each park
            park_ranking = park_sentiment[['POSITIVE', 'NEGATIVE', 'Positive %']].sort_values(by='Positive %', ascending=False)
            
            # Create a gauge chart for each park
            for park, row in park_ranking.iterrows():
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = row['Positive %'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': park},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4CAF50"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffcdd2"},
                            {'range': [50, 75], 'color': "#ffecb3"},
                            {'range': [75, 100], 'color': "#c8e6c9"}
                        ]
                    }
                ))
                fig.update_layout(height=150, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Sentiment Analysis by Feature")
    
    # Filter control
    selected_feature = st.selectbox("Select a Feature:", ["All Features"] + sorted(df['Feature'].unique().tolist()))
    
    if selected_feature != "All Features":
        feature_df = df[df['Feature'] == selected_feature]
    else:
        feature_df = df
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create feature-park heatmap
        pivot_data = []
        for (park, feature), group in feature_df.groupby(['Park', 'Feature']):
            # Calculate sentiment score: +1 for POSITIVE, -1 for NEGATIVE
            sentiment_score = sum(1 if s == 'POSITIVE' else -1 for s in group['Sentiment'])
            pivot_data.append({
                'Park': park,
                'Feature': feature,
                'Sentiment Score': sentiment_score
            })
        
        pivot_df = pd.DataFrame(pivot_data)
        
        if not pivot_df.empty:
            if selected_feature == "All Features":
                # Show heatmap of all features across parks
                heatmap_pivot = pivot_df.pivot(index='Park', columns='Feature', values='Sentiment Score')
                
                # Create heatmap with plotly
                fig = px.imshow(
                    heatmap_pivot,
                    labels=dict(x="Feature", y="Park", color="Sentiment Score"),
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                fig.update_layout(title="Feature Sentiment Score by Park")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show specific feature across parks
                feature_parks = pivot_df[pivot_df['Feature'] == selected_feature]
                feature_parks = feature_parks.sort_values(by='Sentiment Score')
                
                fig = px.bar(
                    feature_parks,
                    x='Sentiment Score',
                    y='Park',
                    title=f"{selected_feature} Sentiment Score by Park",
                    color='Sentiment Score',
                    color_continuous_scale='RdBu',
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if selected_feature != "All Features":
            # Create wordcloud from example sentences for the selected feature
            st.subheader(f"Common Terms in {selected_feature} Reviews")
            
            # Combine example sentences
            all_examples = " ".join(feature_df["Example Sentence 1"].dropna().tolist() + 
                                  feature_df["Example Sentence 2"].dropna().tolist())
            
            if all_examples:
                try:
                    # Download NLTK stopwords if not already downloaded
                    try:
                        nltk.data.find('corpora/stopwords')
                    except LookupError:
                        nltk.download('stopwords')
                    
                    # Generate wordcloud
                    stop_words = set(stopwords.words('english'))
                    wordcloud = WordCloud(width=400, height=300, background_color='white',
                                        stopwords=stop_words, max_words=100).generate(all_examples)
                    
                    # Display wordcloud
                    plt.figure(figsize=(8, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                except Exception as e:
                    st.warning(f"Could not generate wordcloud: {str(e)}")
        else:
            # Feature comparison
            st.subheader("Feature Comparison")
            
            # Calculate average sentiment score for each feature
            feature_scores = feature_df.groupby('Feature')['Sentiment'].apply(
                lambda x: (x == 'POSITIVE').mean() * 100
            ).sort_values(ascending=False)
            
            fig = px.bar(
                feature_scores.reset_index(),
                x='Sentiment',
                y='Feature',
                title="Positive Sentiment Percentage by Feature",
                labels={'Sentiment': 'Positive Sentiment %'},
                orientation='h',
                color='Sentiment',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Review Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        review_park = st.selectbox("Filter by Park:", ["All"] + sorted(df['Park'].unique().tolist()), key="review_park")
    
    with col2:
        review_feature = st.selectbox("Filter by Feature:", ["All"] + sorted(df['Feature'].unique().tolist()), key="review_feature")
    
    # Filter reviews
    filtered_df = df.copy()
    if review_park != "All":
        filtered_df = filtered_df[filtered_df['Park'] == review_park]
    if review_feature != "All":
        filtered_df = filtered_df[filtered_df['Feature'] == review_feature]
    
    # Sort and display reviews
    filtered_df = filtered_df.sort_values(by=['Park', 'Feature'])
    
    for _, row in filtered_df.iterrows():
        with st.expander(f"{row['Park']} - {row['Feature']} ({row['Sentiment']})"):
            sentiment_color = "#c8e6c9" if row['Sentiment'] == "POSITIVE" else "#ffcdd2"
            st.markdown(f"<div style='background-color:{sentiment_color}; padding:10px; border-radius:5px;'>"
                      f"<p><strong>Example 1:</strong> {row['Example Sentence 1']}</p>"
                      f"<p><strong>Example 2:</strong> {row['Example Sentence 2']}</p>"
                      f"<p><strong>Sentiment Score:</strong> {row['Score']:.2f}</p>"
                      "</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### About this Dashboard")
st.markdown("""
This dashboard analyzes sentiment from national park reviews using natural language processing.
It helps identify what visitors appreciate and what aspects of the parks could be improved.

**Features:**
- Sentiment analysis by park and feature
- Interactive visualizations and filtering
- Review examples with sentiment highlighting
- Recommendations for park managers based on sentiment patterns
""")
