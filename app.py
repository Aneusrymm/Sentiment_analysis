import streamlit as st
st.set_page_config(page_title="Advanced YouTube Comment Sentiment Analysis", layout="wide")
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
from PIL import Image
import base64
from io import BytesIO
import time
import os

nltk.data.path.append("/tmp/nltk_data")
os.makedirs("/tmp/nltk_data", exist_ok=True)
# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_resources()

# Initialize session state for caching data
if 'analyzed_videos' not in st.session_state:
    st.session_state['analyzed_videos'] = {}

# Define function to get YouTube video ID from URL
def extract_video_id(url):
    # Regular expressions to extract video ID from different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?]+)',
        r'youtube\.com\/watch\?.+&v=([^&\n?]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

# Function to get video information
def get_video_info(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
        
        if response['items']:
            item = response['items'][0]
            video_info = {
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'publish_date': item['snippet']['publishedAt'],
                'view_count': item['statistics'].get('viewCount', 0),
                'like_count': item['statistics'].get('likeCount', 0),
                'comment_count': item['statistics'].get('commentCount', 0)
            }
            return video_info
        else:
            return None
    except HttpError as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to get comments from YouTube API with pagination and time tracking
def get_comments(api_key, video_id, max_results=100):
    # Create YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        # Get comments
        comments = []
        comment_data = []
        next_page_token = None
        
        with st.progress(0) as progress_bar:
            # Continue fetching comments until we reach the desired number or there are no more
            while len(comments) < max_results:
                # Make API request
                response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token,
                    textFormat='plainText'
                ).execute()
                
                # Extract comments from the response
                for item in response['items']:
                    comment_snippet = item['snippet']['topLevelComment']['snippet']
                    comment = comment_snippet['textDisplay']
                    
                    # Also collect metadata
                    comment_info = {
                        'text': comment,
                        'author': comment_snippet['authorDisplayName'],
                        'published_at': comment_snippet['publishedAt'],
                        'like_count': comment_snippet['likeCount'],
                        'reply_count': item['snippet'].get('totalReplyCount', 0)
                    }
                    
                    comments.append(comment)
                    comment_data.append(comment_info)
                
                # Update progress bar
                progress_bar.progress(min(len(comments) / max_results, 1.0))
                
                # Check if there are more pages
                next_page_token = response.get('nextPageToken')
                if not next_page_token or len(comments) >= max_results:
                    break
                    
            return comments, comment_data
            
    except HttpError as e:
        st.error(f"An error occurred: {e}")
        return [], []

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Keep some basic punctuation that might be meaningful
    text = re.sub(r'[^\w\s\'-]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords but keep short meaningful words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return tokens

# Function to analyze sentiment with multiple methods
def analyze_sentiment_multiple(text):
    # TextBlob sentiment
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity
    
    # VADER sentiment
    sid = SentimentIntensityAnalyzer()
    vader_scores = sid.polarity_scores(text)
    
    # Determine overall sentiment category
    # Weighted average of TextBlob and VADER
    overall_score = (textblob_polarity + vader_scores['compound']) / 2
    
    if overall_score > 0.1:
        sentiment = 'Positive'
    elif overall_score < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    # More detailed emotion detection
    emotion = detect_emotion(text)
    
    return {
        'sentiment': sentiment,
        'textblob_polarity': textblob_polarity,
        'textblob_subjectivity': textblob_subjectivity,
        'vader_compound': vader_scores['compound'],
        'vader_pos': vader_scores['pos'],
        'vader_neu': vader_scores['neu'],
        'vader_neg': vader_scores['neg'],
        'emotion': emotion
    }

# Simple emotion detection function based on keyword matching
def detect_emotion(text):
    text = text.lower()
    
    # Simple keyword-based emotion detection
    emotions = {
        'happy': ['happy', 'joy', 'delighted', 'great', 'excellent', 'love', 'awesome'],
        'sad': ['sad', 'unhappy', 'depressed', 'terrible', 'disappointed', 'awful'],
        'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'hate'],
        'surprised': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow'],
        'fearful': ['scared', 'afraid', 'worried', 'nervous', 'terrified'],
        'disgusted': ['disgusted', 'gross', 'yuck', 'horrible']
    }
    
    # Count occurrences of emotion keywords
    emotion_counts = {emotion: 0 for emotion in emotions}
    
    for emotion, keywords in emotions.items():
        for keyword in keywords:
            if keyword in text:
                emotion_counts[emotion] += 1
    
    # Get the dominant emotion
    if sum(emotion_counts.values()) > 0:
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        return dominant_emotion
    else:
        return 'neutral'

# Function to perform topic modeling
def perform_topic_modeling(comments, num_topics=5):
    # Vectorize the comments
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(comments)
    
    # Create and fit the LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the top words for each topic
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
    
    return topics

# Function to create word frequency visualization with Plotly
def create_word_frequency_viz(all_tokens):
    # Count word frequencies
    word_freq = Counter(all_tokens)
    top_words = word_freq.most_common(20)
    
    # Create DataFrame for visualization
    df_freq = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
    # Create interactive bar chart with Plotly
    fig = px.bar(df_freq, x='Frequency', y='Word', orientation='h',
                color='Frequency', color_continuous_scale='Viridis',
                title='Top 20 Most Frequent Words in Comments')
    
    fig.update_layout(height=600)
    
    return fig

# Function to create word cloud
def create_wordcloud(all_tokens, color_theme='viridis'):
    # Join tokens into a single string
    text = ' '.join(all_tokens)
    
    # Check if we have any valid words
    if not text.strip():
        st.warning("Not enough words to generate word cloud after filtering")
        return None
    
    try:
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=200, contour_width=3, contour_color='steelblue',
                            colormap=color_theme)
        wordcloud.generate(text)
        
        # Convert WordCloud to image
        img = wordcloud.to_image()
        
        # Convert PIL image to base64 string for display
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    except ValueError as e:
        st.warning(f"Could not generate word cloud: {str(e)}")
        return None

# Function to create sentiment visualization with Plotly
def create_sentiment_viz(df):
    # Count sentiment categories
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Create interactive pie chart
    fig = px.pie(sentiment_counts, values='Count', names='Sentiment',
                color='Sentiment', color_discrete_map={'Positive': '#66bb6a', 
                                                     'Neutral': '#42a5f5', 
                                                     'Negative': '#ef5350'},
                title='Sentiment Distribution')
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

# Function to create emotion visualization
def create_emotion_viz(df):
    # Count emotion categories
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']
    
    # Create interactive bar chart
    fig = px.bar(emotion_counts, x='Emotion', y='Count', 
                color='Emotion', title='Emotion Distribution')
    
    fig.update_layout(height=500)
    
    return fig

# Function to create time series analysis of comments
def create_time_series_viz(df):
    # Convert published_at to datetime
    df['published_date'] = pd.to_datetime(df['published_at'])
    
    # Group by date and calculate metrics
    daily_sentiment = df.groupby(df['published_date'].dt.date).agg(
        comment_count=('sentiment', 'count'),
        positive_ratio=('sentiment', lambda x: (x == 'Positive').mean()),
        negative_ratio=('sentiment', lambda x: (x == 'Negative').mean()),
        neutral_ratio=('sentiment', lambda x: (x == 'Neutral').mean()),
        avg_polarity=('textblob_polarity', 'mean')
    ).reset_index()
    
    # Create line chart
    fig = px.line(daily_sentiment, x='published_date', y=['positive_ratio', 'negative_ratio', 'neutral_ratio'],
                 title='Daily Sentiment Trends',
                 labels={'value': 'Ratio', 'published_date': 'Date', 'variable': 'Sentiment Type'})
    
    fig.update_layout(height=500)
    
    return fig

# Function to visualize sentiment by engagement
def create_engagement_viz(df):
    # Create scatter plot of sentiment vs. likes
    fig = px.scatter(df, x='like_count', y='textblob_polarity', 
                    color='sentiment', size='reply_count',
                    hover_data=['author', 'text'],
                    title='Comment Sentiment vs. Engagement',
                    labels={'like_count': 'Likes', 
                           'textblob_polarity': 'Sentiment Polarity',
                           'reply_count': 'Replies'})
    
    fig.update_layout(height=600)
    
    return fig

# Main application with tabs
def main():
    st.title("Advanced YouTube Comment Sentiment Analysis")
    
    with st.sidebar:
        st.header("Settings")
        
        # API Key input
        api_key = st.text_input("Enter your YouTube API Key", type="password")
        
        # YouTube link input
        youtube_url = st.text_input("Enter YouTube Video URL")
        
        # Number of comments to analyze
        max_comments = st.slider("Maximum number of comments to analyze", 10, 1000, 200)
        
        # Analysis options
        st.subheader("Analysis Options")
        show_topics = st.checkbox("Perform Topic Modeling", value=True)
        topic_count = st.slider("Number of Topics", 3, 10, 5) if show_topics else 5
        
        show_time_analysis = st.checkbox("Perform Time-based Analysis", value=True)
        show_engagement_analysis = st.checkbox("Perform Engagement Analysis", value=True)
        
        # Visualization options
        st.subheader("Visualization Options")
        color_theme = st.selectbox("Word Cloud Color Theme", 
                                 ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Reds"])
        
        # Run Analysis
        analyze_button = st.button("Analyze Comments")
    
    if analyze_button:
        if not api_key:
            st.error("Please enter a YouTube API Key")
        elif not youtube_url:
            st.error("Please enter a YouTube Video URL")
        else:
            # Extract video ID
            video_id = extract_video_id(youtube_url)
            
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
            else:
                # Check if we've already analyzed this video
                if video_id in st.session_state['analyzed_videos']:
                    st.success("Loading cached results")
                    analysis_data = st.session_state['analyzed_videos'][video_id]
                    video_info = analysis_data['video_info']
                    df = analysis_data['df']
                    all_tokens = analysis_data['all_tokens']
                    topics = analysis_data.get('topics', None)
                else:
                    with st.spinner("Fetching video information..."):
                        # Get video info
                        video_info = get_video_info(api_key, video_id)
                        
                        if not video_info:
                            st.error("Could not retrieve video information")
                            return
                    
                    with st.spinner("Fetching and analyzing comments..."):
                        # Get comments
                        comments, comment_data = get_comments(api_key, video_id, max_comments)
                        
                        if not comments:
                            st.error("No comments found or unable to retrieve comments")
                            return
                        
                        # Create DataFrame
                        df = pd.DataFrame(comment_data)
                        
                        # Analyze sentiment
                        sentiment_results = []
                        for text in comments:
                            results = analyze_sentiment_multiple(text)
                            sentiment_results.append(results)
                        
                        # Add sentiment analysis results to dataframe
                        for key in sentiment_results[0].keys():
                            df[key] = [result[key] for result in sentiment_results]
                        
                        # Preprocess all comments
                        all_tokens = []
                        for comment in comments:
                            tokens = preprocess_text(comment)
                            all_tokens.extend(tokens)
                        
                        # Perform topic modeling if requested
                        topics = None
                        if show_topics:
                            topics = perform_topic_modeling(comments, num_topics=topic_count)
                        
                        # Cache the results
                        st.session_state['analyzed_videos'][video_id] = {
                            'video_info': video_info,
                            'df': df,
                            'all_tokens': all_tokens,
                            'topics': topics
                        }
                
                # Display video information
                st.header(video_info['title'])
                col1, col2, col3 = st.columns(3)
                col1.metric("Channel", video_info['channel'])
                col2.metric("Views", f"{int(video_info['view_count']):,}")
                col3.metric("Comments Analyzed", f"{len(df)}")
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "😀 Sentiment Analysis", 
                    "📊 Word Frequency", 
                    "🗣️ Topic Analysis", 
                    "📈 Time & Engagement", 
                    "📝 Raw Data"
                ])
                
                # Tab 1: Sentiment Analysis
                with tab1:
                    st.subheader("Sentiment Analysis Results")
                    
                    # Sentiment overview
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart of sentiment
                        fig_sentiment = create_sentiment_viz(df)
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    with col2:
                        # Emotion analysis
                        fig_emotion = create_emotion_viz(df)
                        st.plotly_chart(fig_emotion, use_container_width=True)
                    
                    # Sentiment statistics
                    st.subheader("Sentiment Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    positive_pct = (df['sentiment'] == 'Positive').mean() * 100
                    negative_pct = (df['sentiment'] == 'Negative').mean() * 100
                    neutral_pct = (df['sentiment'] == 'Neutral').mean() * 100
                    
                    col1.metric("Positive Comments", f"{positive_pct:.1f}%")
                    col2.metric("Neutral Comments", f"{neutral_pct:.1f}%")
                    col3.metric("Negative Comments", f"{negative_pct:.1f}%")
                    
                    # Show most positive and negative comments
                    st.subheader("Most Positive Comments")
                    top_positive = df.sort_values('textblob_polarity', ascending=False).head(3)
                    for i, row in top_positive.iterrows():
                        st.info(f"**{row['author']}**: {row['text']}")
                    
                    st.subheader("Most Negative Comments")
                    top_negative = df.sort_values('textblob_polarity').head(3)
                    for i, row in top_negative.iterrows():
                        st.error(f"**{row['author']}**: {row['text']}")
                
                # Tab 2: Word Frequency
                with tab2:
                    st.subheader("Word Frequency Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Word frequency bar chart
                        fig_freq = create_word_frequency_viz(all_tokens)
                        st.plotly_chart(fig_freq, use_container_width=True)
                    
                    with col2:
                        # Word cloud
                        st.subheader("Word Cloud")
                        wordcloud_img = create_wordcloud(all_tokens, color_theme)
                        st.image(f"data:image/png;base64,{wordcloud_img}", use_column_width=True)
                        
                        # Word frequency by sentiment
                        st.subheader("Word Usage by Sentiment")
                        sentiment_option = st.selectbox("Select Sentiment", ['Positive', 'Neutral', 'Negative'])
                        
                        # Filter comments by sentiment and create word cloud
                        sentiment_comments = df[df['sentiment'] == sentiment_option]['text'].tolist()
                        if sentiment_comments:
                            sentiment_tokens = []
                            for comment in sentiment_comments:
                                tokens = preprocess_text(comment)
                                sentiment_tokens.extend(tokens)
                            
                            if sentiment_tokens:
                                sentiment_wordcloud = create_wordcloud(sentiment_tokens, color_theme)
                                st.image(f"data:image/png;base64,{sentiment_wordcloud}", use_column_width=True)
                            else:
                                st.write("Not enough data for this sentiment category")
                        else:
                            st.write("No comments found in this sentiment category")
                
                # Tab 3: Topic Analysis
                with tab3:
                    st.subheader("Topic Analysis")
                    
                    if topics and show_topics:
                        # Display topics
                        for i, topic_words in enumerate(topics):
                            with st.expander(f"Topic {i+1}"):
                                st.write(", ".join(topic_words))
                                
                                # Find example comments for this topic
                                topic_keywords = set(topic_words)
                                example_comments = []
                                
                                for _, row in df.iterrows():
                                    comment_words = set(preprocess_text(row['text']))
                                    overlap = comment_words.intersection(topic_keywords)
                                    if len(overlap) >= 2:  # At least 2 keywords from the topic
                                        example_comments.append((row['text'], row['author'], len(overlap)))
                                
                                # Sort by relevance (number of overlapping words)
                                example_comments.sort(key=lambda x: x[2], reverse=True)
                                
                                # Show top examples
                                if example_comments:
                                    st.write("**Example Comments:**")
                                    for comment, author, _ in example_comments[:3]:
                                        st.write(f"**{author}**: {comment}")
                                else:
                                    st.write("No clear examples found for this topic")
                    else:
                        st.info("Topic modeling was not performed. Enable it in the settings to see topics.")
                
                # Tab 4: Time & Engagement Analysis
                with tab4:
                    if show_time_analysis:
                        st.subheader("Comment Sentiment Over Time")
                        time_series_fig = create_time_series_viz(df)
                        st.plotly_chart(time_series_fig, use_container_width=True)
                    
                    if show_engagement_analysis:
                        st.subheader("Engagement Analysis")
                        engagement_fig = create_engagement_viz(df)
                        st.plotly_chart(engagement_fig, use_container_width=True)
                        
                        # Comment popularity analysis
                        st.subheader("Most Popular Comments")
                        popular_comments = df.sort_values('like_count', ascending=False).head(5)
                        for i, row in popular_comments.iterrows():
                            st.write(f"**{row['author']}** (👍 {row['like_count']}, 💬 {row['reply_count']}): {row['text']}")
                    
                    if not show_time_analysis and not show_engagement_analysis:
                        st.info("Time and engagement analysis are disabled. Enable them in the settings.")
                
                # Tab 5: Raw Data
                with tab5:
                    st.subheader("Raw Comment Data")
                    
                    # Filter options
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        sentiment_filter = st.multiselect("Filter by Sentiment", 
                                                       options=['Positive', 'Neutral', 'Negative'],
                                                       default=['Positive', 'Neutral', 'Negative'])
                    
                    with filter_col2:
                        emotion_filter = st.multiselect("Filter by Emotion",
                                                     options=df['emotion'].unique().tolist(),
                                                     default=df['emotion'].unique().tolist())
                    
                    # Apply filters
                    filtered_df = df[df['sentiment'].isin(sentiment_filter) & df['emotion'].isin(emotion_filter)]
                    
                    # Show dataframe
                    st.dataframe(filtered_df[['author', 'text', 'published_at', 'like_count', 
                                           'reply_count', 'sentiment', 'emotion', 
                                           'textblob_polarity', 'vader_compound']])
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Convert dataframe to CSV
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"youtube_sentiment_{video_id}.csv",
                            mime="text/csv",
                        )
                    
                    with col2:
                        # Convert dataframe to JSON
                        json_data = filtered_df.to_json(orient="records")
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"youtube_sentiment_{video_id}.json",
                            mime="application/json",
                        )
    
    # Tutorial section
    with st.expander("How to Use This App"):
        st.markdown("""
        ### Getting Started
        1. **YouTube API Key**: You need a Google API key with YouTube Data API v3 enabled.
           - Go to [Google Cloud Console](https://console.cloud.google.com/)
           - Create a new project
           - Enable YouTube Data API v3
           - Create an API key and copy it
        
        2. **Analysis Options**:
           - Enter the YouTube video URL
           - Choose how many comments to analyze
           - Select additional analysis options
        
        3. **Features**:
           - **Sentiment Analysis**: See positive, neutral, and negative distribution
           - **Word Frequency**: Find most common words and visualize with word cloud
           - **Topic Analysis**: Discover key topics in the comments
           - **Time & Engagement**: Analyze sentiment trends and engagement metrics
           - **Raw Data**: View and export detailed comment data

        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Created with Streamlit | YouTube Comment Sentiment Analysis Tool"
    )

# Run the application
if __name__ == "__main__":
    main()