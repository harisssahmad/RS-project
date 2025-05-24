import pickle
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os

# Configure page
st.set_page_config(
    page_title="Anime Recommender System",
    page_icon="🎌",
    layout="wide"
)

@st.cache_data
def load_content_based_data():
    """Load content-based recommendation data"""
    try:
        animes = pickle.load(open('anime_list.pkl', 'rb'))
        animes = pd.DataFrame(animes)
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return animes, similarity, True
    except Exception as e:
        st.error(f"Error loading content-based data: {e}")
        return None, None, False

@st.cache_resource
def load_neural_network_model():
    """Load neural network model and mappings"""
    try:
        # Load the trained model
        model = load_model('anime_recommendation_model.h5', 
                          custom_objects={'mse': MeanSquaredError()})
        
        # Load mappings and anime data
        user_id_mapping = pickle.load(open('user_id_mapping.pkl', 'rb'))
        anime_id_mapping = pickle.load(open('anime_id_mapping.pkl', 'rb'))
        anime_df = pd.read_csv('cleaned_anime.csv')
        
        return model, user_id_mapping, anime_id_mapping, anime_df, True
    except Exception as e:
        st.error(f"Error loading neural network model: {e}")
        return None, None, None, None, False

@st.cache_data
def load_ratings_data():
    """Load user ratings data"""
    try:
        ratings_df = pd.read_csv('cleaned_rating.csv')
        return ratings_df, True
    except Exception as e:
        st.error(f"Error loading ratings data: {e}")
        return None, False

def get_user_existing_ratings(user_id, ratings_df, anime_df, top_n=10):
    """Get existing ratings for a specific user"""
    try:
        # Filter ratings for the specific user
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
        
        if user_ratings.empty:
            return None, f"No ratings found for user {user_id}"
        
        # Merge with anime data to get anime names and details
        user_ratings_with_names = user_ratings.merge(
            anime_df[['anime_id', 'name', 'genre', 'type', 'episodes']], 
            on='anime_id', 
            how='left'
        )
        
        # Sort by rating (highest first)
        user_ratings_with_names = user_ratings_with_names.sort_values('rating', ascending=False)
        
        return user_ratings_with_names.head(top_n), None
        
    except Exception as e:
        return None, f"An error occurred: {e}"

def get_content_based_recommendations(anime_selected, animes, similarity, num_recommendations=5):
    """Get content-based recommendations"""
    try:
        idx = animes[animes["name"] == anime_selected].index[0]
        
        if idx >= len(similarity):
            return None, "Similarity matrix index out of bounds."
        
        similarities = similarity[idx]
        sorted_indices = similarities.argsort()[::-1][1:num_recommendations+1]
        
        # Ensure indices are within bounds
        valid_indices = [i for i in sorted_indices if i < len(animes)]
        
        if not valid_indices:
            return None, "No recommendations found for this anime."
        
        recommended_animes = animes.iloc[valid_indices]
        return recommended_animes, None
        
    except Exception as e:
        return None, f"An error occurred: {e}"

def get_neural_network_recommendations(user_id, model, user_id_mapping, anime_id_mapping, anime_df, top_n=10):
    """Get neural network-based recommendations"""
    try:
        if user_id not in user_id_mapping:
            return None, f"User ID {user_id} not found in training data."

        mapped_user_id = user_id_mapping[user_id]
        all_anime = np.array(list(anime_id_mapping.values()))

        user_array = np.full_like(all_anime, mapped_user_id)
        predicted_ratings = model.predict([user_array, all_anime], verbose=0)

        # Sort anime by predicted rating
        top_indices = np.argsort(predicted_ratings[:, 0])[::-1][:top_n]
        
        # Map back to original anime IDs
        reverse_anime_mapping = {v: k for k, v in anime_id_mapping.items()}
        recommended_anime_ids = [reverse_anime_mapping[all_anime[idx]] for idx in top_indices]

        # Return recommendations with ratings
        recommendations = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)].copy()
        
        # Add predicted ratings
        rating_dict = {reverse_anime_mapping[all_anime[idx]]: predicted_ratings[idx, 0] 
                      for idx in top_indices}
        recommendations['predicted_rating'] = recommendations['anime_id'].map(rating_dict)
        
        # Sort by predicted rating
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        return recommendations[['name', 'genre', 'predicted_rating']].head(top_n), None
        
    except Exception as e:
        return None, f"An error occurred: {e}"

# Main app
def main():
    st.title('🎌 Anime Recommender System')
    st.write("Welcome! This system offers two types of anime recommendations to help you discover new shows you might enjoy.")
    
    # Load data
    animes, similarity, content_loaded = load_content_based_data()
    model, user_id_mapping, anime_id_mapping, anime_df, nn_loaded = load_neural_network_model()
    ratings_df, ratings_loaded = load_ratings_data()
    
    # Sidebar for recommendation type selection
    st.sidebar.title("🎯 Recommendation Type")
    recommendation_type = st.sidebar.radio(
        "Choose your recommendation method:",
        ["Content-Based Filtering", "Neural Network Collaborative Filtering"],
        help="Content-based uses anime features, while Neural Network learns from user ratings"
    )
    
    if recommendation_type == "Content-Based Filtering":
        st.header("📊 Content-Based Recommendations")
        st.write("This method recommends anime similar to the one you select based on content features.")
        
        if not content_loaded:
            st.error("❌ Content-based recommendation data not available. Please ensure 'anime_list.pkl' and 'similarity.pkl' are in the directory.")
            return
        
        # User selects an anime
        anime_selected = st.selectbox("🎬 Type or select an anime:", animes["name"].values)
        
        # Number of recommendations
        num_recs = st.slider("Number of recommendations:", 1, 10, 5)
        
        # Display recommended anime when button is clicked
        if st.button('🔍 Show Recommendations', key='content_btn'):
            if anime_selected not in animes["name"].values:
                st.warning("⚠️ Selected anime not found in the dataset.")
            else:
                with st.spinner("Finding similar anime..."):
                    recommendations, error = get_content_based_recommendations(
                        anime_selected, animes, similarity, num_recs
                    )
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success(f"✅ Found {len(recommendations)} recommendations!")
                    st.subheader(f'🎯 Anime Similar to "{anime_selected}"')
                    
                    # Display recommendations in a nice format
                    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.write(f"**{i}.**")
                            with col2:
                                st.write(f"**{row['name']}**")
                                if 'genre' in row and pd.notna(row['genre']):
                                    st.write(f"*Genre: {row['genre']}*")
                            st.divider()
    
    else:  # Neural Network Collaborative Filtering
        st.header("🧠 Neural Network Collaborative Filtering")
        st.write("This method uses deep learning to predict ratings based on user preferences and collaborative patterns.")
        
        if not nn_loaded:
            st.error("❌ Neural network model not available. Please ensure the following files are in the directory:")
            st.write("- `anime_recommendation_model.h5`")
            st.write("- `user_id_mapping.pkl`")
            st.write("- `anime_id_mapping.pkl`")
            st.write("- `cleaned_anime.csv`")
            return
        
        if not ratings_loaded:
            st.warning("⚠️ Ratings data not available. User existing ratings will not be displayed.")
        
        # User ID input
        st.subheader("👤 User Selection")
        
        # Show available users
        available_users = list(user_id_mapping.keys())
        st.info(f"📈 Available users: {len(available_users)} users in the system")
        
        # User input methods
        input_method = st.radio("Select user input method:", 
                               ["Choose from list", "Enter User ID manually"])
        
        if input_method == "Choose from list":
            # Show first 100 users for performance
            display_users = available_users[:100]
            user_id = st.selectbox("🔍 Select a User ID:", display_users)
        else:
            user_id = st.number_input("🔢 Enter User ID:", 
                                    min_value=min(available_users), 
                                    max_value=max(available_users),
                                    value=available_users[0])
        
        # Display user's existing ratings when a user is selected
        if ratings_loaded and user_id:
            st.subheader(f"⭐ User {user_id}'s Existing Ratings")
            
            # Number of existing ratings to show
            num_existing_ratings = st.slider("Number of existing ratings to show:", 1, 20, 10, key="existing_ratings_slider")
            
            existing_ratings, error = get_user_existing_ratings(user_id, ratings_df, anime_df, num_existing_ratings)
            
            if error:
                st.info(f"ℹ️ {error}")
            else:
                st.success(f"✅ Found {len(existing_ratings)} ratings for this user!")
                
                # Create tabs for better organization
                tab1, tab2 = st.tabs(["📊 Ratings Summary", "📝 Detailed Ratings"])
                
                with tab1:
                    # Display summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Ratings", len(existing_ratings))
                    with col2:
                        avg_rating = existing_ratings['rating'].mean()
                        st.metric("Average Rating", f"{avg_rating:.1f}/10")
                    with col3:
                        highest_rating = existing_ratings['rating'].max()
                        st.metric("Highest Rating", f"{highest_rating}/10")
                
                with tab2:
                    # Display detailed ratings
                    for i, (_, row) in enumerate(existing_ratings.iterrows(), 1):
                        with st.container():
                            col1, col2, col3 = st.columns([0.5, 4, 1.5])
                            with col1:
                                st.write(f"**{i}.**")
                            with col2:
                                st.write(f"**{row['name']}**")
                                if 'genre' in row and pd.notna(row['genre']):
                                    st.write(f"*{row['genre']}*")
                                if 'type' in row and pd.notna(row['type']):
                                    st.write(f"Type: {row['type']}")
                            with col3:
                                rating = row['rating']
                                # Color code the rating
                                if rating >= 8:
                                    st.success(f"⭐ {rating}/10")
                                elif rating >= 6:
                                    st.info(f"⭐ {rating}/10")
                                else:
                                    st.warning(f"⭐ {rating}/10")
                            st.divider()
        
        # Number of recommendations
        st.subheader("🤖 Get New Recommendations")
        num_recs = st.slider("Number of recommendations:", 1, 20, 10)
        
        # Display recommended anime when button is clicked
        if st.button('🤖 Get AI Recommendations', key='nn_btn'):
            if user_id not in available_users:
                st.warning(f"⚠️ User ID {user_id} not found in the training data.")
            else:
                with st.spinner("🧠 Neural network is analyzing preferences..."):
                    recommendations, error = get_neural_network_recommendations(
                        user_id, model, user_id_mapping, anime_id_mapping, anime_df, num_recs
                    )
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success(f"✅ Generated {len(recommendations)} personalized recommendations!")
                    st.subheader(f'🎯 Personalized Recommendations for User {user_id}')
                    
                    # Display recommendations with ratings
                    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            col1, col2, col3 = st.columns([1, 5, 2])
                            with col1:
                                st.write(f"**{i}.**")
                            with col2:
                                st.write(f"**{row['name']}**")
                                if 'genre' in row and pd.notna(row['genre']):
                                    st.write(f"*{row['genre']}*")
                            with col3:
                                rating = row['predicted_rating']
                                if rating > 10:
                                    st.metric("Predicted Rating", f"10.0/10")
                                else:
                                    st.metric("Predicted Rating", f"{rating:.1f}/10")
                            st.divider()
    
    # Comparison section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Method Comparison")
    
    with st.sidebar.expander("🔍 Learn More"):
        st.write("**Content-Based Filtering:**")
        st.write("• Uses anime features (genre, type, etc.)")
        st.write("• Recommends similar anime")
        st.write("• No user data needed")
        st.write("• Good for discovering similar content")
        
        st.write("**Neural Network Collaborative:**")
        st.write("• Uses user rating patterns")
        st.write("• Learns complex user preferences")
        st.write("• Provides rating predictions")
        st.write("• Personalized recommendations")

# Footer
footer = """
<style>
a:link, a:visited {
    color: #1f77b4;
    background-color: transparent;
    text-decoration: underline;
}

a:hover, a:active {
    color: #ff7f0e;
    background-color: transparent;
    text-decoration: underline;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0e1117;
    color: white;
    text-align: center;
    padding: 10px 0;
    z-index: 999;
}
</style>
<div class="footer">
<p>Made with ❤️ by K21-4677 Haris Ahmad, K21-3964 Saud Jamal, K21-3415 Muhammad Haris</p>
</div>
"""

if __name__ == "__main__":
    main()
    st.markdown(footer, unsafe_allow_html=True)