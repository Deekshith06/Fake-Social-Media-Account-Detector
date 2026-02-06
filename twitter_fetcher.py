"""
Twitter Profile Fetcher
Extracts account features from Twitter/X profiles for fake account detection.
"""

import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup


def fetch_twitter_profile(username):
    """
    Fetch Twitter profile data by username using web scraping.
    
    Args:
        username (str): Twitter username (without @)
    
    Returns:
        dict: Profile data with all required features
    """
    # Remove @ if present
    username = username.lstrip('@')
    
    try:
        # Use nitter instance (Twitter frontend) for scraping
        # Note: This is a fallback method and may not always work
        url = f"https://nitter.net/{username}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            return {"error": f"User @{username} not found"}
        elif response.status_code != 200:
            return {"error": f"Failed to fetch profile (status code: {response.status_code})"}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract profile data
        profile_data = {
            'username': username,
            'platform': 'X',  # Twitter/X
            'has_profile_pic': 1,  # Assume yes if profile loads
            'verified': 0,
            'followers': 0,
            'following': 0,
            'posts': 0,
            'bio_length': 0,
        }
        
        # Try to extract stats
        stats = soup.find_all('span', class_='profile-stat-num')
        if len(stats) >= 3:
            try:
                profile_data['posts'] = int(stats[0].text.replace(',', ''))
                profile_data['following'] = int(stats[1].text.replace(',', ''))
                profile_data['followers'] = int(stats[2].text.replace(',', ''))
            except:
                pass
        
        # Extract bio
        bio = soup.find('div', class_='profile-bio')
        if bio:
            profile_data['bio_length'] = len(bio.text.strip())
        
        # Check verification
        if soup.find('span', class_='icon-ok'):
            profile_data['verified'] = 1
        
        return profile_data
        
    except requests.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error fetching profile: {str(e)}"}


def extract_features_from_profile(profile_data):
    """
    Convert raw profile data to model features.
    
    Args:
        profile_data (dict): Raw profile data
    
    Returns:
        dict: Features ready for model prediction
    """
    if "error" in profile_data:
        return profile_data
    
    features = {
        'platform': profile_data.get('platform', 'X'),
        'has_profile_pic': profile_data.get('has_profile_pic', 1),
        'bio_length': profile_data.get('bio_length', 0),
        'username_randomness': calculate_username_randomness(profile_data.get('username', '')),
        'followers': profile_data.get('followers', 0),
        'following': profile_data.get('following', 0),
        'follower_following_ratio': calculate_ratio(
            profile_data.get('followers', 0),
            profile_data.get('following', 1)
        ),
        'account_age_days': 365,  # Default estimate (1 year)
        'posts': profile_data.get('posts', 0),
        'posts_per_day': profile_data.get('posts', 0) / 365.0,
        'caption_similarity_score': 0.5,  # Default neutral value
        'content_similarity_score': 0.5,  # Default neutral value
        'follow_unfollow_rate': 0.1,  # Default low value
        'spam_comments_rate': 0.0,  # Default
        'generic_comment_rate': 0.0,  # Default
        'suspicious_links_in_bio': 0,  # Default
        'verified': profile_data.get('verified', 0),
        'username': profile_data.get('username', ''),
        'username_length': len(profile_data.get('username', '')),
        'digits_count': sum(c.isdigit() for c in profile_data.get('username', '')),
        'digit_ratio': calculate_digit_ratio(profile_data.get('username', '')),
        'special_char_count': sum(not c.isalnum() for c in profile_data.get('username', '')),
        'repeat_char_count': count_repeat_chars(profile_data.get('username', '')),
    }
    
    return features


def calculate_username_randomness(username):
    """Calculate randomness score of username (0-1)."""
    if not username:
        return 0.5
    
    # Higher score = more random
    score = 0.0
    
    # Check for excessive numbers
    digit_ratio = sum(c.isdigit() for c in username) / max(len(username), 1)
    score += digit_ratio * 0.4
    
    # Check for random character patterns
    if re.search(r'[0-9]{3,}', username):  # 3+ consecutive digits
        score += 0.3
    
    # Check for underscores/special chars
    special_count = sum(not c.isalnum() for c in username)
    score += min(special_count / len(username), 0.3)
    
    return min(score, 1.0)


def calculate_ratio(numerator, denominator):
    """Safely calculate ratio."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_digit_ratio(username):
    """Calculate ratio of digits to total characters."""
    if not username:
        return 0.0
    return sum(c.isdigit() for c in username) / len(username)


def count_repeat_chars(username):
    """Count repeated consecutive characters."""
    if not username:
        return 0
    
    count = 0
    for i in range(len(username) - 1):
        if username[i] == username[i + 1]:
            count += 1
    return count


if __name__ == "__main__":
    # Test the fetcher
    test_username = "elonmusk"
    print(f"Fetching profile for @{test_username}...")
    
    profile = fetch_twitter_profile(test_username)
    if "error" in profile:
        print(f"Error: {profile['error']}")
    else:
        print(f"\nProfile Data:")
        for key, value in profile.items():
            print(f"  {key}: {value}")
        
        print(f"\nExtracted Features:")
        features = extract_features_from_profile(profile)
        for key, value in features.items():
            print(f"  {key}: {value}")
