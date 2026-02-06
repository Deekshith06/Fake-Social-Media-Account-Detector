"""
Instagram Profile Fetcher
Extracts account features from Instagram profiles for fake account detection.
"""

import re
from datetime import datetime
import instaloader


def fetch_instagram_profile(username):
    """
    Fetch Instagram profile data by username using instaloader.
    
    Args:
        username (str): Instagram username (without @)
    
    Returns:
        dict: Profile data with all required features
    """
    # Remove @ if present
    username = username.lstrip('@')
    
    try:
        # Create instaloader instance
        L = instaloader.Instaloader()
        
        # Fetch profile
        profile = instaloader.Profile.from_username(L.context, username)
        
        # Extract profile data
        profile_data = {
            'username': username,
            'platform': 'Instagram',
            'has_profile_pic': 1 if profile.profile_pic_url else 0,
            'verified': 1 if profile.is_verified else 0,
            'followers': profile.followers,
            'following': profile.followees,
            'posts': profile.mediacount,
            'bio_length': len(profile.biography) if profile.biography else 0,
            'is_private': profile.is_private,
            'full_name': profile.full_name if profile.full_name else '',
        }
        
        return profile_data
        
    except instaloader.exceptions.ProfileNotExistsException:
        return {"error": f"User @{username} not found"}
    except instaloader.exceptions.ConnectionException as e:
        return {"error": f"Connection error: {str(e)}"}
    except instaloader.exceptions.QueryReturnedNotFoundException:
        return {"error": f"Profile @{username} not found or is private"}
    except Exception as e:
        return {"error": f"Error fetching profile: {str(e)}"}


def extract_features_from_profile(profile_data):
    """
    Convert raw Instagram profile data to model features.
    
    Args:
        profile_data (dict): Raw profile data
    
    Returns:
        dict: Features ready for model prediction
    """
    if "error" in profile_data:
        return profile_data
    
    features = {
        'platform': profile_data.get('platform', 'Instagram'),
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
        'special_char_count': sum(not c.isalnum() and c != '_' and c != '.' for c in profile_data.get('username', '')),
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
    
    # Check for underscores/special chars (excluding . and _ which are normal in Instagram)
    special_count = sum(not c.isalnum() and c != '_' and c != '.' for c in username)
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
    test_username = "instagram"
    print(f"Fetching profile for @{test_username}...")
    
    profile = fetch_instagram_profile(test_username)
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
