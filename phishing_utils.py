from urllib.parse import urlparse
import re

def extract_url_features(url, num_features):
    """
    Extract features from a URL for phishing detection.
    Returns a list of numerical features.
    """
    features = [0] * num_features  # Initialize with zeros
    
    try:
        # First, clean the URL by removing any whitespace
        url = url.strip()
        
        # Normalize URL by adding http:// if no protocol is specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        # Parse the URL
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ''
        path = parsed_url.path
        query = parsed_url.query
        netloc = parsed_url.netloc

        # Remove www. from hostname for feature extraction
        clean_hostname = hostname.replace('www.', '') if hostname.startswith('www.') else hostname

        # Basic URL features
        features[0] = len(url)  # length_url
        features[1] = len(clean_hostname)  # length_hostname
        features[2] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', clean_hostname) else 0  # ip
        features[3] = url.count('.')  # nb_dots
        features[4] = url.count('-')  # nb_hyphens
        features[5] = url.count('@')  # nb_at
        features[6] = url.count('?')  # nb_qm
        features[7] = url.count('&')  # nb_and
        features[8] = url.count('|')  # nb_or
        features[9] = url.count('=')  # nb_eq
        features[10] = url.count('_')  # nb_underscore
        features[11] = url.count('~')  # nb_tilde
        features[12] = url.count('%')  # nb_percent
        features[13] = url.count('/')  # nb_slash
        features[14] = url.count('*')  # nb_star
        features[15] = url.count(':')  # nb_colon
        features[16] = url.count(',')  # nb_comma
        features[17] = url.count(';')  # nb_semicolumn
        features[18] = url.count('$')  # nb_dollar
        features[19] = url.count(' ')  # nb_space
        features[20] = 1 if hostname.startswith('www.') else 0  # nb_www
        features[21] = 1 if '.com' in clean_hostname.lower() else 0  # nb_com
        features[22] = 1 if '//' in url[7:] else 0  # nb_dslash
        features[23] = 1 if 'http' in path.lower() else 0  # http_in_path
        features[24] = 1 if url.startswith('https') else 0  # https_token
        
        # Ratio features
        features[25] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0  # ratio_digits_url
        features[26] = sum(c.isdigit() for c in clean_hostname) / len(clean_hostname) if len(clean_hostname) > 0 else 0  # ratio_digits_host
        
        # Domain features
        features[27] = 1 if clean_hostname.startswith('xn--') else 0  # punycode
        features[28] = 1 if parsed_url.port else 0  # port
        features[29] = 1 if any(tld in path for tld in ['.com', '.org', '.net']) else 0  # tld_in_path
        features[30] = 1 if any(tld in clean_hostname for tld in ['.com', '.org', '.net']) else 0  # tld_in_subdomain
        features[31] = 1 if clean_hostname.count('.') > 2 else 0  # abnormal_subdomain
        features[32] = len(clean_hostname.split('.')) - 1 if clean_hostname else 0  # nb_subdomains
        features[33] = 1 if '-' in clean_hostname else 0  # prefix_suffix
        
        # Suspicious features
        features[34] = 1 if any(s in url for s in ['bit.ly', 't.co', 'goo.gl']) else 0  # shortening_service
        features[35] = 1 if path.endswith(('.php', '.html', '.asp')) else 0  # path_extension
        features[36] = 1 if any(tld in clean_hostname for tld in ['.tk', '.ga', '.cf']) else 0  # suspecious_tld
        
        # Word-based features
        features[37] = len(url.split('/')) + len(url.split('.')) - 1  # length_words_raw
        features[38] = sum(url[i] == url[i-1] for i in range(1, len(url)))  # char_repeat
        
        # Word length features
        words = [w for w in url.split('/') if w]
        features[39] = min(len(word) for word in words) if words else 0  # shortest_words_raw
        features[40] = min(len(word) for word in clean_hostname.split('.') if word) if clean_hostname else 0  # shortest_word_host
        features[41] = min(len(word) for word in path.split('/') if word) if path else 0  # shortest_word_path
        features[42] = max(len(word) for word in words) if words else 0  # longest_words_raw
        features[43] = max(len(word) for word in clean_hostname.split('.') if word) if clean_hostname else 0  # longest_word_host
        features[44] = max(len(word) for word in path.split('/') if word) if path else 0  # longest_word_path
        
        # Average word length features
        features[45] = sum(len(word) for word in words) / len(words) if words else 0  # avg_words_raw
        features[46] = sum(len(word) for word in clean_hostname.split('.')) / len(clean_hostname.split('.')) if clean_hostname else 0  # avg_word_host
        features[47] = sum(len(word) for word in path.split('/')) / len(path.split('/')) if path else 0  # avg_word_path
        
        # Phishing hints
        features[48] = sum(1 for hint in ['login', 'secure', 'account', 'bank', 'paypal', 'amazon', 'ebay'] if hint in url.lower())  # phish_hints

    except Exception as e:
        print(f"Error extracting features: {e}")
        # Return zeros if feature extraction fails
        return [0] * num_features

    return features 