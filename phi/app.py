from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from phishing_utils import extract_url_features
import joblib

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['phishing_detector']
users = db['users']
url_history = db['url_history']

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Global variables for model and scaler
model = None
scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the PyTorch model
class PhishingDetector(nn.Module):
    def __init__(self, input_size):
        super(PhishingDetector, self).__init__()
        self.gru1 = nn.GRU(input_size, 64, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.gru2 = nn.GRU(128, 32, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 16)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First GRU layer
        gru1_out, _ = self.gru1(x)
        gru1_out = self.dropout1(gru1_out)
        
        # Second GRU layer
        gru2_out, _ = self.gru2(gru1_out)
        gru2_out = self.dropout2(gru2_out)
        
        # Take the last output of the second GRU
        last_hidden = gru2_out[:, -1, :]
        
        # Fully connected layers
        fc1_out = self.relu(self.fc1(last_hidden))
        fc1_out = self.dropout3(fc1_out)
        out = self.sigmoid(self.fc2(fc1_out))
        
        return out

# Predefined feature names that match the model's expected input
FEATURE_NAMES = [
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and',
    'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon',
    'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash',
    'http_in_path', 'https_token', 'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port',
    'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix',
    'shortening_service', 'path_extension', 'suspecious_tld', 'length_words_raw', 'char_repeat',
    'shortest_words_raw', 'shortest_word_host', 'shortest_word_path', 'longest_words_raw',
    'longest_word_host', 'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path',
    'phish_hints'
]

def load_or_train_model():
    global model, scaler
    
    try:
        # Initialize model and train directly
        print("Initializing and training model...")
        model = PhishingDetector(len(FEATURE_NAMES))
        model.to(device)
        
        # Load and preprocess the dataset
        print("Loading dataset...")
        dataset_path = 'dataset_phishing.csv'  # Using relative path
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"Successfully loaded dataset with {len(df)} rows")
            print("\nDataset columns:", df.columns.tolist())
        except FileNotFoundError:
            print(f"Error: Dataset not found at {dataset_path}")
            print("Please ensure the dataset file exists in the project directory")
            return False
        except pd.errors.EmptyDataError:
            print("Error: The dataset file is empty")
            return False
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
        
        # Verify dataset has required columns
        if 'status' not in df.columns:
            print("Error: Dataset missing 'status' column")
            print("Available columns:", df.columns.tolist())
            return False
            
        # Print dataset info for debugging
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        
        try:
            # Separate features and target
            print("\nSeparating features and target...")
            X = df.drop('status', axis=1)
            y = df['status']
            print("Successfully separated features and target")
            print("Feature columns:", X.columns.tolist())
            print("Target shape:", y.shape)
        except Exception as e:
            print(f"Error during feature separation: {str(e)}")
            return False
        
        # Ensure the dataset has the expected features
        print("\nChecking for required features...")
        missing_features = [feature for feature in FEATURE_NAMES if feature not in X.columns]
        if missing_features:
            print(f"Error: Dataset missing required features: {missing_features}")
            print("Please ensure your dataset contains all required features")
            return False
        
        try:
            # Reorder columns to match FEATURE_NAMES
            X = X[FEATURE_NAMES]
            
            # Scale the features
            print("\nScaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            print("Successfully scaled features")
            
            # Save the scaler
            joblib.dump(scaler, 'scaler.pkl')
            print("Saved scaler to scaler.pkl")
            
            # Split the data
            print("\nSplitting data into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            print(f"Training set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")
            
            # Convert to PyTorch tensors
            print("\nConverting data to PyTorch tensors...")
            X_train = torch.FloatTensor(X_train).reshape(-1, 1, len(FEATURE_NAMES))
            y_train = torch.FloatTensor(y_train.values).reshape(-1, 1)
            X_test = torch.FloatTensor(X_test).reshape(-1, 1, len(FEATURE_NAMES))
            y_test = torch.FloatTensor(y_test.values).reshape(-1, 1)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            print("\nTraining model...")
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            for epoch in range(50):
                model.train()
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_X = X_test.to(device)
                    val_y = y_test.to(device)
                    val_outputs = model(val_X)
                    val_loss = criterion(val_outputs, val_y)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # Evaluate the model
            print("\nEvaluating model...")
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test.to(device))
                test_predictions = (test_outputs > 0.5).float()
                test_accuracy = (test_predictions == y_test.to(device)).float().mean()
                print(f"Test accuracy: {test_accuracy:.4f}")
            
            # Generate and save classification report
            y_pred = test_predictions.cpu().numpy()
            report = classification_report(y_test.numpy(), y_pred, output_dict=True)
            print("\nClassification Report:")
            print(classification_report(y_test.numpy(), y_pred))
            
            # Save the report
            with open('model_report.pkl', 'wb') as f:
                pickle.dump(report, f)
            
            print("Model training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        return False

# Initialize model when app starts
if not load_or_train_model():
    print("WARNING: Model initialization failed. The application may not function correctly.")
    print("Please check the logs for more information.")

def extract_url_features(url, feature_names):
    """Extract comprehensive features from a URL for phishing detection."""
    features = {name: 0 for name in feature_names}

    try:
        # Parse the URL
        parsed_url = urlparse(url if url.startswith('http') else 'http://' + url)
        hostname = parsed_url.hostname or ''
        path = parsed_url.path
        query = parsed_url.query
        netloc = parsed_url.netloc

        # Basic URL features
        features['length_url'] = len(url)
        features['length_hostname'] = len(hostname)
        features['ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0
        features['nb_dots'] = url.count('.')
        features['nb_hyphens'] = url.count('-')
        features['nb_at'] = url.count('@')
        features['nb_qm'] = url.count('?')
        features['nb_and'] = url.count('&')
        features['nb_or'] = url.count('|')
        features['nb_eq'] = url.count('=')
        features['nb_underscore'] = url.count('_')
        features['nb_tilde'] = url.count('~')
        features['nb_percent'] = url.count('%')
        features['nb_slash'] = url.count('/')
        features['nb_star'] = url.count('*')
        features['nb_colon'] = url.count(':')
        features['nb_comma'] = url.count(',')
        features['nb_semicolumn'] = url.count(';')
        features['nb_dollar'] = url.count('$')
        features['nb_space'] = url.count(' ')
        features['nb_www'] = 1 if 'www' in hostname.lower() else 0
        features['nb_com'] = 1 if '.com' in hostname.lower() else 0
        features['nb_dslash'] = 1 if '//' in url[7:] else 0
        features['http_in_path'] = 1 if 'http' in path.lower() else 0
        features['https_token'] = 1 if url.startswith('https') else 0

        # Ratio features
        features['ratio_digits_url'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
        features['ratio_digits_host'] = sum(c.isdigit() for c in hostname) / len(hostname) if len(hostname) > 0 else 0

        # Domain features
        features['punycode'] = 1 if hostname.startswith('xn--') else 0
        features['port'] = 1 if parsed_url.port else 0
        features['tld_in_path'] = 1 if any(tld in path for tld in ['.com', '.org', '.net']) else 0
        features['tld_in_subdomain'] = 1 if any(tld in hostname for tld in ['.com', '.org', '.net']) else 0
        features['abnormal_subdomain'] = 1 if hostname.count('.') > 2 else 0
        features['nb_subdomains'] = len(hostname.split('.')) - 1 if hostname else 0
        features['prefix_suffix'] = 1 if '-' in hostname else 0

        # Suspicious features
        features['shortening_service'] = 1 if any(s in url for s in ['bit.ly', 't.co', 'goo.gl']) else 0
        features['path_extension'] = 1 if path.endswith(('.php', '.html', '.asp')) else 0
        features['suspecious_tld'] = 1 if any(tld in hostname for tld in ['.tk', '.ga', '.cf']) else 0

        # Word-based features
        features['length_words_raw'] = len(url.split('/')) + len(url.split('.')) - 1
        features['char_repeat'] = sum(url[i] == url[i-1] for i in range(1, len(url)))
        features['shortest_words_raw'] = min(len(word) for word in url.split('/') if word) if url.split('/') else 0
        features['shortest_word_host'] = min(len(word) for word in hostname.split('.') if word) if hostname else 0
        features['shortest_word_path'] = min(len(word) for word in path.split('/') if word) if path else 0
        features['longest_words_raw'] = max(len(word) for word in url.split('/') if word) if url.split('/') else 0
        features['longest_word_host'] = max(len(word) for word in hostname.split('.') if word) if hostname else 0
        features['longest_word_path'] = max(len(word) for word in path.split('/') if word) if path else 0
        features['avg_words_raw'] = sum(len(word) for word in url.split('/')) / len(url.split('/')) if url.split('/') else 0
        features['avg_word_host'] = sum(len(word) for word in hostname.split('.')) / len(hostname.split('.')) if hostname else 0
        features['avg_word_path'] = sum(len(word) for word in path.split('/')) / len(path.split('/')) if path else 0

        # Phishing hints
        features['phish_hints'] = sum(1 for hint in ['login', 'secure', 'account', 'bank', 'paypal', 'amazon', 'ebay'] if hint in url.lower())

        # Trust indicators
        features['is_common_domain'] = 1 if any(domain in hostname.lower() for domain in 
            ['google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com', 'netflix.com', 
             'youtube.com', 'twitter.com', 'linkedin.com', 'github.com', 'yahoo.com', 'bing.com']) else 0

        # Additional features (set to 0 as they require external services)
        external_features = [
            'random_domain', 'nb_redirection', 'nb_external_redirection', 'domain_in_brand',
            'brand_in_subdomain', 'brand_in_path', 'statistical_report', 'nb_hyperlinks',
            'ratio_intHyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks', 'nb_extCSS',
            'ratio_intRedirection', 'ratio_extRedirection', 'ratio_intErrors', 'ratio_extErrors',
            'login_form', 'external_favicon', 'links_in_tags', 'submit_email', 'ratio_intMedia',
            'ratio_extMedia', 'sfh', 'iframe', 'popup_window', 'safe_anchor', 'onmouseover',
            'right_clic', 'empty_title', 'domain_in_title', 'domain_with_copyright',
            'whois_registered_domain', 'domain_registration_length', 'domain_age', 'web_traffic',
            'dns_record', 'google_index', 'page_rank'
        ]
        for feature in external_features:
            if feature in features:
                features[feature] = 0

        return [features[name] for name in feature_names]

    except Exception as e:
        print(f"Error extracting features: {e}")
        return [0] * len(feature_names)

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.name = user_data['name']

@login_manager.user_loader
def load_user(user_id):
    user_data = users.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('history'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if users.find_one({'email': email}):
            flash('Email already registered')
            return redirect(url_for('register'))
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_id = users.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.utcnow()
        }).inserted_id
        
        user = User(users.find_one({'_id': user_id}))
        login_user(user)
        return redirect(url_for('history'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_data = users.find_one({'email': email})
        if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data['password']):
            user = User(user_data)
            login_user(user)
            return redirect(url_for('history'))
        
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    user_data = users.find_one({'_id': ObjectId(current_user.id)})
    return render_template('profile.html', user=user_data)

@app.route('/history')
@login_required
def history():
    user_history = list(url_history.find({'user_id': current_user.id}).sort('timestamp', -1))
    return render_template('history.html', history=user_history)

@app.route('/check-url', methods=['POST'])
@login_required
def check_url():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not initialized. Please try again later.'}), 503
    
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        # Extract features using predefined feature names
        features = extract_url_features(url, FEATURE_NAMES)
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Convert to PyTorch tensor and reshape for model input
        features_tensor = torch.FloatTensor(features_scaled).reshape(1, 1, -1).to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(features_tensor).item()
        
        # Use sophisticated classification approach
        parsed_url = urlparse(url if url.startswith('http') else 'http://' + url)
        hostname = parsed_url.hostname or ''
        
        # Check for common legitimate domains
        is_common_domain = any(domain in hostname.lower() for domain in 
            ['google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com', 'netflix.com', 
             'youtube.com', 'twitter.com', 'linkedin.com', 'github.com', 'yahoo.com', 'bing.com'])
        
        if is_common_domain:
            # For common domains, use a higher threshold for phishing
            if prediction > 0.9:
                result = 'Phishing'
                confidence = prediction
            else:
                result = 'Safe'
                confidence = 1 - prediction
        else:
            # For other domains, use standard thresholds
            if prediction < 0.3:
                result = 'Safe'
                confidence = 1 - prediction
            elif prediction > 0.7:
                result = 'Phishing'
                confidence = prediction
            else:
                result = 'Suspicious'
                confidence = max(prediction, 1 - prediction)
        
        # Analyze URL characteristics
        analysis = analyze_url_characteristics(url, features[0])
        
        # Add domain trust information
        if is_common_domain:
            analysis['domain_trust'] = 'trusted'
            analysis['trust_indicators'].append('Recognized legitimate domain')
        
        # Save to history
        url_history.insert_one({
            'user_id': current_user.id,
            'url': url,
            'result': result,
            'confidence': float(confidence),
            'analysis': analysis,
            'timestamp': datetime.utcnow()
        })
        
        return jsonify({
            'result': result,
            'confidence': float(confidence),
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_url_characteristics(url, features):
    """Analyze URL characteristics and return detailed analysis."""
    analysis = {
        'domain_trust': 'neutral',
        'suspicious_patterns': [],
        'trust_indicators': [],
        'recommendations': []
    }
    
    try:
        parsed_url = urlparse(url if url.startswith('http') else 'http://' + url)
        hostname = parsed_url.hostname or ''
        
        # Check for trust indicators
        if url.startswith('https'):
            analysis['trust_indicators'].append('Uses HTTPS encryption')
        if hostname.startswith('www.'):
            analysis['trust_indicators'].append('Has www subdomain')
        if any(tld in hostname for tld in ['.com', '.org', '.net', '.edu', '.gov']):
            analysis['trust_indicators'].append('Uses common TLD')
            
        # Check for suspicious patterns
        if features[34]:  # shortening_service
            analysis['suspicious_patterns'].append('Uses URL shortening service')
        if features[36]:  # suspicious_tld
            analysis['suspicious_patterns'].append('Uses suspicious TLD')
        if features[33]:  # prefix_suffix
            analysis['suspicious_patterns'].append('Contains hyphens in domain')
        if features[31]:  # abnormal_subdomain
            analysis['suspicious_patterns'].append('Has abnormal subdomain structure')
            
        # Generate recommendations
        if analysis['suspicious_patterns']:
            analysis['recommendations'].append('Exercise caution when visiting this URL')
            if not url.startswith('https'):
                analysis['recommendations'].append('Consider using HTTPS version if available')
        else:
            analysis['recommendations'].append('URL appears to follow standard patterns')
            
        # Determine overall domain trust
        if len(analysis['trust_indicators']) > len(analysis['suspicious_patterns']):
            analysis['domain_trust'] = 'trusted'
        elif len(analysis['suspicious_patterns']) > len(analysis['trust_indicators']):
            analysis['domain_trust'] = 'suspicious'
        else:
            analysis['domain_trust'] = 'neutral'
            
    except Exception as e:
        print(f"Error analyzing URL: {e}")
        
    return analysis

@app.route('/delete-check/<check_id>', methods=['POST'])
@login_required
def delete_check(check_id):
    try:
        # Convert string ID to ObjectId
        check_object_id = ObjectId(check_id)
        
        # Delete the check if it belongs to the current user
        result = url_history.delete_one({
            '_id': check_object_id,
            'user_id': current_user.id
        })
        
        if result.deleted_count == 0:
            return jsonify({'error': 'Check not found or unauthorized'}), 404
            
        return jsonify({'message': 'Check deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['POST'])
@login_required
def clear_history():
    try:
        # Delete all checks for the current user
        result = url_history.delete_many({'user_id': current_user.id})
        return jsonify({'message': f'Deleted {result.deleted_count} checks'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dataset-info')
def dataset_info():
    try:
        # Load the dataset
        dataset_path = r'C:\Users\nshem\Downloads\phi\phi\dataset_phishing.csv'
        df = pd.read_csv(dataset_path)
        
        # Get basic information
        info = {
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'first_few_rows': df.head().to_dict('records'),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'status_distribution': df['status'].value_counts().to_dict() if 'status' in df.columns else None
        }
        
        return jsonify({
            'success': True,
            'data': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/dataset-sample')
def dataset_sample():
    try:
        # Load the dataset
        dataset_path = r'C:\Users\nshem\Downloads\phi\phi\dataset_phishing.csv'
        df = pd.read_csv(dataset_path)
        
        # Get a sample of the data
        sample = df.sample(min(5, len(df))).to_dict('records')
        
        return jsonify({
            'success': True,
            'sample_data': sample
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/dataset-stats')
def dataset_stats():
    try:
        # Load the dataset
        dataset_path = r'C:\Users\nshem\Downloads\phi\phi\dataset_phishing.csv'
        df = pd.read_csv(dataset_path)
        
        # Calculate basic statistics
        stats = {
            'numeric_stats': df.describe().to_dict(),
            'column_info': {
                col: {
                    'unique_values': df[col].nunique(),
                    'missing_values': df[col].isnull().sum(),
                    'data_type': str(df[col].dtype)
                } for col in df.columns
            }
        }
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/view-data')
def view_data():
    try:
        dataset_path = r'C:\Users\nshem\Downloads\phi\phi\dataset_phishing.csv'
        df = pd.read_csv(dataset_path)
        
        # Convert DataFrame to HTML
        table_html = df.head(10).to_html(classes='table table-striped')
        
        return f'''
        <html>
            <head>
                <title>Dataset Viewer</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-5">
                    <h2>Dataset Information</h2>
                    <p>Total Rows: {len(df)}</p>
                    <p>Columns: {', '.join(df.columns)}</p>
                    <h3>First 10 Rows:</h3>
                    {table_html}
                </div>
            </body>
        </html>
        '''
    except Exception as e:
        return f'''
        <html>
            <body>
                <h2>Error</h2>
                <p>{str(e)}</p>
            </body>
        </html>
        '''

if __name__ == '__main__':
    app.run(debug=True) 