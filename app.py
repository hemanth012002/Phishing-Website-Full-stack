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
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# Update feature names to match the actual dataset columns (excluding 'url' and 'status')
FEATURE_NAMES = [
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at',
    'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde',
    'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn',
    'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path',
    'https_token', 'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port',
    'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains',
    'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension',
    'nb_redirection', 'nb_external_redirection', 'length_words_raw', 'char_repeat',
    'shortest_words_raw', 'shortest_word_host', 'shortest_word_path', 'longest_words_raw',
    'longest_word_host', 'longest_word_path', 'avg_words_raw', 'avg_word_host',
    'avg_word_path', 'phish_hints', 'domain_in_brand', 'brand_in_subdomain',
    'brand_in_path', 'suspecious_tld', 'statistical_report', 'nb_hyperlinks',
    'ratio_intHyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks', 'nb_extCSS',
    'ratio_intRedirection', 'ratio_extRedirection', 'ratio_intErrors', 'ratio_extErrors',
    'login_form', 'external_favicon', 'links_in_tags', 'submit_email', 'ratio_intMedia',
    'ratio_extMedia', 'sfh', 'iframe', 'popup_window', 'safe_anchor', 'onmouseover',
    'right_clic', 'empty_title', 'domain_in_title', 'domain_with_copyright',
    'whois_registered_domain', 'domain_registration_length', 'domain_age', 'web_traffic',
    'dns_record', 'google_index', 'page_rank'
]

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))  # For session management

# MongoDB setup
client = MongoClient(os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/'))
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

# Updated PyTorch model with Encoder-Decoder architecture
class PhishingDetector(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(PhishingDetector, self).__init__()
        
        # Encoder: Bidirectional GRU
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder: Bidirectional GRU
        self.decoder = nn.GRU(
            input_size=hidden_size * 2,  # *2 because bidirectional
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism - adjust for bidirectional encoder output
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional encoder output
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature extraction layers - adjust input size for bidirectional decoder output
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # hidden_size for bidirectional decoder
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization - fix dimensions for bidirectional outputs
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)  # For bidirectional encoder output
        self.layer_norm2 = nn.LayerNorm(hidden_size)  # For bidirectional decoder output
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder: Process input features
        encoder_output, encoder_hidden = self.encoder(x)
        
        # Apply layer normalization to the last dimension (hidden_size * 2 for bidirectional)
        encoder_output = self.layer_norm1(encoder_output)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(
            encoder_output, encoder_output, encoder_output
        )
        
        # Decoder: Process encoded features
        decoder_output, decoder_hidden = self.decoder(attn_output)
        
        # Apply layer normalization to the last dimension (hidden_size for bidirectional decoder)
        decoder_output = self.layer_norm2(decoder_output)
        
        # Take the last output from decoder
        last_output = decoder_output[:, -1, :]
        
        # Feature extraction
        features = self.feature_extractor(last_output)
        
        # Classification
        output = self.classifier(features)
        
        return output

def load_or_train_model():
    global model, scaler
    
    try:
        # Try to load pre-trained model and scaler
        model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print("Loading pre-trained model and scaler...")
            model = PhishingDetector(len(FEATURE_NAMES))
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()
            
            scaler = joblib.load(scaler_path)
            print("Successfully loaded pre-trained model and scaler")
            return True
        
        print("No pre-trained model found. Training new model with encoder-decoder architecture...")
        # Initialize model with new architecture
        model = PhishingDetector(len(FEATURE_NAMES))
        model.to(device)
        
        # Load and preprocess the dataset
        print("Loading dataset...")
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_phishing.csv')
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"Successfully loaded dataset with {len(df)} rows")
            print("\nDataset columns:", df.columns.tolist())
            
            # Print initial data types
            print("\nInitial data types:")
            print(df.dtypes)
            
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
            
        try:
            # Separate features and target
            print("\nSeparating features and target...")
            X = df.drop(['status', 'url'], axis=1, errors='ignore')  # Drop non-feature columns
            y = df['status']
            
            # Convert target variable to numeric
            print("\nConverting target variable to numeric...")
            y = pd.to_numeric(y, errors='coerce')
            y = y.fillna(0)  # Fill any NaN values with 0
            print("Target variable type after conversion:", y.dtype)
            
            # Ensure the dataset has the expected features
            print("\nChecking for required features...")
            missing_features = [feature for feature in FEATURE_NAMES if feature not in X.columns]
            if missing_features:
                print(f"Error: Dataset missing required features: {missing_features}")
                print("Please ensure your dataset contains all required features")
                return False
            
            # Reorder columns to match FEATURE_NAMES
            X = X[FEATURE_NAMES]
            
            # Print data types before conversion
            print("\nData types before conversion:")
            print(X.dtypes)
            
            # Convert all features to numeric type
            print("\nConverting features to numeric type...")
            for col in X.columns:
                # First try to convert to float
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception as e:
                    print(f"Error converting column {col}: {str(e)}")
                    # If conversion fails, try to convert to boolean first
                    try:
                        X[col] = X[col].astype(bool).astype(int)
                    except:
                        print(f"Failed to convert column {col} to numeric")
                        return False
            
            # Fill any NaN values with 0
            X = X.fillna(0)
            
            # Print data types after conversion
            print("\nData types after conversion:")
            print(X.dtypes)
            
            # Verify all columns are numeric
            non_numeric_cols = X.select_dtypes(include=['object']).columns
            if len(non_numeric_cols) > 0:
                print(f"Error: Non-numeric columns found after conversion: {non_numeric_cols}")
                return False
            
            # Convert to numpy array and verify type
            X_np = X.to_numpy()
            print("\nNumpy array type:", X_np.dtype)
            
            # Scale the features
            print("\nScaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_np)
            print("Successfully scaled features")
            
            # Save the scaler
            joblib.dump(scaler, scaler_path)
            print(f"Saved scaler to {scaler_path}")
            
            # Split the data
            print("\nSplitting data into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Convert to PyTorch tensors
            print("\nConverting data to PyTorch tensors...")
            print("X_train type before conversion:", type(X_train))
            print("y_train type before conversion:", type(y_train))
            print("y_train dtype before conversion:", y_train.dtype if hasattr(y_train, 'dtype') else 'unknown')
            
            # Convert pandas Series to numpy array
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            
            # Ensure numpy arrays are float32
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)
            
            # Convert to PyTorch tensors
            X_train = torch.from_numpy(X_train).reshape(-1, 1, len(FEATURE_NAMES))
            y_train = torch.from_numpy(y_train).reshape(-1, 1)
            X_test = torch.from_numpy(X_test).reshape(-1, 1, len(FEATURE_NAMES))
            y_test = torch.from_numpy(y_test).reshape(-1, 1)
            
            print("X_train tensor type:", X_train.dtype)
            print("y_train tensor type:", y_train.dtype)
            
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
                        # Save the best model
                        torch.save(model.state_dict(), model_path)
                        print(f"Saved best model at epoch {epoch + 1}")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # Load the best model
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            print("Model training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"Error in load_or_train_model: {str(e)}")
        return False

# Initialize model when app starts
if not load_or_train_model():
    print("WARNING: Model initialization failed. The application may not function correctly.")
    print("Please check the logs for more information.")

def extract_url_features(url, feature_names):
    """Extract comprehensive features from a URL for phishing detection."""
    features = [0] * len(feature_names)  # Initialize with zeros

    try:
        # Parse the URL
        parsed_url = urlparse(url if url.startswith('http') else 'http://' + url)
        hostname = parsed_url.hostname or ''
        path = parsed_url.path
        query = parsed_url.query
        netloc = parsed_url.netloc

        # Basic URL features (0-23)
        features[0] = len(url)  # length_url
        features[1] = len(hostname)  # length_hostname
        features[2] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0  # ip
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
        features[21] = 1 if '.com' in hostname.lower() else 0  # nb_com
        features[22] = 1 if '//' in url[7:] else 0  # nb_dslash
        features[23] = 1 if 'http' in path.lower() else 0  # http_in_path
        features[24] = 1 if url.startswith('https') else 0  # https_token
        
        # Ratio features (25-26)
        features[25] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0  # ratio_digits_url
        features[26] = sum(c.isdigit() for c in hostname) / len(hostname) if len(hostname) > 0 else 0  # ratio_digits_host
        
        # Domain features (27-32)
        features[27] = 1 if hostname.startswith('xn--') else 0  # punycode
        features[28] = 1 if parsed_url.port else 0  # port
        features[29] = 1 if any(tld in path for tld in ['.com', '.org', '.net']) else 0  # tld_in_path
        features[30] = 1 if any(tld in hostname for tld in ['.com', '.org', '.net']) else 0  # tld_in_subdomain
        features[31] = 1 if hostname.count('.') > 2 else 0  # abnormal_subdomain
        features[32] = len(hostname.split('.')) - 1 if hostname else 0  # nb_subdomains
        features[33] = 1 if '-' in hostname else 0  # prefix_suffix
        
        # Additional features (34-87)
        features[34] = 0  # random_domain - simplified
        features[35] = 1 if any(s in url for s in ['bit.ly', 't.co', 'goo.gl']) else 0  # shortening_service
        features[36] = 1 if path.endswith(('.php', '.html', '.asp')) else 0  # path_extension
        features[37] = url.count('redirect') + url.count('goto')  # nb_redirection
        features[38] = 1 if any(ext in hostname for ext in ['.tk', '.ga', '.cf', '.ml']) else 0  # nb_external_redirection
        
        # Word-based features (39-50)
        features[39] = len(url.split('/')) + len(url.split('.')) - 1  # length_words_raw
        features[40] = sum(url[i] == url[i-1] for i in range(1, len(url)))  # char_repeat
        
        # Word length features
        words = [w for w in url.split('/') if w]
        features[41] = min(len(word) for word in words) if words else 0  # shortest_words_raw
        features[42] = min(len(word) for word in hostname.split('.') if word) if hostname else 0  # shortest_word_host
        features[43] = min(len(word) for word in path.split('/') if word) if path else 0  # shortest_word_path
        features[44] = max(len(word) for word in words) if words else 0  # longest_words_raw
        features[45] = max(len(word) for word in hostname.split('.') if word) if hostname else 0  # longest_word_host
        features[46] = max(len(word) for word in path.split('/') if word) if path else 0  # longest_word_path
        
        # Average word length features
        features[47] = sum(len(word) for word in words) / len(words) if words else 0  # avg_words_raw
        features[48] = sum(len(word) for word in hostname.split('.')) / len(hostname.split('.')) if hostname else 0  # avg_word_host
        features[49] = sum(len(word) for word in path.split('/')) / len(path.split('/')) if path else 0  # avg_word_path
        
        # Phishing hints (50)
        features[50] = sum(1 for hint in ['login', 'secure', 'account', 'bank', 'paypal', 'amazon', 'ebay'] if hint in url.lower())  # phish_hints
        
        # Brand-related features (51-53)
        features[51] = 0  # domain_in_brand - simplified
        features[52] = 0  # brand_in_subdomain - simplified
        features[53] = 0  # brand_in_path - simplified
        features[54] = 1 if any(tld in hostname for tld in ['.tk', '.ga', '.cf']) else 0  # suspecious_tld
        
        # Statistical and analysis features (55-87)
        features[55] = 0  # statistical_report - simplified
        features[56] = url.count('href') + url.count('link')  # nb_hyperlinks
        features[57] = 0.5  # ratio_intHyperlinks - simplified
        features[58] = 0.3  # ratio_extHyperlinks - simplified
        features[59] = 0.2  # ratio_nullHyperlinks - simplified
        features[60] = url.count('css') + url.count('style')  # nb_extCSS
        features[61] = 0.1  # ratio_intRedirection - simplified
        features[62] = 0.1  # ratio_extRedirection - simplified
        features[63] = 0.05  # ratio_intErrors - simplified
        features[64] = 0.05  # ratio_extErrors - simplified
        features[65] = 1 if 'login' in url.lower() or 'form' in url.lower() else 0  # login_form
        features[66] = 1 if 'favicon' in url.lower() else 0  # external_favicon
        features[67] = url.count('a') + url.count('link')  # links_in_tags
        features[68] = 1 if 'mailto:' in url.lower() else 0  # submit_email
        features[69] = 0.1  # ratio_intMedia - simplified
        features[70] = 0.1  # ratio_extMedia - simplified
        features[71] = 0  # sfh - simplified
        features[72] = 1 if 'iframe' in url.lower() else 0  # iframe
        features[73] = 1 if 'popup' in url.lower() else 0  # popup_window
        features[74] = 1 if 'anchor' in url.lower() else 0  # safe_anchor
        features[75] = 1 if 'mouseover' in url.lower() else 0  # onmouseover
        features[76] = 1 if 'rightclick' in url.lower() else 0  # right_clic
        features[77] = 0  # empty_title - simplified
        features[78] = 1 if hostname in url.lower() else 0  # domain_in_title
        features[79] = 0  # domain_with_copyright - simplified
        features[80] = 1  # whois_registered_domain - simplified
        features[81] = len(hostname) if hostname else 0  # domain_registration_length
        features[82] = 1  # domain_age - simplified
        features[83] = 1  # web_traffic - simplified
        features[84] = 1  # dns_record - simplified
        features[85] = 1  # google_index - simplified
        features[86] = 1  # page_rank - simplified

    except Exception as e:
        print(f"Error extracting features: {e}")
        return [0] * len(feature_names)

    return features

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
        
        user = users.find_one({'email': email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            user_obj = User(user)
            login_user(user_obj)
            return redirect(url_for('check'))
        
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page."""
    try:
        # Get user's URL check history
        history = list(url_history.find(
            {'user_id': current_user.id}
        ).sort('timestamp', -1))
        
        # Count different types of URLs
        safe_count = url_history.count_documents({
            'user_id': current_user.id,
            'result': 'safe'
        })
        
        suspicious_count = url_history.count_documents({
            'user_id': current_user.id,
            'result': 'suspicious'
        })
        
        phishing_count = url_history.count_documents({
            'user_id': current_user.id,
            'result': 'suspicious',
            'analysis.domain_trust': 'suspicious'
        })
        
        return render_template('profile.html',
                             user=current_user,
                             history=history,
                             safe_count=safe_count,
                             suspicious_count=suspicious_count,
                             phishing_count=phishing_count)
    except Exception as e:
        print(f"Error loading profile: {str(e)}")
        return render_template('profile.html',
                             user=current_user,
                             history=[],
                             safe_count=0,
                             suspicious_count=0,
                             phishing_count=0)

@app.route('/check')
@login_required
def check():
    url = request.args.get('url', '')
    return render_template('check.html', url=url)

@app.route('/history')
@login_required
def history():
    # Get user's check history from MongoDB
    history = list(url_history.find({'user_id': current_user.id}).sort('timestamp', -1))
    return render_template('history.html', history=history)

@app.route('/check-url', methods=['POST'])
@login_required
def check_url():
    """Check if a URL is safe or suspicious."""
    try:
        url = request.json.get('url', '').strip()
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
            
        print(f"\nChecking URL: {url}")
        
        # Extract features
        features = extract_url_features(url, FEATURE_NAMES)
        
        # Get detailed analysis
        analysis = analyze_url_characteristics(url, features)
        
        # Convert features to tensor
        features_tensor = torch.FloatTensor(features).reshape(1, 1, -1).to(device)
        
        # Get model prediction
        with torch.no_grad():
            prediction = model(features_tensor)
            probability = prediction.item()
        
        # Determine result based on both model prediction and analysis
        is_suspicious = False
        suspicious_reasons = []
        
        # Check for typosquatting first
        parsed_url = urlparse(url if url.startswith('http') else 'http://' + url)
        domain = parsed_url.hostname.lower() if parsed_url.hostname else ''
        
        # List of known legitimate domains and their variations
        legitimate_domains = {
            'google.com', 'google.co.in', 'google.co.uk', 'google.de', 'google.fr',
            'microsoft.com', 'microsoft.co.in', 'microsoft.co.uk',
            'facebook.com', 'facebook.co.in',
            'amazon.com', 'amazon.co.in', 'amazon.co.uk', 'amazon.de',
            'apple.com', 'apple.co.in', 'apple.co.uk',
            'paypal.com', 'paypal.co.in',
            'netflix.com', 'netflix.co.in',
            'twitter.com',
            'linkedin.com',
            'instagram.com',
            'youtube.com',
            'github.com',
            'yahoo.com',
            'outlook.com',
            'hotmail.com',
            'gmail.com',
            'reddit.com',
            'wikipedia.org',
            'bing.com',
            'mozilla.org',
            'opera.com',
            'brave.com',
            'duckduckgo.com',
            'cloudflare.com',
            'wordpress.com',
            'medium.com',
            'quora.com',
            'pinterest.com',
            'spotify.com',
            'discord.com',
            'slack.com',
            'zoom.us',
            'teams.microsoft.com',
            'office.com',
            'onedrive.live.com',
            'dropbox.com',
            'box.com',
            'adobe.com',
            'adobe.co.in',
            'adobe.co.uk'
        }
        
        # If it's a known legitimate domain, mark as safe
        if domain in legitimate_domains:
            result = "safe"
            color = "green"
            analysis['domain_trust'] = 'trusted'
            analysis['suspicious_patterns'] = []
            analysis['recommendations'] = ["This is a known legitimate domain"]
        else:
            # Common typosquatting patterns
            common_typos = {
                'microsoft': ['micr0soft', 'micr0s0ft', 'm1cr0s0ft', 'm1crosoft', 'micrsoft', 'microsoftt'],
                'google': ['g00gle', 'go0gle', 'g0ogle', 'googl3', 'goog1e'],
                'facebook': ['faceb00k', 'faceb0ok', 'f4cebook', 'facebok'],
                'amazon': ['amaz0n', 'amaz0n', 'amazn', 'amaz0n'],
                'apple': ['app1e', 'appl3', 'appl3', 'appl'],
                'paypal': ['payp4l', 'payp4l', 'paypal1', 'paypall']
            }
            
            # Check for typosquatting
            for legit_domain, typos in common_typos.items():
                if any(typo in domain for typo in typos):
                    is_suspicious = True
                    suspicious_reasons.append(f"Domain appears to be a typosquatting attempt of {legit_domain}")
                    break
            
            # Check for suspicious character substitutions
            substitutions = {
                'o': ['0', 'O'],
                'i': ['1', 'l', 'I'],
                'a': ['4', '@'],
                'e': ['3'],
                's': ['5', '$'],
                't': ['7']
            }
            
            for char, replacements in substitutions.items():
                if any(replacement in domain for replacement in replacements):
                    is_suspicious = True
                    suspicious_reasons.append("Contains suspicious character substitutions")
                    break
            
            # Check other suspicious patterns
            if features[34]:  # shortening_service
                is_suspicious = True
                suspicious_reasons.append("Uses URL shortening service")
            if features[36]:  # suspecious_tld
                is_suspicious = True
                suspicious_reasons.append("Uses suspicious TLD")
            if features[33]:  # prefix_suffix
                is_suspicious = True
                suspicious_reasons.append("Contains hyphens in domain")
            if features[31]:  # abnormal_subdomain
                is_suspicious = True
                suspicious_reasons.append("Has abnormal subdomain structure")
                
            # Final decision
            if is_suspicious:
                result = "suspicious"
                color = "red"
                analysis['domain_trust'] = 'suspicious'
                analysis['suspicious_patterns'].extend(suspicious_reasons)
                analysis['recommendations'] = [
                    "WARNING: This URL appears to be a phishing attempt",
                    "The domain name is suspiciously similar to a known brand",
                    "Do not enter any personal information on this site"
                ]
            else:
                result = "safe" if probability < 0.5 else "suspicious"
                color = "green" if result == "safe" else "red"
            
        # Save to history
        check_data = {
            'user_id': current_user.id,
            'url': url,
            'result': result,
            'analysis': analysis,
            'timestamp': datetime.utcnow()
        }
        
        url_history.insert_one(check_data)
        print("Successfully saved to history")
        
        # Update user's check count
        users.update_one(
            {'_id': current_user.id},
            {'$inc': {'check_count': 1}}
        )
        
        return jsonify({
            'result': result,
            'color': color,
            'analysis': analysis,
            'probability': probability
        })
        
    except Exception as e:
        print(f"Error checking URL: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to check URL'}), 500

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
        domain = hostname.lower()
        
        # List of trusted domains and educational institutions
        trusted_domains = [
            'google.com', 'google.co', 'google.', 'youtube.com', 'youtube.co',
            'facebook.com', 'facebook.co', 'amazon.com', 'amazon.co',
            'microsoft.com', 'microsoft.co', 'apple.com', 'apple.co',
            'github.com', 'github.co', 'linkedin.com', 'linkedin.co',
            'twitter.com', 'twitter.co', 'instagram.com', 'instagram.co'
        ]
        
        # Check for typosquatting
        def check_typosquatting(domain):
            common_typos = {
                'microsoft': ['micr0soft', 'micr0s0ft', 'm1cr0s0ft', 'm1crosoft', 'micrsoft', 'microsoftt'],
                'google': ['g00gle', 'go0gle', 'g0ogle', 'googl3', 'goog1e'],
                'facebook': ['faceb00k', 'faceb0ok', 'f4cebook', 'facebok'],
                'amazon': ['amaz0n', 'amaz0n', 'amazn', 'amaz0n'],
                'apple': ['app1e', 'appl3', 'appl3', 'appl'],
                'paypal': ['payp4l', 'payp4l', 'paypal1', 'paypall']
            }
            
            for legit_domain, typos in common_typos.items():
                if legit_domain in domain:
                    return False  # It's the legitimate domain
                for typo in typos:
                    if typo in domain:
                        return True  # It's a typosquatting attempt
            return False
        
        # Check for suspicious character substitutions
        def check_character_substitutions(domain):
            substitutions = {
                'o': ['0', 'O'],
                'i': ['1', 'l', 'I'],
                'a': ['4', '@'],
                'e': ['3'],
                's': ['5', '$'],
                't': ['7']
            }
            
            for char, replacements in substitutions.items():
                for replacement in replacements:
                    if replacement in domain:
                        return True
            return False
        
        # Check if it's a trusted domain
        is_trusted_domain = any(trusted in domain for trusted in trusted_domains)
        
        # Check for educational institutions
        is_edu = '.edu' in domain or any(edu in domain for edu in ['university', 'college', 'school', 'institute', 'academy'])
        
        # Check for typosquatting
        is_typosquatting = check_typosquatting(domain)
        
        # Check for character substitutions
        has_suspicious_chars = check_character_substitutions(domain)
        
        # Check for trust indicators
        if url.startswith('https'):
            analysis['trust_indicators'].append('Uses HTTPS encryption')
        if hostname.startswith('www.'):
            analysis['trust_indicators'].append('Has www subdomain')
        if any(tld in hostname for tld in ['.com', '.org', '.net', '.edu', '.gov']):
            analysis['trust_indicators'].append('Uses common TLD')
        if is_trusted_domain:
            analysis['trust_indicators'].append('Known trusted domain')
        if is_edu:
            analysis['trust_indicators'].append('Educational institution domain')
            
        # Check for suspicious patterns
        if is_typosquatting:
            analysis['suspicious_patterns'].append('Possible typosquatting detected (similar to known brand)')
        if has_suspicious_chars:
            analysis['suspicious_patterns'].append('Contains suspicious character substitutions')
        if features[34]:  # shortening_service
            analysis['suspicious_patterns'].append('Uses URL shortening service')
        if features[36]:  # suspecious_tld
            analysis['suspicious_patterns'].append('Uses suspicious TLD')
        if features[33]:  # prefix_suffix
            analysis['suspicious_patterns'].append('Contains hyphens in domain')
        if features[31]:  # abnormal_subdomain
            analysis['suspicious_patterns'].append('Has abnormal subdomain structure')
            
        # Generate recommendations
        if is_typosquatting or has_suspicious_chars:
            analysis['recommendations'].append('WARNING: This URL appears to be a phishing attempt')
            analysis['recommendations'].append('The domain name is suspiciously similar to a known brand')
            analysis['domain_trust'] = 'suspicious'
        elif is_trusted_domain or is_edu:
            analysis['recommendations'].append('URL appears to be from a trusted source')
            analysis['domain_trust'] = 'trusted'
        elif analysis['suspicious_patterns']:
            analysis['recommendations'].append('Exercise caution when visiting this URL')
            if not url.startswith('https'):
                analysis['recommendations'].append('Consider using HTTPS version if available')
        else:
            analysis['recommendations'].append('URL appears to be safe')
            
        # Determine overall domain trust
        if is_typosquatting or has_suspicious_chars:
            analysis['domain_trust'] = 'suspicious'
        elif is_trusted_domain or is_edu:
            analysis['domain_trust'] = 'trusted'
        elif len(analysis['trust_indicators']) > len(analysis['suspicious_patterns']):
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
        check_id = ObjectId(check_id)
        
        # Find and delete the check, ensuring it belongs to the current user
        result = url_history.delete_one({
            '_id': check_id,
            'user_id': current_user.id
        })
        
        if result.deleted_count > 0:
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Check not found or unauthorized'}), 404
            
    except Exception as e:
        print(f"Error deleting check: {str(e)}")
        return jsonify({'error': 'Failed to delete check'}), 500

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
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_phishing.csv')
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
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_phishing.csv')
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
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_phishing.csv')
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
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_phishing.csv')
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
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False) 