#!/usr/bin/env python3
"""
Deployment Check Script for Phishing URL Detector
This script verifies all requirements and functionality before deployment.
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_required_files():
    """Check if all required files exist."""
    print("\n🔍 Checking required files...")
    required_files = [
        'app.py',
        'requirements.txt',
        'render.yaml',
        'gunicorn.conf.py',
        'model.pth',
        'scaler.pkl',
        'dataset_phishing.csv',
        'phishing_utils.py',
        'templates/base.html',
        'templates/index.html',
        'templates/login.html',
        'templates/register.html',
        'templates/check.html',
        'templates/history.html',
        'templates/profile.html',
        'static/css/auth.css',
        'static/css/style.css',
        'static/css/landing.css'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_dependencies():
    """Check if all required dependencies can be imported."""
    print("\n🔍 Checking dependencies...")
    dependencies = [
        'flask',
        'flask_login',
        'pymongo',
        'bcrypt',
        'torch',
        'numpy',
        'pandas',
        'scikit_learn',
        'matplotlib',
        'seaborn',
        'joblib',
        'gunicorn',
        'python_dotenv'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            importlib.import_module(dep.replace('-', '_'))
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - NOT INSTALLED")
            missing_deps.append(dep)
    
    return len(missing_deps) == 0

def check_environment_variables():
    """Check environment variable configuration."""
    print("\n🔍 Checking environment variables...")
    
    # Check render.yaml configuration
    with open('render.yaml', 'r') as f:
        render_config = f.read()
    
    required_vars = ['PYTHON_VERSION', 'MONGODB_URI', 'SECRET_KEY', 'FLASK_ENV']
    missing_vars = []
    
    for var in required_vars:
        if var in render_config:
            print(f"✅ {var} - Configured in render.yaml")
        else:
            print(f"❌ {var} - Missing from render.yaml")
            missing_vars.append(var)
    
    return len(missing_vars) == 0

def check_flask_app():
    """Check if Flask app can be imported and initialized."""
    print("\n🔍 Checking Flask application...")
    try:
        # Temporarily set environment variables for testing
        os.environ['MONGODB_URI'] = 'mongodb://localhost:27017/'
        os.environ['SECRET_KEY'] = 'test-secret-key'
        os.environ['FLASK_ENV'] = 'testing'
        
        from app import app
        print("✅ Flask app imported successfully")
        
        # Check if app has required routes
        routes = ['/', '/login', '/register', '/check', '/history', '/profile', '/check-url']
        for route in routes:
            if route in [str(rule) for rule in app.url_map.iter_rules()]:
                print(f"✅ Route {route} - Found")
            else:
                print(f"❌ Route {route} - Missing")
        
        return True
    except Exception as e:
        print(f"❌ Flask app error: {str(e)}")
        return False

def check_model_files():
    """Check if model files are valid."""
    print("\n🔍 Checking model files...")
    
    try:
        import torch
        import joblib
        
        # Check model.pth
        if os.path.exists('model.pth'):
            model_size = os.path.getsize('model.pth') / (1024 * 1024)  # MB
            print(f"✅ model.pth - {model_size:.2f} MB")
        else:
            print("❌ model.pth - Missing")
            return False
        
        # Check scaler.pkl
        if os.path.exists('scaler.pkl'):
            scaler_size = os.path.getsize('scaler.pkl') / 1024  # KB
            print(f"✅ scaler.pkl - {scaler_size:.2f} KB")
        else:
            print("❌ scaler.pkl - Missing")
            return False
        
        # Try to load the model
        try:
            from app import PhishingDetector, FEATURE_NAMES
            model = PhishingDetector(len(FEATURE_NAMES))
            model.load_state_dict(torch.load('model.pth', map_location='cpu'))
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading error: {str(e)}")
            return False
        
        # Try to load the scaler
        try:
            scaler = joblib.load('scaler.pkl')
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ Scaler loading error: {str(e)}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Model files check error: {str(e)}")
        return False

def check_templates():
    """Check if all templates are valid."""
    print("\n🔍 Checking templates...")
    
    template_files = [
        'templates/base.html',
        'templates/index.html',
        'templates/login.html',
        'templates/register.html',
        'templates/check.html',
        'templates/history.html',
        'templates/profile.html'
    ]
    
    for template in template_files:
        try:
            with open(template, 'r', encoding='utf-8') as f:
                content = f.read()
                if '{% extends' in content or '{% block' in content:
                    print(f"✅ {template} - Valid Jinja2 template")
                else:
                    print(f"⚠️  {template} - May not be a valid Jinja2 template")
        except Exception as e:
            print(f"❌ {template} - Error reading: {str(e)}")
            return False
    
    return True

def check_static_files():
    """Check if static files exist and are accessible."""
    print("\n🔍 Checking static files...")
    
    static_files = [
        'static/css/auth.css',
        'static/css/style.css',
        'static/css/landing.css',
        'static/css/profile.css'
    ]
    
    for file_path in static_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} - {size} bytes")
        else:
            print(f"❌ {file_path} - Missing")
            return False
    
    return True

def check_gunicorn_config():
    """Check Gunicorn configuration."""
    print("\n🔍 Checking Gunicorn configuration...")
    
    try:
        with open('gunicorn.conf.py', 'r') as f:
            config = f.read()
        
        required_settings = ['bind', 'workers', 'worker_class', 'timeout']
        for setting in required_settings:
            if setting in config:
                print(f"✅ {setting} - Configured")
            else:
                print(f"❌ {setting} - Missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Gunicorn config error: {str(e)}")
        return False

def check_dataset():
    """Check if dataset file is valid."""
    print("\n🔍 Checking dataset...")
    
    try:
        import pandas as pd
        
        if os.path.exists('dataset_phishing.csv'):
            df = pd.read_csv('dataset_phishing.csv')
            print(f"✅ dataset_phishing.csv - {len(df)} rows, {len(df.columns)} columns")
            
            # Check for required columns
            required_cols = ['status']
            for col in required_cols:
                if col in df.columns:
                    print(f"✅ Required column '{col}' - Found")
                else:
                    print(f"❌ Required column '{col}' - Missing")
                    return False
            
            return True
        else:
            print("❌ dataset_phishing.csv - Missing")
            return False
    except Exception as e:
        print(f"❌ Dataset check error: {str(e)}")
        return False

def main():
    """Run all deployment checks."""
    print("🚀 Phishing URL Detector - Deployment Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Files", check_required_files),
        ("Dependencies", check_dependencies),
        ("Environment Variables", check_environment_variables),
        ("Flask Application", check_flask_app),
        ("Model Files", check_model_files),
        ("Templates", check_templates),
        ("Static Files", check_static_files),
        ("Gunicorn Configuration", check_gunicorn_config),
        ("Dataset", check_dataset)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} - Error: {str(e)}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📊 DEPLOYMENT CHECK SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your application is ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Set up MongoDB Atlas database")
        print("2. Update MONGODB_URI in render.yaml with your Atlas connection string")
        print("3. Deploy to Render")
        return True
    else:
        print("⚠️  Some checks failed. Please fix the issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 