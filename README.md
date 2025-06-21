# Phishing URL Detection System

A full-stack web application that uses machine learning to detect phishing URLs. The system uses a Bidirectional GRU model trained on a dataset of legitimate and phishing URLs.

## Features

- User authentication (register/login)
- URL phishing detection
- History tracking of checked URLs
- User profile management
- Responsive design with custom CSS
- MongoDB integration for data persistence

## Tech Stack

- Frontend: HTML, CSS, JavaScript, jQuery
- Backend: Python (Flask)
- Database: MongoDB
- Machine Learning: TensorFlow (GRU model)

## Project Structure

```
Phishing-Checker/
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   └── history.html
├── app.py
├── model.h5
├── phishing_utils.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start MongoDB service on your system

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Access the application at `http://localhost:5000`

## Usage

1. Register a new account or login with existing credentials
2. Navigate to the History/Checker page
3. Enter a URL to check for phishing
4. View results and check history of previous scans
5. Manage your profile information

## Security Features

- Password hashing using bcrypt
- Session-based authentication
- Input validation and sanitization
- Secure MongoDB connection

## Model Information

The system uses a Bidirectional GRU (Gated Recurrent Unit) neural network trained on a dataset of legitimate and phishing URLs. The model analyzes various URL features to make predictions about the likelihood of a URL being malicious.

## License

MIT License 