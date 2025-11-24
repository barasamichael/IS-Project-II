# StrathQuery

A conversational query system designed specifically for Strathmore University students to access university information.

## Overview

StrathQuery allows students to ask questions about university policies, course information, academic calendar, faculty details, and more. The system connects to an external API that handles the actual query processing and response generation.

## Features

- **User Account Management**: Create and manage student accounts
- **Conversation History**: View and continue previous conversations
- **Message Classification**: Automatic categorization of messages by intent and topic

## Technology Stack

- **Backend**: Flask, SQLAlchemy
- **Database**: SQLite
- **Authentication**: Flask-Login
- **Frontend**: HTML/CSS/JavaScript with Bootstrap

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment tool (virtualenv or venv)

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/Ben-Tait/Strath-Query.git
   cd Strath-Query
   ```

2. Create and activate virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables
   ```bash
   cp .env.example .env
   # Edit .env file with your settings
   ```

5. Initialize the database
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

6. Run the development server
   ```bash
   flask run
   ```

## Project Structure

```
Strath-Query/
├── app/                    # Application package
│   ├── accounts/           # User account management
│   ├── authentication/     # Login, registration, etc.
│   ├── models/             # Database models
│   ├── static/             # CSS, JS, images
│   ├── templates/          # HTML templates
│   ├── __init__.py         # Application factory
├── utilities/              # Utility functions
├── requirements.txt        # Project dependencies
├── config.py               # Environment configuration
└── flasky.py                 # WSGI entry point
```

## Usage

### Student Access

1. Register for an account with your Strathmore email address
2. Log in to the platform
3. Start a new conversation by typing your question
4. View your conversation history in the user dashboard

## Contact

For questions or support, please contact:
- Project Owner: Benjamin Tait (benjamin.tait@strathmore.edu)
