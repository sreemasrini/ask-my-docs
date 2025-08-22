# Ask My Docs - Backend

## Prerequisites
- Python 3.9+
- MongoDB
- OpenAI API Key

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/ask-my-docs.git
cd ask-my-docs/backend
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
export OPENAI_API_KEY='your_openai_api_key'
export MONGO_URI='mongodb://localhost:27017'  # Adjust if using a different MongoDB setup
```

5. Run the application
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `POST /upload`: Upload a PDF or text document
  - Accepts multipart/form-data with a file
  - Processes document, generates embeddings, and stores in MongoDB

- `POST /query`: Query uploaded documents
  - Accepts JSON with a query string
  - Returns an AI-generated answer based on document context

## MongoDB Setup

1. Install MongoDB
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y mongodb
```

2. Start MongoDB service
```bash
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

## Notes
- Supports PDF and text file uploads
- Uses OpenAI for embeddings and answer generation
- Stores document chunks and embeddings in MongoDB
- Retrieves top-3 most relevant document chunks for each query

## Troubleshooting
- Ensure MongoDB is running
- Check that OPENAI_API_KEY is correctly set
- Verify Python dependencies are installed
