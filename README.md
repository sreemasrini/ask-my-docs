# Ask My Docs

A full-stack AI-powered document query application that allows users to upload PDFs or text files and ask questions about their contents.

## Features

- Upload PDF and text documents
- AI-powered document question answering
- Retrieval-Augmented Generation (RAG)
- Source tracking with relevance scores

## Prerequisites

- Python 3.9+
- Node.js 16+
- MongoDB
- OpenAI API Key

## Technology Stack

- **Backend**: Python, FastAPI
- **Frontend**: React, TypeScript, Tailwind CSS
- **Database**: MongoDB
- **AI**: OpenAI Embeddings and GPT

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ask-my-docs.git
cd ask-my-docs
```

### 2. Backend Setup

1. Navigate to backend directory
```bash
cd backend
```

2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set environment variables
```bash
export OPENAI_API_KEY='your_openai_api_key'
export MONGO_URI='mongodb://localhost:27017'
```

### 3. Frontend Setup

1. Navigate to frontend directory
```bash
cd ../frontend
```

2. Install dependencies
```bash
yarn install
```

### 4. MongoDB Setup

1. Install MongoDB (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y mongodb
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

### 5. Running the Application

1. Start Backend (in backend directory)
```bash
uvicorn app.main:app --reload
```

2. Start Frontend (in frontend directory)
```bash
yarn start
```

## Usage

1. Upload a PDF or text document
2. Ask questions about the document
3. Receive AI-generated answers with source references

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `MONGO_URI`: MongoDB connection string

## Troubleshooting

- Ensure MongoDB is running
- Check OpenAI API key is valid
- Verify all dependencies are installed

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- OpenAI
- FastAPI
- React
- MongoDB
