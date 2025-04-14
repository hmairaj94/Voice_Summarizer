# 🔥 Voice-Activated Summarizer with Sentiment Insight

A Streamlit web application that transcribes audio, generates summaries, and analyzes sentiment, powered by Hugging Face models.

## 📝 Overview

This application allows users to upload audio files, which are then:
1. Transcribed into text
2. Summarized for key points
3. Analyzed for sentiment (positive/negative)
4. Stored in a session history for easy reference

## ⚠️ Important: Hugging Face API Key Required

This application requires a Hugging Face API key to function:
- You must create an account on [Hugging Face](https://huggingface.co/)
- Generate an API key in your account settings
- Store this key in a `.env` file as shown in the installation section

The app will not work without a valid API key as it relies on Hugging Face's hosted models.

## 🚀 Features

- **Audio Transcription**: Converts spoken words into text using OpenAI's Whisper model
- **Text Summarization**: Generates concise summaries using Facebook's BART model
- **Sentiment Analysis**: Determines the emotional tone of the content using DistilBERT
- **Analysis History**: Keeps track of previous analyses for comparison and reference
- **User-friendly Interface**: Built with Streamlit for easy interaction

## 🔧 Requirements

- Python 3.7+
- Streamlit
- Requests
- NumPy
- SoundFile
- python-dotenv
- Hugging Face API key

## 📦 Installation

1. Clone this repository
```bash
git clone https://github.com/hmairaj94/Voice_Summarizer.git
cd voice-summarizer
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Hugging Face API key
```
HF_API_KEY=your_huggingface_api_key
```

Without this API key, the application will prompt you to enter it in the UI, but for security reasons, it's recommended to use the `.env` file approach.

## 🏃‍♂️ Usage

1. Run the Streamlit application
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload an audio file (WAV, MP3, or M4A format)

4. Wait for the analysis to complete and review the results

## 🧠 How It Works

The application uses three key models from Hugging Face:

- **openai/whisper-small**: For accurate speech-to-text transcription
- **facebook/bart-large-cnn**: For generating concise summaries
- **distilbert-base-uncased-finetuned-sst-2-english**: For sentiment analysis

## ⚠️ Limitations

- Audio files must be clear for accurate transcription
- Transcripts exceeding 1024 words will be truncated for summarization
- Very short transcripts (less than 10 words) cannot be meaningfully summarized

## 🔒 Privacy

All processing is done through Hugging Face's API. No data is permanently stored beyond your local session.

## 🤝 Contributing
![Screenshot (26)](https://github.com/user-attachments/assets/9fbb56e6-4b7a-4d69-adc8-8517dedafa6f)
![Screenshot (25)](https://github.com/user-attachments/assets/f9e68eca-7e2e-47b8-aed0-d468b1af0c43)
![Screenshot (24)](https://github.com/user-attachments/assets/b01aa84c-f691-4658-9659-131d686a21a1)


Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/hmairaj94/Voice_Summarizer/issues).
