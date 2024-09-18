import os
import time
from flask import Flask, render_template, request, jsonify
from audio_processing import convert_to_wav, transcribe_audio, save_transcription, cleanup_files
from text_processing import textrank_summarize

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['SUMMARIES_FOLDER'] = 'summaries'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['SUMMARIES_FOLDER'], exist_ok=True)

def save_file(file, upload_folder):
    """
    Save the uploaded file to the specified folder.
    """
    upload_path = os.path.join(upload_folder, file.filename)
    file.save(upload_path)
    return upload_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            start_time = time.time()
            
            # Save the uploaded file
            upload_path = save_file(file, app.config['UPLOAD_FOLDER'])
            
            # Convert the audio file to WAV format
            wav_path = convert_to_wav(upload_path)
            
            # Perform transcription
            transcription = transcribe_audio(wav_path)
            
            # Save the transcription to a text file
            save_transcription(transcription, app.config['RESULTS_FOLDER'], file.filename)
            
            # Clean up temporary files
            cleanup_files(upload_path, wav_path)
            
            # Read the transcription text
            transcription_path = os.path.join(app.config['RESULTS_FOLDER'], f'{os.path.splitext(file.filename)[0]}.txt')
            with open(transcription_path, 'r', encoding='utf-8') as transcription_file:
                text = transcription_file.read()
            
            # Process and summarize the transcription
            summary = textrank_summarize(text)
            
            summary_filename = f'summary_{os.path.splitext(file.filename)[0]}.txt'
            summary_path = os.path.join(app.config['SUMMARIES_FOLDER'], summary_filename)
            with open(summary_path, 'w', encoding='utf-8') as summary_file:
                summary_file.write(summary)
            
            end_time = time.time()
            duration = end_time - start_time

            print(f"Processing time: {duration:.2f} seconds")

            return jsonify({
                'transcription': transcription,
                'summary': summary,
                'processing_time': duration
            })
        except Exception as e:
            app.logger.error(f"Error processing audio: {str(e)}")
            return jsonify({'error': 'Error occurred while processing the audio.'}), 500
        

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    transcription = data.get('transcription', '')

    if not transcription:
        return jsonify({'error': 'No transcription provided'}), 400

    try:
        summary = textrank_summarize(transcription)
        return jsonify({'summary': summary})
    except Exception as e:
        app.logger.error(f"Error summarizing text: {str(e)}")
        return jsonify({'error': 'Error occurred while summarizing the text.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
