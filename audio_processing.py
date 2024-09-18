import os
from pydub import AudioSegment
import whisper
import torch

# Load Whisper model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium").to(DEVICE)

def save_file(file, upload_folder):
    """
    Menyimpan file yang diunggah ke folder tertentu.
    """
    upload_path = os.path.join(upload_folder, file.filename)
    file.save(upload_path)
    return upload_path

def convert_to_wav(upload_path):
    """
    Mengkonversi semua format file audio yang diunggah menjadi format WAV.
    """
    audio_format = os.path.splitext(upload_path)[1].lower()  # Dapatkan ekstensi file
    wav_path = os.path.splitext(upload_path)[0] + '.wav'
    
    # Gunakan pydub untuk mengonversi file audio ke wav
    audio = AudioSegment.from_file(upload_path, format=audio_format[1:])
    audio.export(wav_path, format="wav")
    
    return wav_path

def transcribe_audio(wav_path):
    """
    Melakukan transkripsi terhadap file WAV menggunakan model Whisper.
    """
    result = model.transcribe(wav_path, fp16=(DEVICE == "cuda"))
    return result["text"]

def save_transcription(transcription, results_folder, filename):
    """
    Menyimpan hasil transkripsi ke dalam file teks.
    """
    txt_filename = os.path.join(results_folder, f"{os.path.splitext(filename)[0]}.txt")
    with open(txt_filename, 'w', encoding='utf-8') as txt_file:
        txt_file.write(transcription)

def cleanup_files(*file_paths):
    """
    Menghapus file sementara seperti file WAV hasil konversi untuk menghemat ruang penyimpanan.
    """
    for path in file_paths:
        if os.path.exists(path):
            os.unlink(path)

# Fungsi utama untuk menangani file audio
def process_audio_file(file, upload_folder, results_folder):
    """
    Proses utama untuk menyimpan file audio, mengonversinya ke WAV, melakukan transkripsi,
    dan menyimpan hasilnya ke file teks.
    """
    # Simpan file yang diunggah
    upload_path = save_file(file, upload_folder)
    
    # Konversi file audio ke format wav
    wav_path = convert_to_wav(upload_path)
    
    # Lakukan transkripsi
    transcription = transcribe_audio(wav_path)
    
    # Simpan hasil transkripsi ke file teks
    save_transcription(transcription, results_folder, os.path.basename(upload_path))
    
    # Bersihkan file sementara
    cleanup_files(upload_path, wav_path)

    return transcription
