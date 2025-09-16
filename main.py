import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, messagebox, filedialog
import threading
import queue
import pyaudio
import numpy as np
import webrtcvad
import requests
import json
from faster_whisper import WhisperModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import API_KEY,CHANNELS,RATE,CHUNK,FORMAT,DURATION,MODEL_SIZE,EMBEDDING_MODEL,CONTENT_GEN_MODEL
from arrays import text_embeddings, transcript_lines

# ----- CONFIG -----
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
format = pyaudio.paInt16
vad = webrtcvad.Vad(2)

audio_queue = queue.Queue()
embedding_queue = queue.Queue() 
running = False
stream = None
p = pyaudio.PyAudio()

selected_file = None

# ----- EMBEDDING -----
# Gets the texts of the queue in background
def embedding_loop():
    while True:
        text = embedding_queue.get()
        if text is None:
            break
        embeddings = get_embeddings([text])
        if embeddings:
            text_embeddings.append((text, embeddings[0]))
            print(f"Embedding completed: '{text[:30]}...'")
        else:
            print(f"Embedding failed: '{text[:30]}...'")
        embedding_queue.task_done()

# we start the embedding thread 
threading.Thread(target=embedding_loop, daemon=True).start()

# Takes the given text and returns a numeric vector
def get_embeddings(texts):
    url = (
        f"https://generativelanguage.googleapis.com/"
        f"v1beta/models/gemini-embedding-001:batchEmbedContents"
        f"?key={API_KEY}"
    )
    headers = {"Content-Type": "application/json"}
    data = {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {
                    "parts": [{"text": text}]
                }
            }
            for text in texts
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        resp_json = response.json()
        return [item["values"] for item in resp_json["embeddings"]]
    except Exception as e:
        print(f"Embedding hatası: {e}")
        return None


# ----- LLM -----
# returns similiar texts 
def find_most_relevant_context(question, top_n=3):
    if not text_embeddings:
        return "Context not found"
    
    question_embedding = get_embeddings([question])
    if not question_embedding:
        return "Question could not be vectorized"
    
    question_vec = np.array(question_embedding[0]).reshape(1, -1)
    
    similarities = []
    for text, embedding in text_embeddings:
        text_vec = np.array(embedding).reshape(1, -1)
        sim = cosine_similarity(question_vec, text_vec)[0][0]
        similarities.append((text, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return "\n".join([f"[{i+1}] {text}" for i, (text, sim) in enumerate(similarities[:top_n])])

# return generated contents
def gemini_generate_content(prompt_text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }
    data = {"contents": [{"parts": [{"text": prompt_text}]}]}

    try:
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=15
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[API Error] {e}"


# --- GUI Setup ---
root = tk.Tk()
root.title("Live Transcript + Gemini AI")

# Radio button 
source_type = tk.StringVar(value="microphone")

# microphone or file selection
radio_frame = tk.Frame(root)
radio_frame.pack(pady=5)

tk.Radiobutton(radio_frame, text="Microphone", variable=source_type, value="microphone").pack(side=tk.LEFT, padx=10)
tk.Radiobutton(radio_frame, text="File", variable=source_type, value="file").pack(side=tk.LEFT, padx=10)

def select_file():
    global selected_file
    f = filedialog.askopenfilename(
        filetypes=[("Ses Dosyaları", "*.mp3 *.wav *.m4a"), ("Video Dosyaları", "*.mp4 *.mov"), ("Tüm Dosyalar", "*.*")]
    )
    if f:
        selected_file = f
        file_label.config(text=f"Selected file: {f.split('/')[-1]}")
    else:
        selected_file = None
        file_label.config(text="No file selected")

file_btn = tk.Button(root, text="Select file", command=select_file)
file_btn.pack(pady=2)
file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=2)

# lists PyAudio devices  
input_devices = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        input_devices.append((i, info["name"]))

device_var = tk.StringVar()
device_combo = ttk.Combobox(root, textvariable=device_var, state="readonly")
device_combo["values"] = [f"{i} - {name}" for i, name in input_devices]
if input_devices:
    device_combo.current(0)
device_combo.pack(pady=5)

# --- Microphone Functions ---
# really time voice datas collect with filter vad
def record_audio():
    frames = []
    num_frames = int(RATE / CHUNK * DURATION)
    for _ in range(num_frames):
        if not running:
            return None
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except OSError as e:
            if "Input overflowed" in str(e):
                continue
            else:
                raise
    audio_bytes = b"".join(frames)
    frame_bytes = int(RATE * 0.01) * 2  # 10ms
    speech_frames = []
    
    for i in range(0, len(audio_bytes), frame_bytes):
        frame = audio_bytes[i:i+frame_bytes]
        if len(frame) == frame_bytes:
            try:
                if vad.is_speech(frame, RATE):
                    speech_frames.append(frame)
            except:
                pass
    
    if not speech_frames:
        speech_frames = [audio_bytes]
        
    return np.frombuffer(b"".join(speech_frames), dtype=np.int16).astype(np.float32) / 32768.0

# It processes the audio data received from the microphone in sequence and transcribes it.
def transcribe_audio():
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break

        segments, _ = model.transcribe(audio_data, beam_size=2, language=None)
        for segment in segments:
            if segment.no_speech_prob > 0.8 and segment.avg_logprob < -1.5:
                text = "[inaudible]"
            else:
                text = segment.text.strip()

            transcript_lines.append(text);
            embedding_queue.put(text)
            gui_text.insert(tk.END, f"{text}\n")
            gui_text.see(tk.END)

        audio_queue.task_done()

# vectorizer the text and save 
def update_embeddings(text):
    embeddings = get_embeddings([text])
    if embeddings:
        text_embeddings.append((text, embeddings[0]))
        print(f"Text vectorized: {text[:50]}...")

# Runs the record_audio method in an infinite loop
def record_loop():
    while running:
        audio_data = record_audio()
        if audio_data is not None and running:
            audio_queue.put(audio_data)

# It processes the audio data received from the file in sequence and transcribes it.
def transcribe_file():
    global running
    try:
        segments, _ = model.transcribe(selected_file, beam_size=1)
        for segment in segments:
            if not running:
                break
            text = segment.text.strip()
            if text:
                transcript_lines.append(text)
                gui_text.insert(tk.END, text + "\n")
                gui_text.see(tk.END)
                
                threading.Thread(target=update_embeddings, args=(text,), daemon=True).start()
                
    except Exception as e:
        messagebox.showerror("Error", f"Error while transcribing file: {e}")
    finally:
        running = False
        stop_btn.config(state=tk.DISABLED)
        start_btn.config(state=tk.NORMAL)

# Starts recording audio and starts a thread
def start_recording():
    global stream, running

    if running:
        return

    running = True
    transcript_lines.clear()
    text_embeddings.clear()  
    gui_text.delete("1.0", tk.END)
    stop_btn.config(state=tk.NORMAL)
    start_btn.config(state=tk.DISABLED)

    threading.Thread(target=transcribe_audio, daemon=True).start() # !

    if source_type.get() == "microphone":
        if not input_devices:
            messagebox.showerror("Error", "System not found the device.")
            running = False
            return
        try:
            selected_index = int(device_combo.get().split(" - ")[0])
        except Exception:
            messagebox.showerror("Error", "Please select the voice enter device.")
            running = False
            return
            
        stream = p.open(
            format=format,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=selected_index,
            frames_per_buffer=CHUNK
        )
        threading.Thread(target=record_loop, daemon=True).start()

    elif source_type.get() == "file" and selected_file:
        threading.Thread(target=transcribe_file, daemon=True).start()
    else:
        messagebox.showerror("Error", "Please first select a file.")
        running = False

# It stops recording audio
def stop_recording():
    global stream, running
    running = False
    if stream:
        stream.stop_stream()
        stream.close()
        stream = None
    audio_queue.put(None)
    stop_btn.config(state=tk.DISABLED)
    start_btn.config(state=tk.NORMAL)

# --- GUI Buttons ---
frame = tk.Frame(root)
frame.pack(pady=5)

start_btn = tk.Button(frame, text="Start", command=start_recording)
start_btn.grid(row=0, column=0, padx=5)

stop_btn = tk.Button(frame, text="Stop", command=stop_recording, state=tk.DISABLED)
stop_btn.grid(row=0, column=1, padx=5)

tk.Button(frame, text="Show the all texts", command=lambda: show_all_text()).grid(row=1, column=0, padx=5, pady=5)
tk.Button(frame, text="Summarize", command=lambda: summarize_text()).grid(row=1, column=1, padx=5, pady=5)
tk.Button(frame, text="Ask question", command=lambda: qa_text()).grid(row=1, column=2, padx=5, pady=5)
tk.Button(frame, text="Contextual Question", command=lambda: contextual_qa_text()).grid(row=1, column=3, padx=5, pady=5)

gui_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
gui_text.pack(padx=10, pady=10)


# --- LLM methods ---
def show_all_text():
    if not transcript_lines:
        messagebox.showinfo("Info", "No text yet.")
        return
    popup = tk.Toplevel(root)
    popup.title("All Transcript")
    text_box = scrolledtext.ScrolledText(popup, wrap=tk.WORD, width=80, height=30)
    text_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    text_box.insert(tk.END, "\n".join(transcript_lines))
    text_box.config(state=tk.DISABLED)

def summarize_text():
    if not transcript_lines:
        messagebox.showinfo("Info", "Nothing to summarize.")
        return
    full_text = " ".join(transcript_lines)
    summary = gemini_generate_content(f"""
        Lütfen aşağıdaki konuşmayı kısa ve anlaşılır bir şekilde özetle.
        Özellikle **önemli noktaları ve kritik bilgileri** vurgula.
        Gerekirse madde işaretleri ile sun ve gereksiz detayları atla.

        Konuşma:
        {full_text}
        """)
    messagebox.showinfo("Gemini Summary", summary)

def qa_text():
    if not transcript_lines:
        messagebox.showinfo("Info", "There is no text to ask question.")
        return
    question = simpledialog.askstring("Question", "Write a question about the text:")
    if question:
        full_text = " ".join(transcript_lines)
        prompt = f"""
            Aşağıdaki konuşmaya göre soruyu kısa ve anlaşılır bir şekilde yanıtla.
            Metin:
            {full_text}

            Soru:
            {question}

            Cevap:
            """
        answer = gemini_generate_content(prompt)
        messagebox.showinfo("Gemini Response", answer)

def contextual_qa_text():
    if not transcript_lines:
        messagebox.showinfo("Info", "There is no text to ask question.")
        return
    question = simpledialog.askstring("Contextual Question", "Write a contextual question about the text:")
    if question:
        processing_window = tk.Toplevel(root)
        processing_window.title("Processing")
        tk.Label(processing_window, text="Processing question, please wait...").pack(padx=20, pady=20)
        processing_window.update()
        
        try:
            context = find_most_relevant_context(question)
            
            prompt = (
                f"Aşağıdaki bağlamı kullanarak soruyu yanıtla:\n\n"
                f"[BAĞLAM]\n{context}\n\n"
                f"[SORU]\n{question}\n\n"
                f"Cevap:"
            )
            
            answer = gemini_generate_content(prompt)
            
            result_window = tk.Toplevel(root)
            result_window.title("Contextual Answer")
            result_window.geometry("800x600")
            
            # Context panel
            context_frame = tk.LabelFrame(result_window, text="Most Relevant Contexts")
            context_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            context_text = scrolledtext.ScrolledText(context_frame, wrap=tk.WORD, height=8)
            context_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            context_text.insert(tk.END, context)
            context_text.config(state=tk.DISABLED)
            
            # Response panel
            answer_frame = tk.LabelFrame(result_window, text="Gemini Response")
            answer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            answer_text = scrolledtext.ScrolledText(answer_frame, wrap=tk.WORD, height=8)
            answer_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            answer_text.insert(tk.END, answer)
            answer_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing contextual question: {e}")
        finally:
            processing_window.destroy()

def on_close():
    stop_recording()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()