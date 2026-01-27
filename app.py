import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys

# ==========================================
# PART 1: BACKEND LOGIC (Runs in IDE Console)
# ==========================================

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    return text

def init_model():
    """
    Loads data, trains model, and returns necessary objects.
    This runs in the console before the GUI starts.
    """
    print("--------------------------------------------------")
    print("           INITIALIZING FAKE NEWS DETECTOR        ")
    print("--------------------------------------------------")

    try:
        # 1. Data Loading
        print("[1/5] Loading datasets (Fake.csv, True.csv)...")
        fake_df = pd.read_csv("Fake.csv")
        true_df = pd.read_csv("True.csv")

        fake_df["label"] = 1
        true_df["label"] = 0

        # Stats for UI
        stats = {
            "total": len(fake_df) + len(true_df),
            "fake_count": len(fake_df),
            "true_count": len(true_df)
        }

        print(f"      - Loaded {stats['total']} articles.")

        # 2. Preprocessing
        print("[2/5] Merging and shuffling data...")
        df = pd.concat([fake_df, true_df])
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Combine title and text
        df["content"] = df["title"] + " " + df["text"]

        print("[3/5] Cleaning text (removing special chars, lowercasing)...")
        # Optimization: cleaning can take time, doing it in place
        df["content"] = df["content"].apply(clean_text)

        X = df["content"]
        y = df["label"]

        # 3. Vectorization
        print("[4/5] Vectorizing text (TF-IDF)...")
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
        X_vec = vectorizer.fit_transform(X)

        # 4. Training
        print("[5/5] Training Logistic Regression Model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_vec, y)

        # Accuracy Check
        accuracy = accuracy_score(y, model.predict(X_vec))
        print(f"\n>>> Model Ready! Accuracy: {accuracy*100:.2f}%")
        print(">>> Launching GUI Application...")
        
        return model, vectorizer, accuracy, stats

    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not find CSV files. Make sure 'Fake.csv' and 'True.csv' are in the directory.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during initialization.")
        print(f"Details: {e}")
        sys.exit(1)

# ==========================================
# PART 2: FRONTEND UI (Tkinter)
# ==========================================

class FakeNewsApp:
    def __init__(self, root, model, vectorizer, accuracy, stats):
        self.root = root
        self.model = model
        self.vectorizer = vectorizer
        self.accuracy = accuracy
        self.stats = stats

        self.root.title("Fake News Detector")
        self.root.geometry("700x750")
        self.root.configure(bg="#f0f2f6")

        self._setup_ui()

    def _setup_ui(self):
        # --- Header ---
        header_frame = tk.Frame(self.root, bg="#0e1117", pady=15)
        header_frame.pack(fill="x")
        
        title_lbl = tk.Label(header_frame, text="📰 Fake News Detection AI", font=("Helvetica", 18, "bold"), fg="white", bg="#0e1117")
        title_lbl.pack()
        
        subtitle_lbl = tk.Label(header_frame, text="Enter a news article below to check authenticity", font=("Helvetica", 10), fg="#b0b0b0", bg="#0e1117")
        subtitle_lbl.pack()

        # --- Sidebar / Stats Frame (simulating Streamlit sidebar) ---
        stats_frame = tk.LabelFrame(self.root, text="📊 Dataset Statistics", font=("Helvetica", 10, "bold"), bg="#f0f2f6", padx=10, pady=5)
        stats_frame.pack(fill="x", padx=20, pady=10)

        stat_str = (f"Total Articles: {self.stats['total']}  |  "
                    f"Fake: {self.stats['fake_count']}  |  "
                    f"Real: {self.stats['true_count']}  |  "
                    f"Model Accuracy: {self.accuracy*100:.2f}%")
        
        stat_lbl = tk.Label(stats_frame, text=stat_str, bg="#f0f2f6", fg="#333")
        stat_lbl.pack(anchor="w")

        # --- Main Input Area ---
        input_frame = tk.Frame(self.root, bg="#f0f2f6")
        input_frame.pack(fill="both", expand=True, padx=20, pady=5)

        tk.Label(input_frame, text="✍️ Enter News Text:", bg="#f0f2f6", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(5,0))
        
        self.text_area = scrolledtext.ScrolledText(input_frame, height=10, font=("Segoe UI", 10))
        self.text_area.pack(fill="both", expand=True, pady=5)

        # --- Buttons ---
        btn_frame = tk.Frame(self.root, bg="#f0f2f6")
        btn_frame.pack(fill="x", padx=20, pady=10)

        check_btn = tk.Button(btn_frame, text="🔍 Check News", command=self.check_news, 
                              bg="#ff4b4b", fg="white", font=("Helvetica", 10, "bold"), 
                              padx=20, pady=5, relief="flat", cursor="hand2")
        check_btn.pack(side="left")

        clear_btn = tk.Button(btn_frame, text="🔄 Clear", command=self.clear_text, 
                              bg="white", fg="#333", font=("Helvetica", 10), 
                              padx=20, pady=5, relief="flat", cursor="hand2")
        clear_btn.pack(side="left", padx=10)

        # --- Results Area ---
        self.result_frame = tk.LabelFrame(self.root, text="📌 Prediction Result", font=("Helvetica", 11, "bold"), bg="#f0f2f6", padx=10, pady=10)
        self.result_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.result_lbl = tk.Label(self.result_frame, text="Waiting for input...", font=("Helvetica", 14), bg="#f0f2f6", fg="#666")
        self.result_lbl.pack(pady=5)

        self.confidence_lbl = tk.Label(self.result_frame, text="", font=("Helvetica", 10), bg="#f0f2f6")
        self.confidence_lbl.pack()

        # Feature importance area
        self.feature_lbl = tk.Label(self.result_frame, text="", font=("Courier New", 9), bg="#f0f2f6", justify="left")
        self.feature_lbl.pack(pady=10)

    def clear_text(self):
        self.text_area.delete("1.0", tk.END)
        self.result_lbl.config(text="Waiting for input...", fg="#666")
        self.confidence_lbl.config(text="")
        self.feature_lbl.config(text="")

    def check_news(self):
        user_input = self.text_area.get("1.0", tk.END).strip()
        
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter some news text to check.")
            return

        # 1. Clean
        cleaned = clean_text(user_input)
        
        # 2. Vectorize
        vec_input = self.vectorizer.transform([cleaned])

        # 3. Predict
        prediction = self.model.predict(vec_input)[0]
        probability = self.model.predict_proba(vec_input).max()

        # 4. Display Results
        if prediction == 1:
            self.result_lbl.config(text="⚠️ Fake News Detected", fg="#d9534f") # Red
        else:
            self.result_lbl.config(text="✅ Real News Detected", fg="#28a745") # Green

        self.confidence_lbl.config(text=f"Confidence: {probability*100:.2f}%")

        # 5. Feature Importance (Top influencing words)
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Get indices of non-zero elements in input vector
        # (This logic finds which words *in the user input* were important)
        input_indices = vec_input.indices
        
        if len(input_indices) > 0:
            relevant_words = []
            for idx in input_indices:
                word = feature_names[idx]
                score = coefficients[idx]
                relevant_words.append((word, score))
            
            # Sort by absolute impact (highest magnitude first)
            relevant_words.sort(key=lambda x: abs(x[1]), reverse=True)
            top_words = relevant_words[:8] # Top 8 words

            features_text = "🧠 Influential words in this text:\n"
            for word, score in top_words:
                impact = "Fake-leaning" if score > 0 else "Real-leaning"
                features_text += f"• {word} ({impact})\n"
            
            self.feature_lbl.config(text=features_text)
        else:
            self.feature_lbl.config(text="No significant keywords found in training dictionary.")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    model, vectorizer, acc, stats = init_model()

    # 2. Launch GUI
    root = tk.Tk()


    app = FakeNewsApp(root, model, vectorizer, acc, stats)
    root.mainloop()