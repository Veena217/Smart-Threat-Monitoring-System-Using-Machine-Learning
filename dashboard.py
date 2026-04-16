print("### THIS FILE IS RUNNING ###")


import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.utils import to_categorical


class CyberThreatDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Cyber Threat Detection Dashboard")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.feature_extractor = None # To store the trained TF-IDF Vectorizer
        self.trained_models = {}      # To store all trained models
        self.label_encoder = None     # To store the trained LabelEncoder
        self.label_names = [] 
        self.lstm_results = None      # To store the original label names
        
        self.cnn_result_text = ""
        self.lstm_result_text = ""

        self.create_ui()
        
    def create_ui(self):
        # Header
        header = tk.Label(self.root, text="🛡️ CYBER THREAT DETECTION SYSTEM", 
                         font=("Arial", 18, "bold"), bg="#34495e", fg="white", pady=15)
        header.pack(fill=tk.X)
        
        # Main Frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Dataset Section
        dataset_frame = ttk.LabelFrame(main_frame, text="📊 Dataset Management", padding=10)
        dataset_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(dataset_frame, text="Load Dataset", command=self.load_dataset, width=30).pack(pady=5)
        self.dataset_label = tk.Label(dataset_frame, text="No dataset loaded", fg="red" , bg="#f0f0f0")
        self.dataset_label.pack(pady=5)

        ttk.Button(dataset_frame, text="Run Preprocessing (TF-IDF)", command=self.prepare_data, width=30).pack(pady=5)
        

        ttk.Button(dataset_frame, text="Generate Event Vector", command=self.generate_event_vector, width=30).pack(pady=5)

        # Models Section
        models_frame = ttk.LabelFrame(main_frame, text="🤖 Train Models", padding=10)
        models_frame.pack(fill=tk.X, pady=10)
        
        button_frame = tk.Frame(models_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Neural Network Profiling", command=self.neural_network_profiling, width=20).pack(side=tk.LEFT, padx=5, pady=5)

        models = [
            ("SVM", self.train_svm),
            ("KNN", self.train_knn),
            ("Decision Tree", self.train_dt),
            ("Random Forest", self.train_rf),
            ("Naïve Bayes", self.train_nb)
        ]
        
        for name, cmd in models:
            ttk.Button(button_frame, text=name, command=cmd, width=14).pack(side=tk.LEFT, padx=5, pady=5)
        

            
        # Threat Detection Section (New Block)
        detection_frame = ttk.LabelFrame(main_frame, text="🚨 Live Threat Detection", padding=10)
        detection_frame.pack(fill=tk.X, pady=10)
        
        # Add a button to trigger threat detection
        ttk.Button(detection_frame, text="Detect Threat for New Event Profile", command=self.detect_threat, width=40).pack(pady=5)
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="📈 Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results Text
        self.results_text = tk.Text(results_frame, height=15, width=90, bg="#ecf0f1", font=("Courier", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text['yscrollcommand'] = scrollbar.set
        
        # Comparison Button
        ttk.Button(main_frame, text="📊 Show Comparison Graph", command=self.show_comparison).pack(pady=10)
        
    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            initialdir="SOURCE CODE/SOURCE CODE/CyberThreat/datasets/",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
          try:
            self.dataset = pd.read_csv(file_path)
            self.dataset_label.config(text=f"✅ Loaded: {file_path.split('/')[-1]}", fg="#27ae60")
            self.print_result(f"Dataset loaded successfully!\nShape: {self.dataset.shape}\nColumns: {list(self.dataset.columns)}\n")
            # Clear any previous processed data/models
            self.X_train = self.X_test = self.y_train = self.y_test = None
            self.trained_models = {}
            self.results = {}
            self.feature_extractor = None
            self.label_encoder = None
            self.label_names = []
          except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")


    def prepare_data(self):
        try:
            self.label_encoder = preprocessing.LabelEncoder()
            dataset = self.dataset.copy()
            
            # Handle labels column (usually 'labels' or similar)
            label_col = [col for col in dataset.columns if 'label' in col.lower()][0]
            self.label_names = dataset[label_col].unique() # Store original label names
            dataset[label_col] = self.label_encoder.fit_transform(dataset[label_col])
            
            cols = dataset.shape[1] - 1
            X = dataset.values[:, 0:cols]
            Y = dataset.values[:, cols].astype('int')
            
            # Convert to TF-IDF features
            doc = []
            for i in range(len(X)):
                strs = ' '.join([str(x) for x in X[i]])
                doc.append(strs)
            
            self.feature_extractor = TfidfVectorizer() # Store the vectorizer
            tfidf = self.feature_extractor.fit_transform(doc)
            X = tfidf.toarray()
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42
            )
            
            self.print_result(f"Data prepared:\n- Training samples: {len(self.X_train)}\n- Test samples: {len(self.X_test)}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Data preparation failed: {str(e)}")
    
    def generate_event_vector(self):
        if self.dataset is None or self.feature_extractor is None:
            messagebox.showwarning("Warning", "Please load dataset and run preprocessing first!")
            return
        try:
            self.results_text.delete("1.0", tk.END)

            # Identify label column
            label_col = [col for col in self.dataset.columns if 'label' in col.lower()][0]
            
            # Dataset-derived info
            unique_events = sorted(self.dataset[label_col].astype(str).unique().tolist())
            total = len(self.dataset)
            train = len(self.X_train)
            test = len(self.X_test)

            # TF-IDF feature dimension
            vector_dim = self.X_train.shape[1]

            # Show a sample event vector (first training instance)
            sample_vector = self.X_train[0][:20]
            output = f"""
        Cyber Threat Detection Using Event Profiles

Unique event types detected in dataset:
{unique_events}

Total events in dataset        : {total}
Training events               : {train}
Testing events                : {test}

Event vector generation method : TF-IDF
Event vector dimension         : {vector_dim} features

Sample event vector (first 20 values):
{np.round(sample_vector, 3).tolist()} ...
"""
            self.print_result(output)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate event vector: {str(e)}")
            

    def neural_network_profiling(self):
      if self.X_train is None:
        messagebox.showwarning("Warning", "Please load and preprocess dataset first!")
        return
      
      # Train CNN if not trained
      if "CNN" not in self.trained_models:
         self.print_result("Training CNN...")
         self.train_cnn()  

      # Train LSTM
      if "LSTM" not in self.trained_models:
         self.print_result("Training LSTM...")
         self.train_lstm() 

    # Example: show a simple summary of neural network models
      summary_lines = []

      for name in ["CNN", "LSTM"]:
        model = self.trained_models.get(name)
        if model:
           model.summary(print_fn=lambda x: summary_lines.append(x))

      if summary_lines:
         self.results_text.insert(tk.END, "\n".join(summary_lines) + "\n")
         self.results_text.see(tk.END)
      else:
         messagebox.showinfo("Info", "No neural network models trained yet.")



    def train_svm(self):
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        self.print_result("Training SVM...\n")
        try:
            model = svm.SVC(C=2.0, gamma='scale', kernel='linear', random_state=0)
            model.fit(self.X_train, self.y_train)
            self.trained_models["SVM"] = model # Store the model
            y_pred = model.predict(self.X_test)
            self.calculate_metrics("SVM", y_pred)
        except Exception as e:
            messagebox.showerror("Error", f"SVM training failed: {str(e)}")
    
    def train_knn(self):
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        self.print_result("Training KNN...\n")
        try:
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(self.X_train, self.y_train)
            self.trained_models["KNN"] = model # Store the model
            y_pred = model.predict(self.X_test)
            self.calculate_metrics("KNN", y_pred)
        except Exception as e:
            messagebox.showerror("Error", f"KNN training failed: {str(e)}")
    
    def train_dt(self):
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        self.print_result("Training Decision Tree...\n")
        try:
            model = DecisionTreeClassifier(random_state=42)
            model.fit(self.X_train, self.y_train)
            self.trained_models["Decision Tree"] = model # Store the model
            y_pred = model.predict(self.X_test)
            self.calculate_metrics("Decision Tree", y_pred)
        except Exception as e:
            messagebox.showerror("Error", f"Decision Tree training failed: {str(e)}")
    
    def train_rf(self):
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        self.print_result("Training Random Forest...\n")
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(self.X_train, self.y_train)
            self.trained_models["Random Forest"] = model # Store the model
            y_pred = model.predict(self.X_test)
            self.calculate_metrics("Random Forest", y_pred)
        except Exception as e:
            messagebox.showerror("Error", f"Random Forest training failed: {str(e)}")
    
    def train_nb(self):
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        self.print_result("Training Naïve Bayes...\n")
        try:
            model = BernoulliNB()
            model.fit(self.X_train, self.y_train)
            self.trained_models["Naïve Bayes"] = model # Store the model
            y_pred = model.predict(self.X_test)
            self.calculate_metrics("Naïve Bayes", y_pred)
        except Exception as e:
            messagebox.showerror("Error", f"Naïve Bayes training failed: {str(e)}")

    def train_cnn(self):
     if self.X_train is None:
        messagebox.showwarning("Warning", "Load dataset first!")
        return

     try:
        # Reshape for Conv1D: (samples, timesteps, features)
        X_train = np.array(self.X_train).reshape(-1, self.X_train.shape[1], 1)
        X_test = np.array(self.X_test).reshape(-1, self.X_test.shape[1], 1)

        # Determine number of classes
        num_classes = len(np.unique(self.y_train))
        y_train_cat = to_categorical(self.y_train, num_classes=num_classes)
        y_test_cat = to_categorical(self.y_test, num_classes=num_classes)

        # Build CNN model
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
            MaxPooling1D(2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model with progress messages
        self.print_result("Training CNN... Please wait.")
        model.fit(X_train, y_train_cat, epochs=5, batch_size=32, verbose=1)

        # Predict on test set
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred) * 100
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100

        self.results["CNN"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        self.trained_models["CNN"] = model

        messagebox.showinfo(
            "Success", 
            f"CNN trained successfully!\nAccuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1 Score: {f1:.2f}%"
        )

        self.cnn_results = f"""
        ==================== CNN RESULTS ====================
        Accuracy   : {accuracy:.2f} %
        Precision  : {precision:.2f} %
        Recall     : {recall:.2f} %
        F1 Score   : {f1:.2f} %
        """
        self.update_nn_results()

     except Exception as e:
        messagebox.showerror("Error", f"CNN training failed: {str(e)}")


    def train_lstm(self):
     if self.X_train is None:
        messagebox.showwarning("Warning", "Load dataset first!")
        return

     try:
        X_train = np.array(self.X_train).reshape(-1, self.X_train.shape[1], 1)
        X_test = np.array(self.X_test).reshape(-1, self.X_test.shape[1], 1)

        num_classes = len(np.unique(self.y_train))
        y_train_cat = to_categorical(self.y_train, num_classes=num_classes)
        y_test_cat = to_categorical(self.y_test, num_classes=num_classes)

        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], 1)),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.print_result("Training LSTM... Please wait.")
        model.fit(X_train, y_train_cat, epochs=5, batch_size=32, verbose=1)

        # Predict on test set
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred) * 100
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100

        self.results["LSTM"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        self.trained_models["LSTM"] = model

        messagebox.showinfo(
            "Success", 
            f"LSTM trained successfully!\nAccuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1 Score: {f1:.2f}%"
        )

        self.lstm_results = f"""
        ==================== LSTM RESULTS ====================
        Accuracy   : {accuracy:.2f} %
        Precision  : {precision:.2f} %
        Recall     : {recall:.2f} %
        F1 Score   : {f1:.2f} %
        """
        self.update_nn_results()


     except Exception as e:
        messagebox.showerror("Error", f"LSTM training failed: {str(e)}")

    def update_nn_results(self):
       self.results_text.delete("1.0", tk.END)

       if self.cnn_results:
         self.results_text.insert(tk.END, self.cnn_results + "\n")

       if self.lstm_results:
         self.results_text.insert(tk.END, self.lstm_results + "\n")

       self.results_text.see(tk.END)

    # --- New Block for Threat Detection ---
    def detect_threat(self):
        if not self.trained_models or self.feature_extractor is None:
            messagebox.showwarning("Warning", "Please load a dataset and train at least one model first!")
            return

        # Simple way to select a trained model (you might want a dropdown in a real app)
        model_name = simpledialog.askstring("Model Selection", "Enter the name of the model to use (e.g., SVM, Random Forest):", parent=self.root)
        
        if model_name not in self.trained_models:
             best_model = max(self.results, key=lambda m: self.results[m]['f1']) if self.results else list(self.trained_models.keys())[0]
             use_model = messagebox.askyesno("Model Not Found", f"Model '{model_name}' not found. Use the best-performing model ({best_model})?")
             if use_model:
                 model_name = best_model
             else:
                 self.print_result("⚠️ Threat detection aborted by user.\n")
                 return

        model = self.trained_models.get(model_name)
        if model is None:
            messagebox.showerror("Error", "Selected model is not available.")
            return

        # Get the event profile string from the user
        event_profile = simpledialog.askstring(
            "Input Event Profile", 
            "Enter the event profile string (e.g., user_id timestamp source_ip dest_port bytes):", 
            parent=self.root
        )
        
        if event_profile:
            self.print_result(f"--- DETECTING THREAT using {model_name} ---\nInput: {event_profile}")
            try:
                # 1. Preprocess the new event profile (using the *fitted* TF-IDF vectorizer)
                # The input needs to be a list of documents (strings)
                new_doc = [event_profile]
                new_X = self.feature_extractor.transform(new_doc).toarray()
                new_X = np.array(new_X, dtype=np.float32)
                
                if model_name in ["CNN", "LSTM"]:
                    new_X = new_X.reshape(1, new_X.shape[1], 1)

                # 2. Predict the threat class
                prediction = model.predict(new_X)
                
                # 3. Decode the prediction back to the original label name
                if model_name in ["CNN", "LSTM"]:
                   pred_index = np.argmax(prediction, axis=1)[0]  # get the index of highest probability
                else:
                   pred_index = prediction[0]

                if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                   threat_label = self.label_encoder.inverse_transform([pred_index])[0]
                else:
                   threat_label = f"Class {pred_index}"

                
                # 4. Generate alert/result
                
                alert_status = "🚨 THREAT DETECTED!" if pred_index != 0 else "✅ No Threat Detected (Normal)"

                # You can customize the 'Normal' label based on your dataset's labels
                if self.label_names.size > 0:
                    # Assuming the 'Normal' label is one of the decoded labels, e.g. label_names[0]
                    # This logic should be adapted based on which label corresponds to 'Normal'
                    normal_label = next((l for l in self.label_names if 'normal' in str(l).lower()), self.label_names[0] if self.label_names.size > 0 else "Normal")
                    if threat_label != normal_label:
                        alert_status = f"🚨 THREAT DETECTED! Type: {threat_label.upper()}"
                    else:
                        alert_status = f"✅ No Threat Detected (Type: {normal_label})"
                        
                self.print_result(f"""
{'*'*70}
{alert_status}
{'*'*70}
Predicted Class (Internal ID): {pred_index}
Predicted Threat Label:        {threat_label}
Model Used:                    {model_name}
""")
                messagebox.showinfo("Detection Result", alert_status)
                
            except Exception as e:
                messagebox.showerror("Detection Error", f"Threat detection failed: {str(e)}")
                self.print_result(f"ERROR: Threat detection failed: {str(e)}\n")

    # --- End of New Block ---
    
    def calculate_metrics(self, model_name, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred) * 100
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0) * 100
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        result_str = f"""
{'='*70}
✅ {model_name.upper()} - Training Complete
{'='*70}
Accuracy:   {accuracy:7.2f}%
Precision:  {precision:7.2f}%
Recall:     {recall:7.2f}%
F1-Score:   {f1:7.2f}%
{'='*70}
"""
        self.print_result(result_str)
    
    def print_result(self, text):
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def show_comparison(self):
        if not self.results:
            messagebox.showwarning("Warning", "No models trained yet!")
            return
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()

        models = list(self.results.keys())
        accuracy = [self.results[m]['accuracy'] for m in models]
        precision = [self.results[m]['precision'] for m in models]
        recall = [self.results[m]['recall'] for m in models]
        f1 = [self.results[m]['f1'] for m in models]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        axes[0, 0].bar(models, accuracy, color='#3498db')
        axes[0, 0].set_title('Accuracy (%)')
        axes[0, 0].set_ylim([0, 100])
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(models, precision, color='#e74c3c')
        axes[0, 1].set_title('Precision (%)')
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(models, recall, color='#2ecc71')
        axes[1, 0].set_title('Recall (%)')
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(models, f1, color='#f39c12')
        axes[1, 1].set_title('F1-Score (%)')
        axes[1, 1].set_ylim([0, 100])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        #plt.show()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both' , expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = CyberThreatDashboard(root)
    root.mainloop()