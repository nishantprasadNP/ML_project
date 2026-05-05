import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Global variables to store the loaded dataset
current_df = None

# Global Tkinter variables for UI
knn_k_var = None
dt_depth_var = None
lr_fit_intercept_var = None
param_frame = None
dataset_var = None
algo_var = None
output_text = None
status_var = None
canvas = None
ax = None

def update_hyperparameters_ui(*args):
    """Updates the parameter frame based on the selected algorithm."""
    for widget in param_frame.winfo_children():
        widget.destroy()
        
    algo = algo_var.get()
    
    if algo == "Linear Regression":
        cb = tk.Checkbutton(param_frame, text="Fit Intercept", variable=lr_fit_intercept_var)
        cb.pack(anchor="w")
    elif algo == "KNN":
        lbl = tk.Label(param_frame, text="Number of Neighbors (k):")
        lbl.pack(anchor="w")
        slider = tk.Scale(param_frame, from_=1, to=20, orient="horizontal", variable=knn_k_var)
        slider.pack(fill="x")
    elif algo == "Decision Tree":
        lbl = tk.Label(param_frame, text="Max Depth (leave empty for None):")
        lbl.pack(anchor="w", pady=(0, 2))
        entry = tk.Entry(param_frame, textvariable=dt_depth_var)
        entry.pack(fill="x")

def load_dataset():
    global current_df
    dataset_choice = dataset_var.get()
    status_var.set("Loading dataset...")
    
    try:
        if dataset_choice == "Upload CSV...":
            file_path = fd.askopenfilename(
                title="Select a CSV file",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if file_path:
                current_df = pd.read_csv(file_path)
                output_text.delete("1.0", tk.END)
                output_text.insert(tk.END, f"Loaded CSV: {file_path}\n\n")
                output_text.insert(tk.END, "Preview (first 5 rows):\n")
                output_text.insert(tk.END, current_df.head().to_string() + "\n")
                status_var.set("CSV Dataset loaded successfully!")
            else:
                status_var.set("Dataset load cancelled.")
        else:
            if dataset_choice == "Iris":
                data = datasets.load_iris(as_frame=True)
            elif dataset_choice == "Breast Cancer":
                data = datasets.load_breast_cancer(as_frame=True)
            elif dataset_choice == "Wine":
                data = datasets.load_wine(as_frame=True)
            
            current_df = data.frame
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, f"Loaded built-in dataset: {dataset_choice}\n\n")
            output_text.insert(tk.END, "Preview (first 5 rows):\n")
            output_text.insert(tk.END, current_df.head().to_string() + "\n")
            status_var.set(f"Built-in '{dataset_choice}' dataset loaded successfully!")
            
    except Exception as e:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Error loading dataset: {e}\n")
        status_var.set("Error loading dataset!")

def train_model():
    algo = algo_var.get()
    status_var.set(f"Training {algo}...")
    output_text.insert(tk.END, f"\n--- Training {algo} ---\n")
    
    if current_df is None:
        output_text.insert(tk.END, "Please load a dataset first!\n")
        status_var.set("Error: No dataset loaded.")
        return
        
    try:
        X = current_df.iloc[:, :-1]
        y = current_df.iloc[:, -1]
        
        # Clear the plot area
        ax.clear()
        
        if algo == "Linear Regression":
            X_plot = X.iloc[:, 0:1]
            feature_name = X_plot.columns[0]
            
            X_train, X_test, y_train, y_test = train_test_split(X_plot, y, test_size=0.2, random_state=42)
            
            fit_intercept = lr_fit_intercept_var.get()
            model = LinearRegression(fit_intercept=fit_intercept)
            
            model.fit(X_train.values, y_train.values)
            predictions = model.predict(X_test.values)
            score = mean_squared_error(y_test, predictions)
            
            output_text.insert(tk.END, f"Note: Trained on first feature '{feature_name}' for visualization.\n")
            output_text.insert(tk.END, f"Mean Squared Error: {score:.4f}\n")
            
            ax.scatter(X_test.values, y_test.values, color='black', label="Test Data")
            sort_idx = np.argsort(X_test.values.flatten())
            ax.plot(X_test.values[sort_idx], predictions[sort_idx], color='blue', linewidth=3, label="Regression Line")
            ax.set_title("Linear Regression")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Target")
            ax.legend()
            
            status_var.set("Linear Regression Model Trained Successfully!")
            
        elif algo in ["KNN", "Decision Tree"]:
            if X.shape[1] >= 2:
                X_plot = X.iloc[:, 0:2]
            else:
                output_text.insert(tk.END, "Error: Classification plot requires at least 2 features.\n")
                status_var.set("Error: Need at least 2 features.")
                return
                
            feat1, feat2 = X_plot.columns[0], X_plot.columns[1]
            X_train, X_test, y_train, y_test = train_test_split(X_plot, y, test_size=0.2, random_state=42)
            
            if algo == "KNN":
                k = knn_k_var.get()
                model = KNeighborsClassifier(n_neighbors=k)
                title = f"KNN Boundary (k={k})"
            else:
                depth_str = dt_depth_var.get().strip()
                max_depth = None
                if depth_str:
                    try:
                        max_depth = int(depth_str)
                    except ValueError:
                        output_text.insert(tk.END, "Error: Max Depth must be an integer. Training aborted.\n")
                        status_var.set("Error: Invalid Max Depth.")
                        return
                model = DecisionTreeClassifier(max_depth=max_depth)
                title = f"Decision Tree Boundary (depth={max_depth})"
                
            model.fit(X_train.values, y_train.values)
            predictions = model.predict(X_test.values)
            score = accuracy_score(y_test, predictions)
            
            output_text.insert(tk.END, f"Note: Trained on features '{feat1}' & '{feat2}' for visualization.\n")
            output_text.insert(tk.END, f"Accuracy: {score:.4f} ({score*100:.2f}%)\n")
            
            x_min, x_max = X_plot.iloc[:, 0].min() - 1, X_plot.iloc[:, 0].max() + 1
            y_min, y_max = X_plot.iloc[:, 1].min() - 1, X_plot.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            
            mesh_data = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(mesh_data)
            Z = Z.reshape(xx.shape)
            
            unique_classes = np.unique(y)
            num_classes = len(unique_classes)
            if Z.dtype.kind in {'U', 'O'}:
                class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
                Z = np.vectorize(class_to_int.get)(Z)
                y_test_ints = np.vectorize(class_to_int.get)(y_test)
            else:
                y_test_ints = y_test
                
            cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF', '#AAFFFF'][:num_classes])
            cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'][:num_classes])
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
            ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test_ints, cmap=cmap_bold, edgecolor='k', s=40)
            ax.set_title(title)
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            
            status_var.set(f"{algo} Model Trained Successfully!")
            
        output_text.see(tk.END) # Auto-scroll to the bottom
        canvas.draw()
            
    except Exception as e:
        output_text.insert(tk.END, f"Error during training: {e}\n")
        status_var.set("Error during training!")

def main():
    global knn_k_var, dt_depth_var, lr_fit_intercept_var, param_frame, dataset_var, algo_var, output_text, status_var
    global canvas, ax
    
    root = tk.Tk()
    root.title("ML Visualizer Pro")
    root.geometry("1100x700") 
    
    # Create two panes: left for controls, right for plot
    left_frame = tk.Frame(root, width=400)
    left_frame.pack(side="left", fill="y", padx=15, pady=15)
    left_frame.pack_propagate(False) 
    
    right_frame = tk.Frame(root)
    right_frame.pack(side="right", fill="both", expand=True, padx=15, pady=15)
    
    # Initialize Matplotlib Figure and embed it
    fig, ax = plt.subplots(figsize=(6, 5))
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)
    
    # --- 1. Dataset Section ---
    dataset_frame = tk.LabelFrame(left_frame, text="1. Dataset Configuration", padx=10, pady=10, font=("Segoe UI", 9, "bold"))
    dataset_frame.pack(fill="x", pady=(0, 15))
    
    dataset_label = tk.Label(dataset_frame, text="Select Source:")
    dataset_label.pack(anchor="w", pady=(0, 5))
    
    dataset_var = tk.StringVar(value="Iris")
    dataset_dropdown = ttk.Combobox(
        dataset_frame, 
        textvariable=dataset_var, 
        values=["Iris", "Breast Cancer", "Wine", "Upload CSV..."],
        state="readonly"
    )
    dataset_dropdown.pack(fill="x", pady=(0, 10))
    
    load_btn = tk.Button(dataset_frame, text="Load Dataset", command=load_dataset, bg="#E0E0E0", cursor="hand2")
    load_btn.pack(fill="x")

    # --- 2. Model Section ---
    model_frame = tk.LabelFrame(left_frame, text="2. Model Configuration", padx=10, pady=10, font=("Segoe UI", 9, "bold"))
    model_frame.pack(fill="x", pady=(0, 15))
    
    algo_label = tk.Label(model_frame, text="Select Algorithm:")
    algo_label.pack(anchor="w", pady=(0, 5))
    
    algo_var = tk.StringVar(value="Linear Regression")
    algo_dropdown = ttk.Combobox(
        model_frame, 
        textvariable=algo_var, 
        values=["Linear Regression", "KNN", "Decision Tree"],
        state="readonly"
    )
    algo_dropdown.pack(fill="x", pady=(0, 10))
    
    algo_dropdown.bind("<<ComboboxSelected>>", update_hyperparameters_ui)
    
    # Initialize hyperparameter variables
    knn_k_var = tk.IntVar(value=5)
    dt_depth_var = tk.StringVar(value="")
    lr_fit_intercept_var = tk.BooleanVar(value=True)
    
    # Simple Frame for dynamic hyperparams
    param_frame = tk.Frame(model_frame)
    param_frame.pack(fill="x", pady=(5, 10))
    
    update_hyperparameters_ui()
    
    train_btn = tk.Button(model_frame, text="Train Model", command=train_model, bg="#4CAF50", fg="white", font=("Segoe UI", 10, "bold"), cursor="hand2")
    train_btn.pack(fill="x", pady=(5, 0))

    # --- 3. Output Section ---
    output_frame = tk.LabelFrame(left_frame, text="3. System Output", padx=10, pady=10, font=("Segoe UI", 9, "bold"))
    output_frame.pack(fill="both", expand=True, pady=(0, 10))
    
    output_text = tk.Text(output_frame, height=10, bg="#F8F9FA", font=("Consolas", 9), relief=tk.FLAT)
    output_text.pack(fill="both", expand=True)

    # --- 4. Status Bar ---
    status_var = tk.StringVar(value="Ready. Awaiting dataset load.")
    status_bar = tk.Label(left_frame, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor="w", fg="#555", font=("Segoe UI", 8))
    status_bar.pack(fill="x", side="bottom")

    root.mainloop()

if __name__ == "__main__":
    main()
