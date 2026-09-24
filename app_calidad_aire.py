import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class AirQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Ozono - Regresión Lineal, k-NN y PCA")
        self.root.geometry("1250x780")
        self.root.minsize(1050, 680)

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.mean_train = None
        self.std_train = None
        self.theta = None
        self.lineal_metrics = None
        self.knn_metrics = None
        self.pca_metrics = None
        self.best_k = None
        self.pred_lineal = None
        self.pred_knn = None
        self.pred_pca = None
        self.components = None
        self.varianza_acumulada = None
        self.n_components = None
        self.feature_names = ["NO2", "CO", "PM2.5", "PM10", "SO2"]

        self.create_interface()
        self.load_default_file()

    def create_interface(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        style.configure("Info.TLabel", font=("Arial", 10))

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Predicción de Ozono (O3)", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Cargar CSV", command=self.load_csv).pack(side="right", padx=5)
        ttk.Button(top, text="Ejecutar análisis", command=self.run_analysis).pack(side="right", padx=5)

        self.status = ttk.Label(
            top,
            text="Cargue el CSV o use el archivo por defecto.",
            style="Info.TLabel"
        )
        self.status.pack(side="right", padx=15)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.tab_data = ttk.Frame(self.notebook, padding=10)
        self.tab_metrics = ttk.Frame(self.notebook, padding=10)
        self.tab_graphs = ttk.Frame(self.notebook, padding=10)
        self.tab_pca = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_data, text="Datos")
        self.notebook.add(self.tab_metrics, text="Modelos y métricas")
        self.notebook.add(self.tab_graphs, text="Gráficas")
        self.notebook.add(self.tab_pca, text="PCA")

        self.create_data_tab()
        self.create_metrics_tab()
        self.create_graphs_tab()
        self.create_pca_tab()

    def create_data_tab(self):
        self.data_summary = tk.Text(self.tab_data, height=12, font=("Consolas", 10))
        self.data_summary.pack(fill="x", pady=(0, 10))

        frame = ttk.Frame(self.tab_data)
        frame.pack(fill="both", expand=True)

        columns = ("Date", "NO2", "CO", "PM2.5", "PM10", "SO2", "O3")
        self.data_table = ttk.Treeview(frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.data_table.heading(col, text=col)
            self.data_table.column(col, width=120, anchor="center")
        self.data_table.column("Date", width=130)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.data_table.yview)
        self.data_table.configure(yscrollcommand=scrollbar.set)
        self.data_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_metrics_tab(self):
        top = ttk.Frame(self.tab_metrics)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="k máximo a evaluar:").pack(side="left")
        self.k_limit = tk.IntVar(value=30)
        ttk.Spinbox(top, from_=2, to=60, textvariable=self.k_limit, width=8).pack(side="left", padx=5)
        ttk.Button(top, text="Actualizar análisis", command=self.run_analysis).pack(side="left", padx=5)

        columns = ("Modelo", "Configuración", "R²", "RMSE (ppm)", "MAE (ppm)")
        self.metrics_table = ttk.Treeview(self.tab_metrics, columns=columns, show="headings", height=8)
        for col in columns:
            self.metrics_table.heading(col, text=col)
            self.metrics_table.column(col, width=175, anchor="center")
        self.metrics_table.pack(fill="x", pady=10)

        self.explanation = tk.Text(self.tab_metrics, height=19, wrap="word", font=("Arial", 11))
        self.explanation.pack(fill="both", expand=True)

    def create_graphs_tab(self):
        controls = ttk.Frame(self.tab_graphs)
        controls.pack(fill="x")

        ttk.Button(controls, text="Serie temporal O3", command=self.plot_time_series).pack(side="left", padx=4)
        ttk.Button(controls, text="Correlación", command=self.plot_correlation).pack(side="left", padx=4)
        ttk.Button(controls, text="Real vs. predicho", command=self.plot_real_vs_pred).pack(side="left", padx=4)
        ttk.Button(controls, text="Residuos", command=self.plot_residuals).pack(side="left", padx=4)
        ttk.Button(controls, text="Comparación de k", command=self.plot_k_results).pack(side="left", padx=4)

        self.graph_frame = ttk.Frame(self.tab_graphs)
        self.graph_frame.pack(fill="both", expand=True, pady=10)
        self.canvas = None

    def create_pca_tab(self):
        self.pca_summary = tk.Text(self.tab_pca, height=11, wrap="word", font=("Arial", 11))
        self.pca_summary.pack(fill="x", pady=(0, 10))

        self.pca_graph_frame = ttk.Frame(self.tab_pca)
        self.pca_graph_frame.pack(fill="both", expand=True)
        self.pca_canvas = None

    def load_default_file(self):
        default = "epa_air_quality_10730023_2025_clean.csv"
        if os.path.exists(default):
            self.load_data(default)

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("Archivos CSV", "*.csv")]
        )
        if path:
            self.load_data(path)

    def load_data(self, path):
        try:
            df = pd.read_csv(path)
            required = ["Date"] + self.feature_names + ["O3"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError("Faltan columnas: " + ", ".join(missing))

            df["Date"] = pd.to_datetime(df["Date"])
            self.df = df
            self.status.config(text=f"Archivo cargado: {os.path.basename(path)}")
            self.show_data()
            self.run_analysis()
        except Exception as error:
            messagebox.showerror("Error al cargar", str(error))

    def show_data(self):
        if self.df is None:
            return

        clean = self.df.dropna().copy()
        summary = (
            f"Filas originales: {len(self.df)}\n"
            f"Filas completas para modelar: {len(clean)}\n"
            f"Filas eliminadas por valores faltantes: {len(self.df) - len(clean)}\n\n"
            f"Valores faltantes por columna:\n{self.df.isna().sum().to_string()}"
        )
        self.data_summary.delete("1.0", tk.END)
        self.data_summary.insert(tk.END, summary)

        for item in self.data_table.get_children():
            self.data_table.delete(item)

        for _, row in self.df.head(100).iterrows():
            values = [row["Date"].strftime("%Y-%m-%d")]
            for col in self.feature_names + ["O3"]:
                value = row[col]
                values.append("" if pd.isna(value) else f"{value:.4f}")
            self.data_table.insert("", tk.END, values=values)

    def split_and_scale(self, X, y):
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        n_test = int(len(X) * 0.20)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        self.X_train = X[train_indices]
        self.X_test = X[test_indices]
        self.y_train = y[train_indices]
        self.y_test = y[test_indices]

        self.mean_train = np.mean(self.X_train, axis=0)
        self.std_train = np.std(self.X_train, axis=0)
        self.std_train[self.std_train == 0] = 1

        self.X_train_scaled = (self.X_train - self.mean_train) / self.std_train
        self.X_test_scaled = (self.X_test - self.mean_train) / self.std_train

    @staticmethod
    def metrics(y_real, y_pred):
        residuals = y_real - y_pred
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        r2 = 1 - np.sum(residuals ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)
        return r2, rmse, mae

    def linear_regression_manual(self):
        train_bias = np.column_stack((np.ones(len(self.X_train_scaled)), self.X_train_scaled))
        test_bias = np.column_stack((np.ones(len(self.X_test_scaled)), self.X_test_scaled))
        self.theta = np.linalg.pinv(train_bias.T @ train_bias) @ train_bias.T @ self.y_train
        self.pred_lineal = test_bias @ self.theta
        return self.metrics(self.y_test, self.pred_lineal)

    @staticmethod
    def knn_predict(X_train, y_train, X_test, k):
        predictions = []
        for point in X_test:
            distances = np.sqrt(np.sum((X_train - point) ** 2, axis=1))
            neighbors = np.argsort(distances)[:k]
            predictions.append(np.mean(y_train[neighbors]))
        return np.array(predictions)

    def pca_manual(self):
        covariance = np.cov(self.X_train_scaled, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        explained = eigenvalues / np.sum(eigenvalues)
        cumulative = np.cumsum(explained)
        self.n_components = np.argmax(cumulative >= 0.95) + 1
        self.components = eigenvectors[:, :self.n_components]
        self.varianza_acumulada = cumulative

        train_pca = self.X_train_scaled @ self.components
        test_pca = self.X_test_scaled @ self.components
        self.pred_pca = self.knn_predict(train_pca, self.y_train, test_pca, self.best_k)
        return self.metrics(self.y_test, self.pred_pca), explained

    def run_analysis(self):
        if self.df is None:
            return
        try:
            clean = self.df.dropna().copy()
            X = clean[self.feature_names].to_numpy(dtype=float)
            y = clean["O3"].to_numpy(dtype=float)
            self.split_and_scale(X, y)

            self.lineal_metrics = self.linear_regression_manual()

            all_results = []
            limit = min(self.k_limit.get(), len(self.X_train))
            for k in range(1, limit + 1):
                pred = self.knn_predict(self.X_train_scaled, self.y_train, self.X_test_scaled, k)
                r2, rmse, mae = self.metrics(self.y_test, pred)
                all_results.append((k, r2, rmse, mae))

            self.knn_results = pd.DataFrame(all_results, columns=["k", "R2", "RMSE", "MAE"])
            best = self.knn_results.sort_values("RMSE").iloc[0]
            self.best_k = int(best["k"])
            self.pred_knn = self.knn_predict(
                self.X_train_scaled, self.y_train, self.X_test_scaled, self.best_k
            )
            self.knn_metrics = self.metrics(self.y_test, self.pred_knn)

            self.pca_metrics, self.explained = self.pca_manual()
            self.update_metrics()
            self.update_pca_tab()
            self.status.config(
                text=f"Análisis listo: {len(clean)} filas, mejor k={self.best_k}"
            )
            self.plot_real_vs_pred()
        except Exception as error:
            messagebox.showerror("Error en el análisis", str(error))

    def update_metrics(self):
        for item in self.metrics_table.get_children():
            self.metrics_table.delete(item)

        rows = [
            ("Regresión lineal", "Ecuación normal", *self.lineal_metrics),
            ("k-NN", f"k={self.best_k}", *self.knn_metrics),
            ("k-NN con PCA", f"{self.n_components} componentes", *self.pca_metrics),
        ]
        for model, config, r2, rmse, mae in rows:
            self.metrics_table.insert(
                "", tk.END,
                values=(model, config, f"{r2:.6f}", f"{rmse:.6f}", f"{mae:.6f}")
            )

        r2_l, rmse_l, mae_l = self.lineal_metrics
        r2_k, rmse_k, mae_k = self.knn_metrics
        r2_p, rmse_p, mae_p = self.pca_metrics

        text = (
            "Interpretación de las métricas\n\n"
            "R² indica qué proporción de la variación de O3 explica el modelo. "
            "Un valor mayor es mejor; 1 representa predicción perfecta y 0 equivale a predecir siempre el promedio.\n\n"
            "RMSE mide el tamaño típico del error y penaliza con mayor fuerza los errores grandes. "
            "MAE representa el error absoluto promedio. Ambos se expresan en ppm y valores menores son mejores.\n\n"
            f"Regresión lineal: R²={r2_l:.4f}, RMSE={rmse_l:.5f}, MAE={mae_l:.5f}.\n"
            f"k-NN con k={self.best_k}: R²={r2_k:.4f}, RMSE={rmse_k:.5f}, MAE={mae_k:.5f}.\n"
            f"k-NN con PCA: R²={r2_p:.4f}, RMSE={rmse_p:.5f}, MAE={mae_p:.5f}.\n\n"
            "La regresión lineal se calculó con la ecuación normal. k-NN usa distancia euclidiana "
            "y el promedio de los vecinos más cercanos. PCA se calculó con covarianza, autovalores y autovectores."
        )
        self.explanation.delete("1.0", tk.END)
        self.explanation.insert(tk.END, text)

    def update_pca_tab(self):
        if self.n_components is None:
            return
        text = (
            f"PCA redujo {len(self.feature_names)} predictores a {self.n_components} componentes principales.\n\n"
            "Varianza explicada por componente:\n" +
            "\n".join(
                f"PC{i + 1}: {value * 100:.2f}% | acumulada: {self.varianza_acumulada[i] * 100:.2f}%"
                for i, value in enumerate(self.explained)
            )
        )
        self.pca_summary.delete("1.0", tk.END)
        self.pca_summary.insert(tk.END, text)

        self.clear_pca_graph()
        figure, ax = plt.subplots(figsize=(8, 4.5))
        positions = np.arange(1, len(self.explained) + 1)
        ax.bar(positions, self.explained * 100, label="Varianza individual")
        ax.plot(positions, self.varianza_acumulada * 100, marker="o", color="red", label="Varianza acumulada")
        ax.axhline(95, color="green", linestyle="--", label="Meta 95%")
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Varianza explicada (%)")
        ax.set_title("Varianza explicada por PCA")
        ax.set_xticks(positions)
        ax.legend()
        ax.grid(alpha=0.3)
        self.pca_canvas = FigureCanvasTkAgg(figure, master=self.pca_graph_frame)
        self.pca_canvas.draw()
        self.pca_canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear_graph(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def clear_pca_graph(self):
        if self.pca_canvas is not None:
            self.pca_canvas.get_tk_widget().destroy()
            self.pca_canvas = None

    def show_figure(self, figure):
        self.clear_graph()
        self.canvas = FigureCanvasTkAgg(figure, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_time_series(self):
        if self.df is None:
            return
        figure, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df["Date"], self.df["O3"], color="green")
        ax.set_title("Concentración diaria de ozono (O3) durante 2025")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("O3 (ppm)")
        ax.grid(alpha=0.3)
        self.show_figure(figure)

    def plot_correlation(self):
        if self.df is None:
            return
        clean = self.df.dropna()
        correlation = clean[self.feature_names + ["O3"]].corr()
        figure, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        labels = self.feature_names + ["O3"]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if abs(correlation.iloc[i, j]) > 0.55 else "black"
                ax.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center", color=color)
        figure.colorbar(image, ax=ax)
        ax.set_title("Matriz de correlación entre contaminantes")
        self.show_figure(figure)

    def plot_real_vs_pred(self):
        if self.pred_knn is None:
            return
        figure, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self.y_test, self.pred_knn, alpha=0.75, edgecolor="black")
        lower = min(self.y_test.min(), self.pred_knn.min())
        upper = max(self.y_test.max(), self.pred_knn.max())
        ax.plot([lower, upper], [lower, upper], "r--", label="Predicción perfecta")
        ax.set_xlabel("O3 real (ppm)")
        ax.set_ylabel("O3 predicho (ppm)")
        ax.set_title(f"O3 real vs. O3 predicho — k-NN manual (k={self.best_k})")
        ax.legend()
        ax.grid(alpha=0.3)
        self.show_figure(figure)

    def plot_residuals(self):
        if self.pred_knn is None:
            return
        residuals = self.y_test - self.pred_knn
        figure, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.pred_knn, residuals, color="darkorange", alpha=0.75, edgecolor="black")
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("O3 predicho (ppm)")
        ax.set_ylabel("Residuo: O3 real - O3 predicho")
        ax.set_title(f"Residuos — k-NN manual (k={self.best_k})")
        ax.grid(alpha=0.3)
        self.show_figure(figure)

    def plot_k_results(self):
        if not hasattr(self, "knn_results"):
            return
        figure, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(self.knn_results["k"], self.knn_results["RMSE"], marker="o", label="RMSE")
        ax1.plot(self.knn_results["k"], self.knn_results["MAE"], marker="s", label="MAE")
        ax1.set_xlabel("Número de vecinos k")
        ax1.set_ylabel("Error (ppm)")
        ax1.axvline(self.best_k, color="red", linestyle="--", label=f"Mejor k={self.best_k}")
        ax1.set_title("Errores de k-NN según el valor de k")
        ax1.grid(alpha=0.3)
        ax1.legend()
        self.show_figure(figure)


if __name__ == "__main__":
    root = tk.Tk()
    app = AirQualityApp(root)
    root.mainloop()
