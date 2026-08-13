import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from perceptronModel import Perceptron
from Optimizer import SimulatedAnnealingOptimizer
from dataLoad import (
    loadTrainingData, normalizeFeatures, split_data,
    preparePlants, countPredicted, getNeedy, REQUIRED_COLUMNS
)


# =========================
# GUI
# =========================
class PlantWateringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Plant Watering Scheduler")
        self.root.geometry("1100x760")
        self.root.configure(bg="#f4f6f8")

        self.file_path = ""
        self.df = None
        self.model = None
        self.plants = []
        self.optimizer = None
        self.train_acc = 0
        self.test_acc = 0

        title = tk.Label(
            root,
            text="AI Smart Plant Watering Scheduler",
            font=("Arial", 18, "bold"),
            bg="#f4f6f8"
        )
        title.pack(pady=10)

        top_frame = tk.Frame(root, bg="#f4f6f8")
        top_frame.pack(pady=5)

        tk.Button(
            top_frame,
            text="Choose Excel File",
            width=18,
            command=self.choose_file
        ).grid(row=0, column=0, padx=5)

        self.file_label = tk.Label(
            top_frame,
            text="No file selected",
            width=55,
            anchor="w",
            bg="#f4f6f8"
        )
        self.file_label.grid(row=0, column=1, padx=5)

        tk.Label(
            top_frame,
            text="Train Ratio (0-1):",
            bg="#f4f6f8",
            font=("Arial", 11)
        ).grid(row=0, column=2, padx=5)

        self.ratio_entry = tk.Entry(top_frame, width=6, font=("Arial", 11))
        self.ratio_entry.insert(0, "0.7")
        self.ratio_entry.grid(row=0, column=3, padx=5)

        tk.Button(
            top_frame,
            text="Train Model",
            width=18,
            command=self.train_model
        ).grid(row=0, column=4, padx=5)

        info_frame = tk.Frame(root, bg="#f4f6f8")
        info_frame.pack(pady=10)

        self.train_label = tk.Label(info_frame, text="Train Accuracy: -", font=("Arial", 11), bg="#f4f6f8")
        self.train_label.grid(row=0, column=0, padx=15)

        self.test_label = tk.Label(info_frame, text="Test Accuracy: -", font=("Arial", 11), bg="#f4f6f8")
        self.test_label.grid(row=0, column=1, padx=15)

        self.suggested_label = tk.Label(
            info_frame,
            text="Suggested number of plants to water: -",
            font=("Arial", 11),
            bg="#f4f6f8"
        )
        self.suggested_label.grid(row=0, column=2, padx=15)

        input_frame = tk.Frame(root, bg="#f4f6f8")
        input_frame.pack(pady=5)

        tk.Label(
            input_frame,
            text="Number of plants to include:",
            bg="#f4f6f8",
            font=("Arial", 11)
        ).grid(row=0, column=0, padx=5)

        self.count_entry = tk.Entry(input_frame, width=10, font=("Arial", 11))
        self.count_entry.grid(row=0, column=1, padx=5)

        tk.Button(
            input_frame,
            text="Run Optimization",
            width=18,
            command=self.run_optimization
        ).grid(row=0, column=2, padx=10)

        tk.Button(
            input_frame,
            text="Show Garden",
            width=14,
            command=self.show_garden
        ).grid(row=0, column=3, padx=5)

        tk.Button(
            input_frame,
            text="Learning Curve",
            width=16,
            command=self.draw_learning_curve
        ).grid(row=0, column=4, padx=5)

        test_frame = tk.Frame(root, bg="#f4f6f8")
        test_frame.pack(pady=5)

        tk.Label(test_frame, text="Test plant:", bg="#f4f6f8", font=("Arial", 11, "bold")).grid(row=0, column=0, padx=5)
        tk.Label(test_frame, text="Soil", bg="#f4f6f8", font=("Arial", 10)).grid(row=0, column=1, padx=3)
        self.test_soil_entry = tk.Entry(test_frame, width=7, font=("Arial", 10))
        self.test_soil_entry.grid(row=0, column=2, padx=3)

        tk.Label(test_frame, text="Last watered", bg="#f4f6f8", font=("Arial", 10)).grid(row=0, column=3, padx=3)
        self.test_last_entry = tk.Entry(test_frame, width=7, font=("Arial", 10))
        self.test_last_entry.grid(row=0, column=4, padx=3)

        tk.Label(test_frame, text="Type", bg="#f4f6f8", font=("Arial", 10)).grid(row=0, column=5, padx=3)
        self.test_type_entry = tk.Entry(test_frame, width=7, font=("Arial", 10))
        self.test_type_entry.grid(row=0, column=6, padx=3)

        tk.Button(
            test_frame,
            text="Test Perceptron",
            width=16,
            command=self.test_perceptron
        ).grid(row=0, column=7, padx=8)

        add_frame = tk.Frame(root, bg="#f4f6f8")
        add_frame.pack(pady=5)

        tk.Label(add_frame, text="Add plant:", bg="#f4f6f8", font=("Arial", 11, "bold")).grid(row=0, column=0, padx=5)
        tk.Label(add_frame, text="Soil", bg="#f4f6f8", font=("Arial", 10)).grid(row=0, column=1, padx=3)
        self.add_soil_entry = tk.Entry(add_frame, width=7, font=("Arial", 10))
        self.add_soil_entry.grid(row=0, column=2, padx=3)

        tk.Label(add_frame, text="Last", bg="#f4f6f8", font=("Arial", 10)).grid(row=0, column=3, padx=3)
        self.add_last_entry = tk.Entry(add_frame, width=7, font=("Arial", 10))
        self.add_last_entry.grid(row=0, column=4, padx=3)

        tk.Label(add_frame, text="Type", bg="#f4f6f8", font=("Arial", 10)).grid(row=0, column=5, padx=3)
        self.add_type_entry = tk.Entry(add_frame, width=7, font=("Arial", 10))
        self.add_type_entry.grid(row=0, column=6, padx=3)

        tk.Label(add_frame, text="X", bg="#f4f6f8", font=("Arial", 10)).grid(row=0, column=7, padx=3)
        self.add_x_entry = tk.Entry(add_frame, width=7, font=("Arial", 10))
        self.add_x_entry.grid(row=0, column=8, padx=3)

        tk.Label(add_frame, text="Y", bg="#f4f6f8", font=("Arial", 10)).grid(row=0, column=9, padx=3)
        self.add_y_entry = tk.Entry(add_frame, width=7, font=("Arial", 10))
        self.add_y_entry.grid(row=0, column=10, padx=3)

        tk.Button(
            add_frame,
            text="Add Plant",
            width=13,
            command=self.add_plant
        ).grid(row=0, column=11, padx=8)

        middle_frame = tk.Frame(root, bg="#f4f6f8")
        middle_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            middle_frame, width=430, height=420, bg="white",
            highlightthickness=1, highlightbackground="#cccccc"
        )
        self.canvas.pack(side="left", padx=10, pady=10)

        self.output_box = ScrolledText(middle_frame, width=60, height=25, font=("Courier New", 10))
        self.output_box.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        legend = tk.Label(
            root,
            text=("Legend: Red = Needs Water | Green = No Water Needed | "
                  "Blue = Selected for watering"),
            bg="#f4f6f8",
            font=("Arial", 10, "italic")
        )
        legend.pack(pady=5)

    def choose_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if path:
            self.file_path = path
            self.file_label.config(text=path)

    def show_garden(self):
        if not self.plants:
            messagebox.showerror("Error", "Please train the model first.")
            return
        self.draw_plants(self.plants)

    def draw_learning_curve(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return

        losses = self.model.loss_history
        accuracies = self.model.accuracy_history
        if not losses or not accuracies:
            messagebox.showerror("Error", "No learning history is available.")
            return

        self.canvas.delete("all")
        margin = 45
        canvas_w = 430
        canvas_h = 420
        plot_w = canvas_w - (2 * margin)
        plot_h = canvas_h - (2 * margin)

        self.canvas.create_text(canvas_w / 2, 18, text="Perceptron Learning Curve", font=("Arial", 12, "bold"))
        self.canvas.create_line(margin, canvas_h - margin, canvas_w - margin, canvas_h - margin, fill="black")
        self.canvas.create_line(margin, margin, margin, canvas_h - margin, fill="black")

        max_loss = max(max(losses), 1)
        epochs = len(losses)

        loss_points = []
        accuracy_points = []
        for i in range(epochs):
            x = margin + (i / max(epochs - 1, 1)) * plot_w
            loss_y = canvas_h - margin - (losses[i] / max_loss) * plot_h
            accuracy_y = canvas_h - margin - accuracies[i] * plot_h
            loss_points.extend([x, loss_y])
            accuracy_points.extend([x, accuracy_y])

        if len(loss_points) >= 4:
            self.canvas.create_line(*loss_points, fill="red", width=2)
            self.canvas.create_line(*accuracy_points, fill="blue", width=2)

        self.canvas.create_text(margin + 75, margin - 18, text="Red: loss", fill="red", font=("Arial", 9, "bold"))
        self.canvas.create_text(margin + 190, margin - 18, text="Blue: accuracy", fill="blue", font=("Arial", 9, "bold"))
        self.canvas.create_text(margin - 20, canvas_h - margin, text="0", font=("Arial", 8))
        self.canvas.create_text(canvas_w - margin, canvas_h - margin + 15, text=str(epochs), font=("Arial", 8))

    def draw_plants(self, plants, solution=None):
        self.canvas.delete("all")

        if not plants:
            return

        margin = 30
        canvas_w = 430
        canvas_h = 420

        xs = [p["x"] for p in plants]
        ys = [p["y"] for p in plants]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        range_x = max(max_x - min_x, 1)
        range_y = max(max_y - min_y, 1)

        selected_ids = set()
        if solution:
            selected_ids = {p["id"] for p in solution}

        scaled_positions = {}

        for plant in plants:
            sx = margin + ((plant["x"] - min_x) / range_x) * (canvas_w - 2 * margin)
            sy = margin + ((plant["y"] - min_y) / range_y) * (canvas_h - 2 * margin)

            scaled_positions[plant["id"]] = (sx, sy)

            # Color logic: selected plants override base color.
            if plant["id"] in selected_ids:
                color = "blue"
            else:
                color = "red" if plant["predicted"] == 1 else "green"

            r = 8
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=color, outline="black")
            self.canvas.create_text(sx, sy - 14, text=str(plant["id"]), font=("Arial", 8, "bold"))

        if solution and len(solution) >= 2:
            for i in range(len(solution) - 1):
                x1, y1 = scaled_positions[solution[i]["id"]]
                x2, y2 = scaled_positions[solution[i + 1]["id"]]
                self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

    def test_perceptron(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return

        try:
            soil_moisture = float(self.test_soil_entry.get())
            last_watered = float(self.test_last_entry.get())
            plant_type = int(self.test_type_entry.get())

            if not (0 <= soil_moisture <= 100):
                raise ValueError("Soil moisture must be between 0 and 100.")
            if not (0 <= last_watered <= 48):
                raise ValueError("Last watered must be between 0 and 48 hours.")
            if plant_type not in (0, 1, 2):
                raise ValueError("Plant type must be 0, 1, or 2.")

            features = normalizeFeatures([[soil_moisture, last_watered, plant_type]])[0]
            prediction = self.model.predictSingle(features)
            label = "Needs water" if prediction == 1 else "Does not need water"

            self.output_box.insert(tk.END, "\n=== PERCEPTRON TEST ===\n")
            self.output_box.insert(
                tk.END,
                f"Soil={soil_moisture}, Last watered={last_watered}, Type={plant_type} -> {label}\n"
            )
            self.output_box.see(tk.END)

        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def add_plant(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return

        try:
            soil_moisture = float(self.add_soil_entry.get())
            last_watered = float(self.add_last_entry.get())
            plant_type = int(self.add_type_entry.get())
            x = float(self.add_x_entry.get())
            y = float(self.add_y_entry.get())

            if not (0 <= soil_moisture <= 100):
                raise ValueError("Soil moisture must be between 0 and 100.")
            if not (0 <= last_watered <= 48):
                raise ValueError("Last watered must be between 0 and 48 hours.")
            if plant_type not in (0, 1, 2):
                raise ValueError("Plant type must be 0, 1, or 2.")

            features = normalizeFeatures([[soil_moisture, last_watered, plant_type]])[0]
            predicted = self.model.predictSingle(features)
            plant = {
                "id": len(self.plants) + 1,
                "soil_moisture": soil_moisture,
                "last_watered": last_watered,
                "plant_type": plant_type,
                "x": x,
                "y": y,
                "predicted": predicted
            }

            self.plants.append(plant)
            predicted_count = countPredicted(self.plants)
            self.suggested_label.config(text=f"Suggested number of plants to water: {predicted_count}")
            self.draw_plants(self.plants)

            label = "needs water" if predicted == 1 else "no water"
            self.output_box.insert(tk.END, "\n=== PLANT ADDED ===\n")
            self.output_box.insert(
                tk.END,
                f"Plant {plant['id']}: soil={soil_moisture}, last={last_watered}, "
                f"type={plant_type}, pos=({x}, {y}), prediction={label}\n"
            )
            self.output_box.see(tk.END)

        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def insert_learning_history(self):
        self.output_box.insert(tk.END, "\n=== PERCEPTRON LEARNING PROCESS ===\n")
        self.output_box.insert(tk.END, "Epoch | Loss | Training Accuracy\n")

        total_epochs = len(self.model.loss_history)
        for i, (loss, accuracy) in enumerate(
            zip(self.model.loss_history, self.model.accuracy_history),
            start=1
        ):
            if i <= 10 or i == total_epochs or i % 10 == 0:
                self.output_box.insert(
                    tk.END,
                    f"{i:>5} | {loss:>4} | {round(accuracy * 100, 2):>6}%\n"
                )

    def insert_plant_features(self):
        self.output_box.insert(tk.END, "\n=== PLANTS AND FEATURES ===\n")
        for plant in self.plants:
            label = "needs water" if plant["predicted"] == 1 else "no water"
            self.output_box.insert(
                tk.END,
                f"Plant {plant['id']}: soil={plant['soil_moisture']}, "
                f"last={plant['last_watered']}, type={plant['plant_type']}, "
                f"pos=({plant['x']}, {plant['y']}), prediction={label}\n"
            )

    def insert_sa_history(self):
        self.output_box.insert(tk.END, "\n=== SA OPTIMIZATION STEPS ===\n")
        self.output_box.insert(tk.END, "Step | Temp | Current Cost | Best Cost\n")

        history = self.optimizer.history
        total_steps = len(history)
        for i, item in enumerate(history):
            should_show = i < 10 or i >= total_steps - 5 or i % 25 == 0
            if should_show:
                self.output_box.insert(
                    tk.END,
                    f"{item['step']:>4} | {item['temperature']:>6.3f} | "
                    f"{item['currentCost']:>12.2f} | {item['bestCost']:>9.2f}\n"
                )

    def train_model(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please choose an Excel file first.")
            return

        try:
            X, y, df = loadTrainingData(self.file_path)

            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    messagebox.showerror("Error", f"Missing column in Excel file: {col}")
                    return

            train_ratio = float(self.ratio_entry.get())
            if not (0 < train_ratio < 1):
                messagebox.showerror("Error", "Train ratio must be between 0 and 1.")
                return

            X = normalizeFeatures(X)
            X_train, y_train, X_test, y_test = split_data(X, y, train_ratio)

            self.model = Perceptron()
            self.model.fit(X_train, y_train)

            self.train_acc = self.model.evaluate(X_train, y_train)
            self.test_acc = self.model.evaluate(X_test, y_test)

            self.df = df
            self.plants = preparePlants(df, self.model)
            predicted_count = countPredicted(self.plants)

            self.train_label.config(text=f"Train Accuracy: {round(self.train_acc * 100, 2)}%")
            self.test_label.config(text=f"Test Accuracy: {round(self.test_acc * 100, 2)}%")
            self.suggested_label.config(text=f"Suggested number of plants to water: {predicted_count}")

            self.draw_plants(self.plants)

            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, "=== MODEL TRAINED SUCCESSFULLY ===\n\n")
            self.output_box.insert(tk.END, f"Weights: {self.model.weights}\n")
            self.output_box.insert(tk.END, f"Threshold: {self.model.threshold}\n")
            self.output_box.insert(tk.END, f"Train Accuracy: {round(self.train_acc * 100, 2)}%\n")
            self.output_box.insert(tk.END, f"Test Accuracy: {round(self.test_acc * 100, 2)}%\n")
            self.output_box.insert(tk.END, f"Predicted plants needing water: {predicted_count}\n")
            self.output_box.insert(tk.END, f"Total plants in dataset: {len(self.plants)}\n")
            self.insert_learning_history()
            self.insert_plant_features()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_optimization(self):
        if self.model is None or not self.plants:
            messagebox.showerror("Error", "Please train the model first.")
            return

        try:
            try:
                selected_count = int(self.count_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer.")
                return

            if selected_count <= 0 or selected_count > len(self.plants):
                messagebox.showerror(
                    "Error",
                    f"Number must be between 1 and {len(self.plants)} (total plants)."
                )
                return

            needy_plants = getNeedy(self.plants)
            if not needy_plants:
                messagebox.showerror("Error", "The model did not predict any plants that need water.")
                return

            self.optimizer = SimulatedAnnealingOptimizer(plants=self.plants)

            # Step 1: Random selection from ALL plants
            initial_sol, initial_cost = self.optimizer.get_initial_solution(selected_count)
            non_needy_in_sol = [p for p in initial_sol if p.get("predicted", 0) == 0]

            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, "=== INITIAL RANDOM SELECTION ===\n")
            self.output_box.insert(
                tk.END,
                f"Randomly selected {selected_count} plants from all {len(self.plants)}:\n"
            )
            for p in initial_sol:
                tag = "needs water" if p["predicted"] == 1 else "NO water needed"
                self.output_box.insert(
                    tk.END, f"  Plant {p['id']} at ({p['x']}, {p['y']}) - {tag}\n"
                )
            self.output_box.insert(tk.END, f"\nInitial Cost: {round(initial_cost, 2)}\n")
            self.output_box.insert(
                tk.END, f"Non-needy plants in selection: {len(non_needy_in_sol)}\n"
            )
            self.draw_plants(self.plants, initial_sol)
            self.output_box.see(tk.END)

            # Step 2: Ask about swap if any non-needy plants are in the selection
            current_sol = initial_sol[:]
            if non_needy_in_sol:
                do_swap = messagebox.askyesno(
                    "Swap Plants?",
                    f"Your selection includes {len(non_needy_in_sol)} plant(s) that do NOT need water.\n\n"
                    "Do you want to swap them with the closest needy plant(s)?"
                )
                if do_swap:
                    current_sol = self.optimizer.replace_with_closest(initial_sol)
                    swapped_cost = self.optimizer.costFun(current_sol)
                    self.output_box.insert(tk.END, "\n=== AFTER SWAP (replaced with closest needy) ===\n")
                    for p in current_sol:
                        tag = "needs water" if p["predicted"] == 1 else "NO water needed"
                        self.output_box.insert(
                            tk.END, f"  Plant {p['id']} at ({p['x']}, {p['y']}) - {tag}\n"
                        )
                    self.output_box.insert(tk.END, f"Cost after swap: {round(swapped_cost, 2)}\n")
                    self.draw_plants(self.plants, current_sol)
                    self.output_box.see(tk.END)

            # Step 3: Run SA on the final selection
            best_solution, best_cost = self.optimizer.run_sa(current_sol)
            total_distance = self.optimizer.total_distance(best_solution)

            selected_ids = {p["id"] for p in best_solution}
            missed = sum(1 for p in needy_plants if p["id"] not in selected_ids)
            extra = sum(1 for p in best_solution if p["predicted"] == 0)

            self.draw_plants(self.plants, best_solution)

            self.output_box.insert(tk.END, "\n=== PERCEPTRON RESULTS ===\n")
            self.output_box.insert(tk.END, f"Weights: {self.model.weights}\n")
            self.output_box.insert(tk.END, f"Threshold: {self.model.threshold}\n")
            self.output_box.insert(tk.END, f"Train Accuracy: {round(self.train_acc * 100, 2)}%\n")
            self.output_box.insert(tk.END, f"Test Accuracy: {round(self.test_acc * 100, 2)}%\n\n")
            self.output_box.insert(tk.END, f"Predicted plants needing water: {len(needy_plants)}\n")
            self.output_box.insert(tk.END, f"User requested count: {selected_count}\n\n")

            self.output_box.insert(tk.END, "=== FINAL COST BREAKDOWN ===\n")
            self.output_box.insert(tk.END, f"Plants missed (needy but skipped): {missed}\n")
            self.output_box.insert(tk.END, f"Extra watering (not needy but watered): {extra}\n")
            self.output_box.insert(tk.END, f"Total distance: {round(total_distance, 2)}\n")
            self.output_box.insert(tk.END, f"Best Cost: {round(best_cost, 2)}\n\n")
            self.insert_sa_history()
            self.output_box.insert(tk.END, "\n")

            self.output_box.insert(tk.END, "=== OPTIMIZED WATERING ORDER ===\n")
            for i, plant in enumerate(best_solution, start=1):
                tag = "NEEDS WATER" if plant["predicted"] == 1 else "EXTRA"
                self.output_box.insert(
                    tk.END,
                    f"{i}. Plant {plant['id']} at ({plant['x']}, {plant['y']}) - {tag}\n"
                )
            self.output_box.see(tk.END)

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = PlantWateringGUI(root)
    root.mainloop()
