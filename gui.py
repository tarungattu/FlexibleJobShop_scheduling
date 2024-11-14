import tkinter as tk
from tkinter import ttk
from FlexibleJobShopScheduler import FlexibleJobShopScheduler
import benchmarks
import distances
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Initialize main application window
root = tk.Tk()
root.title("Flexible Job Shop Scheduling GUI")
root.geometry("600x500")  # Adjusted to accommodate the new input field
root.config(bg="#f5f5f5")

# Global dictionary to hold GA parameters
ga_params = {
    "num_amrs": 0,
    "num_machines": 0,
    "num_jobs": 0,
    "population_size": 0,
    "crossover_rate": 0.7,
    "mutation_rate": 0.5,
    "max_generations": 0,
    "machine_data": []  # New field for storing machine_data
}

# Toggle variables for each setting
activate_termination = tk.IntVar(value=1)
enable_travel_time = tk.IntVar(value=1)
display_convergence = tk.IntVar()
display_schedule = tk.IntVar(value=1)
create_txt_file = tk.IntVar()
update_json_file = tk.IntVar()
machine_restrictions = tk.IntVar()

# Create frames for layout
left_frame = tk.Frame(root, bg="#f5f5f5", padx=10, pady=10)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

right_frame = tk.Frame(root, bg="#f5f5f5", padx=10, pady=10)
right_frame.pack(side="right", fill="y", padx=10, pady=10)

# Toggle Button Setup
settings_label = tk.Label(left_frame, text="Settings", font=("Helvetica", 14, "bold"), bg="#f5f5f5")
settings_label.pack(anchor="w", pady=5)

settings = [
    ("Activate Termination", activate_termination),
    ("Enable Travel Time", enable_travel_time),
    ("Display Convergence", display_convergence),
    ("Display Schedule", display_schedule),
    ("Create TXT File", create_txt_file),
    ("Apply Machine restrictions", machine_restrictions),
]

for text, var in settings:
    chk = tk.Checkbutton(left_frame, text=text, variable=var, bg="#f5f5f5", font=("Helvetica", 10))
    chk.pack(anchor="w", pady=2)

# Dropdown for benchmark selection
benchmark_label = tk.Label(left_frame, text="Benchmark Selection", font=("Helvetica", 12, "bold"), bg="#f5f5f5")
benchmark_label.pack(anchor="w", pady=10)

def update_benchmark_data(selected_benchmark):
    global workcenter_data, machine_data, ptime_data
    benchmark = benchmarks.__dict__.get(selected_benchmark)
    workcenter_data = benchmark['workcenter_data']
    machine_data = benchmark['machine_data']
    ptime_data = benchmark['ptime_data']

benchmark_var = tk.StringVar()
benchmark_menu = ttk.Combobox(left_frame, textvariable=benchmark_var, values=list(benchmarks.__dict__.keys()), state="readonly")
benchmark_menu.set("Select Benchmark")
benchmark_menu.bind("<<ComboboxSelected>>", lambda event: update_benchmark_data(benchmark_var.get()))
benchmark_menu.pack(anchor="w", pady=5)

# GA Parameter Inputs
ga_label = tk.Label(right_frame, text="GA Parameters", font=("Helvetica", 14, "bold"), bg="#f5f5f5")
ga_label.pack(anchor="w", pady=5)

param_labels = ["Population Size", "Max Generations", "Number of AMRs", "Number of Workcenters", "Number of Jobs"]
param_entries = {}
for label in param_labels:
    lbl = tk.Label(right_frame, text=label, bg="#f5f5f5", font=("Helvetica", 10))
    lbl.pack(anchor="w", pady=2)
    entry = tk.Entry(right_frame, width=20)
    entry.pack(anchor="w", pady=2)
    param_entries[label] = entry

# New input field for machine_data
machine_data_label = tk.Label(right_frame, text="Machine Data (Enter as a list)", bg="#f5f5f5", font=("Helvetica", 10))
machine_data_label.pack(anchor="w", pady=2)
machine_data_entry = tk.Entry(right_frame, width=20)
machine_data_entry.pack(anchor="w", pady=2)

# Run Button and Algorithm Execution
def run_algorithm():
    try:
        ga_params["population_size"] = int(param_entries["Population Size"].get())
        ga_params["max_generations"] = int(param_entries["Max Generations"].get())
        ga_params["num_amrs"] = int(param_entries["Number of AMRs"].get())
        ga_params["num_workcenters"] = int(param_entries["Number of Workcenters"].get())
        ga_params["num_jobs"] = int(param_entries["Number of Jobs"].get())
        # Convert machine_data input to a Python list
        if machine_data_entry.get().strip():
        # Convert machine_data input to a Python list if provided
            ga_params["machine_data"] = eval(machine_data_entry.get())
        else:
            # Use default machine_data if no input is provided
            ga_params["machine_data"] = machine_data
    except ValueError:
        print("Please enter valid integer values.")
        return
    
    

    # Create scheduler instance
    scheduler = FlexibleJobShopScheduler(
        ga_params["num_workcenters"],
        ga_params["num_jobs"],
        ga_params["num_amrs"],
        ga_params["population_size"],
        ga_params["crossover_rate"],
        ga_params["mutation_rate"],
        ga_params["max_generations"],
        workcenter_data,
        machine_data,
        ptime_data
    )
    
    # Set scheduler attributes based on checkboxes
    scheduler.activate_termination = activate_termination.get()
    scheduler.enable_travel_time = enable_travel_time.get()
    scheduler.display_convergence = display_convergence.get()
    scheduler.display_schedule = display_schedule.get()
    scheduler.create_txt_file = create_txt_file.get()
    scheduler.update_json_file = update_json_file.get()
    scheduler.machine_restriction = machine_restrictions.get()
    scheduler.machine_data = ga_params["machine_data"]  # Set machine_data in scheduler
    if ga_params["num_workcenters"] == 4:
        scheduler.set_distance_matrix(distances.four_machine_matrix)
    elif ga_params["num_workcenters"] == 6:
        scheduler.set_distance_matrix(distances.six_machine_matrix)
    elif ga_params["num_workcenters"] == 5:
        scheduler.set_distance_matrix(distances.five_machine_matrix)
    elif ga_params["num_workcenters"] == 10:
        scheduler.set_distance_matrix(distances.ten_machine_matrix)

    # Run the Genetic Algorithm
    best_chromosome = scheduler.GeneticAlgorithm()

run_button = tk.Button(right_frame, text="Run Algorithm", command=run_algorithm, bg="#4CAF50", fg="white", font=("Helvetica", 12), pady=5)
run_button.pack(anchor="w", pady=15)

root.mainloop()
