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

# Global dictionary to hold GA parameters
ga_params = {
    "num_amrs": 0,
    "num_machines": 0,
    "num_jobs": 0,
    "population_size": 0,
    "crossover_rate": 0.7,
    "mutation_rate": 0.5,
    "max_generations": 0,
}

# Toggle variables for each setting
activate_termination = tk.IntVar()
enable_travel_time = tk.IntVar()
display_convergence = tk.IntVar()
display_schedule = tk.IntVar(value=1)
create_txt_file = tk.IntVar()
update_json_file = tk.IntVar()

# Toggle Button Setup
settings = [
    ("Activate Termination", activate_termination),
    ("Enable Travel Time", enable_travel_time),
    ("Display Convergence", display_convergence),
    ("Display Schedule", display_schedule),
    ("Create TXT File", create_txt_file),
    # ("Update JSON File", update_json_file),
]

for text, var in settings:
    chk = tk.Checkbutton(root, text=text, variable=var)
    chk.pack()

# Dropdown for benchmark selection
def update_benchmark_data(selected_benchmark):
    global workcenter_data, machine_data, ptime_data
    benchmark = benchmarks.__dict__.get(selected_benchmark)
    workcenter_data = benchmark['workcenter_data']
    machine_data = benchmark['machine_data']
    ptime_data = benchmark['ptime_data']

benchmark_var = tk.StringVar()
benchmark_menu = ttk.Combobox(root, textvariable=benchmark_var, values=list(benchmarks.__dict__.keys()))
benchmark_menu.set("Select Benchmark")
benchmark_menu.bind("<<ComboboxSelected>>", lambda event: update_benchmark_data(benchmark_var.get()))
benchmark_menu.pack()

# GA Parameter Inputs
param_labels = ["Population Size", "Max Generations", "Number of AMRs", "Number of Workcenters", "Number of Jobs"]
param_entries = {}
for label in param_labels:
    lbl = tk.Label(root, text=label)
    lbl.pack()
    entry = tk.Entry(root)
    entry.pack()
    param_entries[label] = entry

# Run Button and Algorithm Execution
def run_algorithm():
    try:
        ga_params["population_size"] = int(param_entries["Population Size"].get())
        ga_params["max_generations"] = int(param_entries["Max Generations"].get())
        ga_params["num_amrs"] = int(param_entries["Number of AMRs"].get())
        ga_params["num_workcenters"] = int(param_entries["Number of Workcenters"].get())
        ga_params["num_jobs"] = int(param_entries["Number of Jobs"].get())
        # ga_params["machine_data"] = list(param_entries["Number of machines each"].get())
        
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

run_button = tk.Button(root, text="Run Algorithm", command=run_algorithm)
run_button.pack()

root.mainloop()
