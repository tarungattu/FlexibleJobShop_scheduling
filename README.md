# Flexible Job Shop Scheduling Problem Solver using Genetic Algorithm

Welcome to my genetic algorithm implementation for solving the Job Shop Scheduling Problem (JSSP). This project aims to provide an efficient solution to JSSP, which is a complex combinatorial optimization problem. This code implements the Job Shop Scheduling problem with AMR constraints, making it a step towards realization of Lineless Assembly Systems! (Stay tuned for the article currently in developement).

## Getting Started

### Prerequisites

Ensure you have the following software installed:
- Python 3
- Git

**Clone the Repository**

   Clone the `main` branch of the repository:

   ```bash
   git clone https://github.com/tarungattu/JobShopGA_withAMR.git
   ```
### Running the Algorithm

The scheduler is built as a class, having various methods including the main genetic algorithm method. In order to run it you must import it in your python file.
Dont forget to import the distance matrices and the Benchmarks so you can test any instances.

```bash
from JobShopScheduler import JobShopScheduler
import benchmarks
import distances
```

The various **parameters** must be set before creating a scheduler

```bash
m - number of Machines
n - number of Jobs
num_amrs - number of AMRs
N - Population size for Genetic Algorithm
pc - Crossover Probability
pm - Mutation Probability
T - Maximum number of generations
machine_data - job data for machine sequence. Can be used from the benchmarks file
ptime_data - processing time data for each job on each machine. Can be used from the benchmarks file.
```

Each benchmark is a dictionary containing two keys: machine_data and ptime_data
There are 4 distance matrices predefined according to layouts shown below.
(will work on it later)



<img src="https://github.com/user-attachments/assets/7634ef83-dd3c-4dd6-85b5-24eca7914209" alt="machine5 6" width="500"/>

<img src="https://github.com/user-attachments/assets/4e69a26f-1474-4d5a-a79f-ddec2f912472" alt="machine10" width="500"/>
<img src="https://github.com/user-attachments/assets/5376780c-5741-401c-80b3-43db72b32f3d" alt="Screenshot 2024-10-17 141455" width="500"/>


## Example Run - Pinedo instance 4 machines, 3 jobs
Here's an example to make a scheduler object in your main function. We use the pinedo benchmark, which is simpler problem having 4 machines and 3 jobs. We define num_amrs as 2, and give ur GA parameters to the scheduler. After creating the scheduler object, we must also select the distance matrix for calulation of travel time. The velocity of the AMRs can be changes in the amr.py file. For this example we use the four_machine_matrix for a layout of 4 machines.

```bash
example_scheduler = JobShopScheduler(4, 3, 2, 50, 0.7, 0.5, 100, benchmarks.pindeo['machine_data'] , benchmarks.pinedo['ptime_data'])    
example_scheduler.set_distance_matrix(distances.four_machine_matrix)
```

There are more scheduler parameters we can modify after creating the object, 0 - OFF; 1 - ON
```bash
activate_termination = 0
stagnation_limit = 50
enable_travel_time = 0
display_convergence = 0
display_schedule = 0
create_txt_file = 0
update_json_file = 0
runs = 1
```

These allow you to do usefull stuff such as terminate the code after reaching a certain stagnamtion limit, enable or disable using the travel time, display the evolution graph of the generations, display the schedule as a gantt chart (matplotlib brokenbar), and even write the solution information into a text file after specifying the saving dirsctory for the scheduler!
      
**Note: Saving the file will create a folder called runs, and will dump the best chromosome gantt chart and txt file in it.**

Finally run the genetic algorithm, optionally you can store the best chromosome for accessing any data needed later on for any other application.

```bash
best_chromosome = example_scheduler.GeneticAlgorithm()
```


<img src="https://github.com/user-attachments/assets/a52a6b00-94cc-40b0-b303-1526492a93df" alt="Example solution" width="700"/>

## Guide to ADD instances to the Benchmarks file (coming soon)
Benchmark instances in the benchmarks file are stored as dictionaries, (might later make them into json files). The dictionary will have workcenter data; the routing of the job on the shop floor, and ptime_data, the amount of time the job will stay at each machine for processing.
Machine data will contain number of machines in each workcenter. (index wise)
For example:
```bash
pinedo = {
    'workcenter_data' : [0,1,2,3, 1,0,3,2, 0,1,3,2],
    'ptime_data' : [10,8,4,0, 8,3,5,6, 4,7,3,0],
    'machine_data: [2,2,2,2]
}
```
The first four numbers in machine_data are the routing of machine numbers job 0 will go to:= machine 0-> machine 1-> machine 2-> machine 3.
The first four numbers in ptime_data are the corresponding processing time at each machine. 10 seconds processing at machine 0, 8 seconds processing at machine 1, and so on. Each Job will have the route including every machine, even if it isnt going to visit the machine. (In this case the processing time is set to zero, like job 0 at machine 3, since the processing time is zero, it will not visit that machine.)

Hence the data for job 1 will be after the first 4 values in both lists. (As there are a total of 4 machines in this instance)
This follows the Taillard standard of organizing Job shop scheduling instances)

