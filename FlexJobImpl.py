import random
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import rankdata
import time
import os
import json

from job import Job
from machine import Machine
from operation import Operation
from chromosome import Chromosome
from datetime import datetime
from amr import AMR
from workcenter import Workcenter

import distances
import benchmarks

import traceback
import inspect


c = 6
n = 6
num_amrs = 6
N = 100
pc = 0.7
pm = 0.5
pswap = 0.5
pinv = 0.5
T = 250

workcenter_data = benchmarks.ft06['workcenter_data']
ptime_data = benchmarks.ft06['ptime_data']
machine_data = benchmarks.ft06['machine_data']

'''
algo-params
'''

activate_termination = 0
enable_travel_time = 0
display_convergence = 0
display_schedule = 1
create_txt_file = 0
update_json_file = 0
runs = 1

if enable_travel_time:

    distance_matrix = distances.six_machine_matrix
else:
    distance_matrix = distances.empty_matrix


'''
initialization code

test_get_amr_assignments(): for testing assignment

get_amr_assignmnts(): will assign random amr to each job

create_operation_data(): compiles the input data to a sublist structure which is further used to easily derive operation data 
generate machines and generate_operations create internal machines and operation objects for each class using methods
assign_data_to_operations(): goes through operation data and assigns data to job objects
assign_amrs_to_jobs(): assigns amrs to each job object, and job to each amr object
set_travel_time(): sets the travel time for each operation based on its operation location and location of next operation


'''

def generate_population(N, n, c):
    population = []
    for _ in range(N):
        num = [round(random.uniform(0,c*n), 2) for _ in range(n*c)]
        population.append(num)
    return population

def generate_random_population(N, n, c, amr_assignments):
    encoded_lists = []
    population = []
    for _ in range(N):
        num = [round(random.uniform(0,c*n), 2) for _ in range(n*c)]
        encoded_lists.append(num)
        chromosome = process_chromosome(num, amr_assignments)
        population.append(chromosome)
        
    return population



def create_operation_data(workcenter_data, ptime_data, c):
    matrix = []
    sublist = []
    for i in range(len(workcenter_data)):
        sublist.append([workcenter_data[i], ptime_data[i]])
        if (i + 1) % c == 0:
            matrix.append(sublist)
            sublist = []
    # Check if there are remaining elements
    if sublist:
        matrix.append(sublist)
    return matrix
        
        
operation_data = create_operation_data(workcenter_data, ptime_data, c)
print(operation_data)

        
def test_get_amr_assignments():
    amr_assignments = [0, 1, 2, 1, 2, 0]
    # for num in range(n):
    #     amr_num = random.randint(0,num_amrs - 1)
    #     amr_assignments.append(amr_num)
        
    return amr_assignments

def get_amr_assignments():
    amr_assignments = []
    for num in range(n):
        amr_num = random.randint(0,num_amrs - 1)
        amr_assignments.append(amr_num)
        
    return amr_assignments
    
def remove_duplicates(numbers):
    seen = set()
    modified_numbers = []
    
    for num in numbers:
        # Check if the number is already in the set
        if num in seen:
            # Modify the number slightly
            modified_num = num + 0.01
            # Keep modifying until it's unique
            while modified_num in seen:
                modified_num += 0.01
            modified_numbers.append(modified_num)
            seen.add(modified_num)
        else:
            modified_numbers.append(num)
            seen.add(num)
        
    
    return modified_numbers


def generate_machines(workcenters, machine_data):
    for workcenter, qty in zip(workcenters, machine_data):
        workcenter.generate_machines(qty)
        
def generate_operations(jobs):
    for job in jobs:
        job.generate_operations(c)
        
def assign_data_to_operations(jobs, operation_data):
    for job,sublist in zip(jobs, operation_data):
        for operation,i in zip(job.operations, range(c)):
            operation.operation_number = i
            operation.workcenter = sublist[i][0]
            operation.Pj = sublist[i][1]
            
def assign_amrs_to_jobs(jobs, amrs, amr_assignments):
    for job, amr_num in zip(jobs, amr_assignments):
        job.amr_number = amr_num
        amrs[job.amr_number].assigned_jobs.append(job.job_number)
        
def set_travel_time(jobs, amrs, distance_matrix):
    for job in jobs:
        for operation in job.operations:
            operation.travel_time = operation.calculate_travel_time(amrs, jobs, distance_matrix, enable_travel_time)
        
        
'''
Decoding functions:

get_integer_list(): gets the integer list which will have the random numbers ranked e.g: [4, 10 ,6, 7, 3, 8, 2, 5, 9, 1]

get_jobindex_list(): takes in the integer list and permutates the job number e.g:        [2, 2, 1, 2, 1, 3, 3, 3, 1, 2]

get_machine_indices_list(): generate a list of ?(randomly) selected machine index for each workcenter for each operation

get_operation_objects(): takes the operation objects for all jobs and puts them in the list according to job index e.g :    [job2.op1, job2.op2, job1.op1, ..., job2.op4 ]. Also adds the machine number permutated to each operation based on machine indices.

get_workcenter_and_ptime_sequence(): takes in list sequence of operation objects and appends the workcenter number and Pj of each object.

get_machine_objects_list(): use the list to get machine objects in correct sequence <--- NEEDED?

assign_machine_operationlist(machines, operation_schedule): adds the operation objects inside the machines

'''
def get_integer_list(chromosome):    
    ranks = rankdata(chromosome)
    return [int(rank - 1) for rank in ranks]

def get_jobindex_list(chromosome):
    new_index = 0
    operation_index_pop = []

    tlist = []
    temp = chromosome
    for j in range(len(chromosome)):
        new_index = (temp[j] % n)
        tlist.append(new_index)
    operation_index_pop = tlist
    
    return operation_index_pop

def get_operation_objects(chromosome, jobs):
    operation_list = []
    explored = []
    # print(chromosome)
    # x = Counter(chromosome)
    # for i in x.elements():
    #     print( "% s : % s" % (i, x[i]), end ="\n")
    
    for i in range(len(chromosome)):
        explored.append(chromosome[i])
        numcount = explored.count(chromosome[i])
        # if numcount < m:
        operation_list.append(jobs[chromosome[i]].operations[numcount-1])
    return operation_list

def get_workcenter_and_time_sequence(operation_schedule):
    workcenter_sequence = []
    ptime_sequence = []
    for operation in operation_schedule:
        workcenter_sequence.append(operation.workcenter)
        ptime_sequence.append(operation.Pj)
    return workcenter_sequence , ptime_sequence

def get_machine_indices_list(encoded_list, workcenter_sequence, machine_data, operation_objects):
    
    machine_indices_list = []
    for r, c, operation in zip(encoded_list, workcenter_sequence, operation_objects):
        r = round(r,0)
        machine_no = int(r % machine_data[c])
        machine_indices_list.append(machine_no)
        operation.machine = machine_no
        
    return machine_indices_list

def assign_machine_operationlist(workcenters, operation_schedule):
    for operation in operation_schedule:
        workcenters[operation.workcenter].machines[operation.machine].operationlist.append(operation)

def get_Cmax(workcenters):
    runtimes = []
    max_runtime = 0
    for workcenter in workcenters:
        for machine in workcenter.machines:
            if machine.finish_operation_time > max_runtime:
                max_runtime = machine.finish_operation_time
        
    return max_runtime
        
def calculate_Cj_with_amr(operation_schedule, workcenters, jobs, amrs):
    t_op = operation_schedule
    skipped = []
    while t_op != []:
        # print('running')
        for operation in t_op:
            # CHECK IF AMR IS ASSIGNED TO A JOB, ONLY ASSIGN IF THE OPERATION NUMBER IS ZERO
            if amrs[jobs[operation.job_number].amr_number].current_job == None and operation.operation_number == 0:
                amrs[jobs[operation.job_number].amr_number].current_job = operation.job_number
                amrs[jobs[operation.job_number].amr_number].job_objects.append(jobs[operation.job_number]) # APPEND JOB OBJECTS
                # IF AMR JUST COMPLETED A JOB UPDATE THE NEXT JOBS MACHINE START TO THE TIME WHEN AMR COMPLETED PREVIOUS JOB
                if workcenters[operation.workcenter].machines[operation.machine].finish_operation_time < amrs[jobs[operation.job_number].amr_number].job_completion_time:
                    workcenters[operation.workcenter].machines[operation.machine].finish_operation_time = amrs[jobs[operation.job_number].amr_number].job_completion_time
                
                
            # CHECK IF AMR IS CURRENTLY PROCESSING THIS JOB
            if operation.job_number == amrs[jobs[operation.job_number].amr_number].current_job:
                
                if operation.operation_number == 0:
                    if amrs[jobs[operation.job_number].amr_number].completed_jobs == []:
                        operation.start_time = workcenters[operation.workcenter].machines[operation.machine].finish_operation_time
                    else:
                        # MAKE SURE THE PREVIOUS JOBS TRAVEL TIME SHOULD BE GIVEN TO NEXT JOB IF M'TH JOB IS HAVING PJ = 0
                        i = 0
                        while jobs[amrs[jobs[operation.job_number].amr_number].completed_jobs[-1]].operations[c-i-1].Pj == 0:
                            i+=1   
                        operation.start_time = workcenters[operation.workcenter].machines[operation.machine].finish_operation_time + jobs[amrs[jobs[operation.job_number].amr_number].completed_jobs[-1]].operations[c-i-1].travel_time
                        
                    jobs[operation.job_number].job_start_time = operation.start_time # SET JOB START TIME
                    operation.Cj = operation.start_time + operation.Pj
                    workcenters[operation.workcenter].machines[operation.machine].finish_operation_time = operation.Cj
                    # print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
                    
                    
                else:
                    # IF MACHINE RUN TIME IS LESSER THAN JOB COMPLETION TIME AND TRAVEL TIME FROM PREVIOUS LOCATION COMBINED.
                    if jobs[operation.job_number].operations[operation.operation_number - 1].Cj + jobs[operation.job_number].operations[operation.operation_number - 1].travel_time < workcenters[operation.workcenter].machines[operation.machine].  finish_operation_time:
                        operation.start_time = workcenters[operation.workcenter].machines[operation.machine].finish_operation_time
                        operation.Cj = operation.start_time + operation.Pj
                        workcenters[operation.workcenter].machines[operation.machine].finish_operation_time = operation.Cj 
                        # print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
                        
                    else:
                        operation.start_time = jobs[operation.job_number].operations[operation.operation_number - 1].Cj + jobs[operation.job_number].operations[operation.operation_number - 1].travel_time
                        operation.Cj = operation.start_time + operation.Pj
                        if operation.Pj != 0:
                            workcenters[operation.workcenter].machines[operation.machine].finish_operation_time = operation.Cj
                        # print(f'machine no: {machines[operation.machine].machine_id}, new finish time :{machines[operation.machine].finish_operation_time}')
                
                
            # SKIP THE JOB AND RETURN TO IT LATER
            else:
                skipped.append(operation)
            
            # UPDATE PARAMETERS ONCE A JOB IS COMPLETED
            if operation.operation_number == c - 1 and amrs[jobs[operation.job_number].amr_number].current_job == operation.job_number:
                        amrs[jobs[operation.job_number].amr_number].current_job = None
                        if amrs[jobs[operation.job_number].amr_number].assigned_jobs != []:
                            amrs[jobs[operation.job_number].amr_number].assigned_jobs.remove(operation.job_number)
                        amrs[jobs[operation.job_number].amr_number].completed_jobs.append(operation.job_number)
                        # IF FINAL JOB PJ IS ZERO TAKE PREV COMPLETED TIME
                        if operation.Pj != 0:
                            amrs[jobs[operation.job_number].amr_number].job_completion_time = operation.Cj
                            jobs[operation.job_number].job_completion_time = amrs[jobs[operation.job_number].amr_number].job_completion_time
                        else:
                            i = 0
                            while jobs[operation.job_number].operations[operation.operation_number - i].Pj == 0:
                                i += 1
                            amrs[jobs[operation.job_number].amr_number].job_completion_time = jobs[operation.job_number].operations[operation.operation_number -  i].Cj
                        jobs[operation.job_number].job_completion_time = amrs[jobs[operation.job_number].amr_number].job_completion_time
                
        t_op = skipped
        skipped = []
        
def PlotGanttChart_with_amr_scalable(chromosome, workcenter_machine_list):

    # Get the makespan (Cmax) from the chromosome
    Cmax = chromosome.Cmax

    # Figure and set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [8, 1]})

    # Bottom Gantt chart (main) - Workcenters and Machines
    ax = axs[0]
    ax.set_ylabel('Workcenter\nMachine', fontweight='bold', loc='top', color='black', fontsize=12)

    # Calculate total number of machines
    total_machines = sum(workcenter_machine_list)

    # Create yticks based on the total number of machines and the workcenters
    yticks = []
    ytick_labels = []
    current_tick = 0

    for wc_index, machines_in_wc in enumerate(workcenter_machine_list):
        for machine_num in range(machines_in_wc):
            yticks.append(current_tick)
            ytick_labels.append(f'{wc_index+1}-{machine_num+1}')  # Workcenter-Machine label
            current_tick += 1

    ax.set_ylim(-0.5, total_machines - 0.5)
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(ytick_labels, minor=False)
    ax.tick_params(axis='y', labelcolor='black', labelsize=10)
    
    ax.set_xlim(0, Cmax + 2)
    ax.tick_params(axis='x', labelcolor='black', labelsize=12)
    ax.grid(True, linestyle='--')

    tmpTitle = f'Scheduling for c={c}; n={n} and AMRs={num_amrs} with Cmax={round(Cmax, 2)}'
    ax.set_title(tmpTitle, size=14, color='black')

    colors = ['orange', 'deepskyblue', 'indianred', 'limegreen', 'slateblue', 'gold', 'violet', 'grey', 'red', 'magenta', 'blue', 'green', 'silver', 'lavender', 'turquoise', 'orchid'] # Adjust based on how many jobs you want

    # Plotting the job operations for each machine within each workcenter
    machine_count = 0  # Keep track of the global machine index across workcenters
    for wc_index, machines_in_wc in enumerate(workcenter_machine_list):
        for machine_index in range(machines_in_wc):
            joblen = len(chromosome.workcenter_list[wc_index].machines[machine_index].operationlist)
            for k in range(joblen):
                j = chromosome.workcenter_list[wc_index].machines[machine_index].operationlist[k]
                ST = j.start_time
                if j.Pj != 0:
                    # Job operation block
                    ax.broken_barh([(ST, j.Pj)], (-0.3 + machine_count, 0.6), facecolor=colors[j.job_number], linewidth=1, edgecolor='black')
                    # Travel time block
                    ax.broken_barh([(j.Cj, j.travel_time)], (-0.3 + machine_count, 0.6), facecolor='black', linewidth=1, edgecolor='black')
                    # Text in the middle of job blocks
                    ax.text(ST + (j.Pj / 2 - 0.3), machine_count + 0.03, '{}'.format(j.job_number + 1), fontsize=10, color='white')

            machine_count += 1  # Move to the next machine in global index

    # Top Gantt chart with custom y-ticks (AMRs)
    top_ax = axs[1]
    top_ax.set_ylabel('AMR', fontweight='bold', loc='top', color='black', fontsize=12)
    top_ax.set_xlabel('time', fontweight='bold', loc='right', color='black', fontsize=12)
    top_ax.set_ylim(-0.5, num_amrs - 0.5)
    top_ax.set_yticks(range(num_amrs), minor=False)
    top_ax.set_yticklabels(range(1, num_amrs + 1), minor=False)
    top_ax.tick_params(axis='y', labelcolor='black', labelsize=10)
    top_ax.set_xlim(0, Cmax + 2)
    top_ax.tick_params(axis='x', labelcolor='black', labelsize=12)
    top_ax.grid(True, linestyle='--')

    # Plot the AMR jobs
    for i in range(num_amrs):
        joblen = len(chromosome.amr_list[i].job_objects)
        for k in range(joblen):
            j = chromosome.amr_list[i].job_objects[k]
            ST = j.job_start_time
            duration = j.job_completion_time - j.job_start_time
            if duration != 0:
                top_ax.broken_barh([(ST, duration)], (-0.3 + i, 0.6), facecolor=colors[j.job_number], linewidth=1, edgecolor='black')
                top_ax.text(ST + (duration) / 2 , i - 0.2, '{}'.format(j.job_number + 1), fontsize=10, ha='center', color='white')

    plt.tight_layout()

    # Add a legend manually to the figure
    # from matplotlib.lines import Line2D
    # legend_elements = [Line2D([0], [0], color='green', lw=4, label='Job 1'),
    #                    Line2D([0], [0], color='red', lw=4, label='Job 2'),
    #                    Line2D([0], [0], color='blue', lw=4, label='Job 3'),
    #                    Line2D([0], [0], color='black', lw=4, label='AMR Travel Time')]
    # ax.legend(handles=legend_elements, loc='lower right')

    if create_txt_file:
        # CHANGE DIRECTORY FOR SAVING FIGURE
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'E:\\Python\\JobShopGA\\Results\\pc0.6pm0.5\\la23\\gantt{timestamp}'
        plt.savefig(filename)

'''
process_chromosome(): takes in an encoded list and spits out a fully processed chromosome.
'''

def process_chromosome(chromosome, amr_assignments):
    
    jobs = [Job(number) for number in range(n)]
    amrs = [AMR(number) for number in range(num_amrs)]
    workcenters = [Workcenter(number) for number in range(c)]
    
    generate_machines(workcenters, machine_data)
    # assign_operations(jobs, operation_data)
    
    chromosome = remove_duplicates(chromosome)

    ranked_list = get_integer_list(chromosome)

    operation_index_list = get_jobindex_list(ranked_list)
    
    # # CASE 1
    # operation_index_list = [0, 1, 2, 0, 2, 1,  1, 0, 2, 1, 0, 2]
    
    
    generate_operations(jobs)
    assign_data_to_operations(jobs, operation_data)

    operation_objects = get_operation_objects(operation_index_list, jobs)   

    assign_amrs_to_jobs(jobs, amrs, amr_assignments)
    
    # # get the sequence of machines and ptimes
    workcenter_sequence, ptime_sequence = get_workcenter_and_time_sequence(operation_objects)
    
    machine_sequence = get_machine_indices_list(chromosome, workcenter_sequence, machine_data, operation_objects)
    # # SET TRAVEL TIMES FOR EACH JOB
    set_travel_time(jobs, amrs, distance_matrix)
    
    # calculate_Cj(operation_schedule, machines, jobs)
    calculate_Cj_with_amr(operation_objects, workcenters, jobs, amrs)
    assign_machine_operationlist(workcenters, operation_objects)
    Cmax = get_Cmax(workcenters)
    
    chromosome = Chromosome(chromosome)
        
    chromosome.ranked_list = ranked_list
    chromosome.operation_index_list = operation_index_list
    chromosome.job_list = jobs
    chromosome.amr_list = amrs
    chromosome.operation_schedule = operation_objects
    chromosome.workcenter_sequence = workcenter_sequence
    chromosome.machine_sequence = machine_sequence
    chromosome.workcenter_list = workcenters
    chromosome.ptime_sequence = ptime_sequence
    chromosome.Cmax = Cmax
    chromosome.fitness = chromosome.Cmax + chromosome.penalty
    
    return chromosome


'''
Genetic Code:

tournament(): performs two way tournament of population and returns a list of winner chromosomes

single_point_crossover(): performs crossover between two parents and returns two offsprings

single_bit_mutation(): selects single point in chromosome and mutates, returns another mutated chromsome.

next_gen_selection(): selections of population moving on to next generation

swapping(): performs swapping operation on chromosome, returns new swapped chromsome

inversion(): performs inversion operation on chromsome encoded list, returns new chromosome

'''

def tournament(population):
    indices2 = [x for x in range(N)]
    
    winners = []
    while len(indices2) != 0:
        i1 = random.choice(indices2)
        i2 = random.choice(indices2)
        while i1 == i2:
            i2 = random.choice(indices2)
            
        if population[i1].fitness < population[i2].fitness:
            winners.append(population[i1])
        else:
            winners.append(population[i2])
            
        indices2.remove(i1)
        indices2.remove(i2)
    
    indices2 = [x for x in range(N)]
    
    while len(indices2) != 0:
        i1 = random.choice(indices2)
        i2 = random.choice(indices2)
        while i1 == i2:
            i2 = random.choice(indices2)
            
        if population[i1].fitness < population[i2].fitness:
            winners.append(population[i1])
        else:
            winners.append(population[i2])
            
        indices2.remove(i1)
        indices2.remove(i2)
        
    return winners

def single_point_crossover(chrom1, chrom2, amr_assignments):
    
    parent1 = chrom1.encoded_list
    parent2 = chrom2.encoded_list
    
    r = random.uniform(0,1)
    # r = 0.4
    
    p = random.randint(0,len(parent1))
    if r > pc:
        return chrom1 , chrom2
    else:
        offspring1 = parent1[0:p] + parent2[p:]
        offspring2 = parent2[0:p] + parent1[p:]
        # checked_offsp1 = remove_duplicates(offspring1)[:]
        # checked_offsp2 = remove_duplicates(offspring2)[:]
        chrom_out1 = process_chromosome(offspring1, amr_assignments)
        chrom_out2 = process_chromosome(offspring2, amr_assignments)
    
    return chrom_out1, chrom_out2

def single_bit_mutation(chromosome, amr_assignments):
    
    r = random.uniform(0,1)
    code = chromosome.encoded_list[:]
    
    if r > pm:
        return chromosome
    else:
        index = random.randint(0, len(code) - 1)
        code[index] = round(random.uniform(0,c*n), 2)
        # checked_code = remove_duplicates(code)[:]
        mutated_chromosome = process_chromosome(code, amr_assignments)
    
    return mutated_chromosome

def next_gen_selection(parents, offsprings):
    total_population = []
    total_population.extend(parents)
    total_population.extend(offsprings)
    
    sortedGen = []
    sortedGen = sorted(total_population, key = lambda x : x.fitness)
    return sortedGen[:N], sortedGen[0]

def swapping(chromosome, amr_assignments):
    r = random.uniform(0,1)
    if r > pswap:
        return chromosome
    
    code = chromosome.encoded_list[:]
    indexes = [num for num in range(len(code))]
    
    p = random.choice(indexes)
    q = random.choice(indexes)
    while p == q:
        q = random.choice(indexes)
        
    code[p], code[q] = code[q], code[p]
    
    swapped_chromosome = process_chromosome(code, amr_assignments)
    return swapped_chromosome

def inversion(chromosome, amr_assignments):
    
    r = random.uniform(0,1)
    if r > pinv:
        return chromosome
    
    code = chromosome.encoded_list[:]
    indexes = [num for num in range(len(code))]
    p = random.choice(indexes)
    q = random.choice(indexes)
    while p == q:
        q = random.choice(indexes)
        
    
    p, q = min(p, q), max(p, q)
    code[p:q+1] = reversed(code[p:q+1])
    
    inverted_chromosome = process_chromosome(code, amr_assignments)
    
    return inverted_chromosome



'''
Main Driver code
'''

def tests():
    random_list = generate_population(N, n, c)

    amr_assignments = test_get_amr_assignments()
    
    processed_chromosome = process_chromosome(random_list[0], amr_assignments)
    PlotGanttChart_with_amr_scalable(processed_chromosome, machine_data)
    plt.show()
    print('\n')


def GeneticAlgorithm():
    
    start_time = time.time()
    flag = 0
    count = 0
    t = 0
    ypoints = []
    
    amr_assignments = get_amr_assignments()
    population = generate_random_population(N, n, c, amr_assignments)
        
    sorted_population = sorted(population, key = lambda  x : x.fitness )
        
    best_chromosome = sorted_population[0]
    
    history = 0
    stagnation = 0
    
    while t < T:
            
            new_amr_assignments = get_amr_assignments()
            
            # create mating pool
            winners_list = tournament(population)
            # winners_list = three_way_tournament(population)
            
            
            
            # perform crossover on mating pool
            indices = [x for x in range(N)]
            offspring_list = winners_list
            while len(indices) != 0:
                i1 = random.choice(indices)
                i2 = random.choice(indices)
                while i1 == i2:
                    i2 = random.choice(indices)
                    
                rchoice = random.uniform(0,1)
                if rchoice < 1:
                    offspring1, offspring2 = single_point_crossover(winners_list[i1], winners_list[i2], new_amr_assignments)
                # else:
                #     # potential bug, skipping job
                #     offspring1, offspring2 = double_point_crossover(winners_list[i1], winners_list[i2], new_amr_assignments)
                offspring_list[i1] = offspring1
                offspring_list[i2] = offspring2
                
                indices.remove(i1)
                indices.remove(i2)
                
            # perform mutation
            enhanced_list = []
            for chromosome in offspring_list:
                mutated_chromosome = single_bit_mutation(chromosome, new_amr_assignments)
                
                # perform swapping operation
                swap_chromosome = swapping(mutated_chromosome, new_amr_assignments)
                
                if swap_chromosome.Cmax < mutated_chromosome.Cmax:
                    enhanced_list.append(swap_chromosome)
                    inverted_chromosome = inversion(swap_chromosome, new_amr_assignments)
                    if inverted_chromosome.Cmax < swap_chromosome.Cmax:
                        enhanced_list.append(inverted_chromosome)
                    else:
                        enhanced_list.append(swap_chromosome)
                else:    
                    enhanced_list.append(mutated_chromosome)
            
                # # perform inversion operation on chromosome
                # inverted_chromosome = inversion(swap_chromosome, new_amr_assignments)
                
                # enhanced_list.append(mutated_chromosome)
                
                # # selection of survivors for next generation
            
            survivors, best_in_gen = next_gen_selection(winners_list, enhanced_list)
            
            survivors[-1] = best_in_gen
            if best_in_gen.fitness < best_chromosome.fitness:
                best_chromosome = best_in_gen
                amr_assignments = new_amr_assignments
                
            if best_chromosome.fitness == history and activate_termination == 1:
                stagnation += 1
                
            # if stagnation > 10:
            #     elapsed = time.time() - start_time
            #     converged_at = elapsed
            # else:
            #     converged_at = 0
            
            #CHECK IF AMR ASSIGNMENT IS BETTER OR WORSE
            
            history = best_chromosome.fitness
                
            ypoints.append(best_chromosome.fitness)
            winners_list = survivors
            
            if (t + 1) % 25 == 0:
                print(f'At generation {t + 1}, best fitness :{best_chromosome.fitness}')
            
            
    
            t += 1
            # end of loop
            
            
    xpoints = [x for x in range(1, t+ 1)]
    
    if display_convergence:
        plt.plot(xpoints, ypoints,  color= 'b')
    
    # Record the end time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # if create_txt_file:
    #     get_file(best_chromosome, processing_time, converged_at)
    
    
    # print(f'best Cmax = {ypoints[N-1]}')
    print(f'best Cmax = {best_chromosome.fitness}')
    
    print('random generated numbers:',best_chromosome.encoded_list)
    print(f'ranked list : {best_chromosome.ranked_list}\n operation_index :{best_chromosome.operation_index_list},\n operation object{best_chromosome.operation_schedule}\n')
    print(f'machine sequence: {best_chromosome.machine_sequence}\n ptime sequence: {best_chromosome.ptime_sequence}\n Cmax: {best_chromosome.Cmax}')



    PlotGanttChart_with_amr_scalable(best_chromosome, machine_data)
    
    
    if display_schedule:
        plt.show()
    
    # machine_seq_amrs, ptime_seq_amrs = get_sequences_in_amr(best_chromosome.amr_list)
    # print(machine_seq_amrs,'\n',ptime_seq_amrs)   
    
    # if update_json_file:
    #     create_amr_json(machine_seq_amrs, ptime_seq_amrs, 'amr_data.json')

    # plt.show()
    
    print('\n')
    
if __name__ == '__main__':
    GeneticAlgorithm()