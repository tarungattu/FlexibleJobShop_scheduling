import random
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import rankdata
import time
import os
import json
import math

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


class FlexibleJobShopScheduler():
    def __init__(self, c, n, num_amrs, N, pc, pm, T, workcenter_data, machine_data, ptime_data):
        # GA params
        self.c = c
        self.n = n
        self.num_amrs = num_amrs
        self.N = N
        self.pc = pc
        self.pm = pm
        self.pswap = pm
        self.pinv = pm
        self.T = T
        self.workcenter_data = workcenter_data
        self.machine_data = machine_data
        self.ptime_data = ptime_data
        self.stagnation_limit = 100
        self.parent_perc = 0.2
        self.offspring_perc = 0.8
        
        # tools
        self.activate_termination = 0
        self.enable_travel_time = 0
        self.display_convergence = 0
        self.display_schedule = 1
        self.create_txt_file = 0
        self.update_json_file = 0
        self.runs = 1
        
        # constraints
        self.machine_restriction = 0
        
        if self.machine_restriction:
            self.M_j = [[0, 0, [1]], [1, 0, [1]]]  # [job, operation, [elligible machines]]
        else:
            self.M_j = None
        
        self.distance_matrix = None
        self.save_file_directory = 'D:\\SDP\\Results\\Default'
        
        if self.enable_travel_time:
            self.distance_matrix = distances.four_machine_matrix
        else:
            self.distance_matrix = distances.empty_matrix
            
            
        self.operation_data = self.create_operation_data(self.workcenter_data, self.ptime_data, self.c)
        self.amr_assignments = None
        
        
    # recieve number of completed jobs and reschedule the remaining jobs, also mention number of amrs present during rescheduling.
    # def reschedule(self, num_completed_jobs, num_amrs):
    #     self.machine_data = self.machine_data[num_completed_jobs * self.m:]
    #     self.ptime_data = self.ptime_data[num_completed_jobs * self.m:]
    #     self.operation_data = self.create_operation_data(self.workcenter_data, self.ptime_data, self.m)
    #     self.n = self.n - num_completed_jobs
    #     self.num_amrs = num_amrs
        
        
        
    def set_distance_matrix(self, matrix):
        self.distance_matrix = matrix
                    

    def create_operation_data(self, workcenter_data, ptime_data, c):
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
    
    
    def generate_population(self, N, c):
        population = []
        for _ in range(N):
            num = [round(random.uniform(0,c*self.n), 2) for _ in range(self.n*c)]
            population.append(num)
        return population  
    
    
    def generate_random_population(self, N, n, c, amr_assignments):
        encoded_lists = []
        population = []
        heuristic = 0
        for _ in range(N):
            num = [round(random.uniform(0,c*n), 2) for _ in range(n*c)]
            encoded_lists.append(num)
            chromosome = self.process_chromosome(num, amr_assignments, heuristic)
            population.append(chromosome)
            
        return population
    
    def generate_heuristic_population(self, N, n, c, amr_assignments):
        encoded_lists = []
        population = []
        heuristic = 1
        for _ in range(N):
            num = [round(random.uniform(0,c*n), 2) for _ in range(n*c)]
            encoded_lists.append(num)
            chromosome = self.process_chromosome(num, amr_assignments, heuristic)
            population.append(chromosome)
            
        return population
    
    def integer_list(self, population):
        ranked_population = []
        for i in range(self.N):
            sorted_list = []
            ranks = {}
            # Sort the list to get ranks in ascending order
            sorted_list = sorted(population[i])
                
            # Create a dictionary to store the ranks of each float number
            ranks = {value: index + 1 for index, value in enumerate(sorted_list)}
                
            # Convert each float number to its corresponding rank
            rank_list = [ranks[value] for value in population[i]]
            ranked_population.append(rank_list)
            
        return ranked_population    
    
    def indiv_integer_list(self, chromosome):    
        ranks = rankdata(chromosome)
        return [int(rank - 1) for rank in ranks]
    
    def remove_duplicates(self, numbers):
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
    
    def generate_machines(self, workcenters, machine_data):
        for workcenter, qty in zip(workcenters, machine_data):
            workcenter.generate_machines(qty)
    
    def generate_operations(self, jobs):
        for job in jobs:
            job.generate_operations(self.c)
    
    def getJobindex(self, population):
        new_index = 0
        operation_index_pop = []
        for i in range(self.N):
            tlist = []
            temp = population[i]
            for j in range(self.m*self.n):
                new_index = (temp[j] % self.n) + 1
                tlist.append(new_index)
            operation_index_pop.append(tlist)
        
        return operation_index_pop
    
    def indiv_getJobindex(self, chromosome):
        new_index = 0
        operation_index_pop = []

        tlist = []
        temp = chromosome
        for j in range(len(chromosome)):
            new_index = (temp[j] % self.n)
            tlist.append(new_index)
        operation_index_pop = tlist
        
        return operation_index_pop
    
    
    def indiv_schedule_operations(self, chromosome, jobs):
        operation_list = []
        explored = []

        for i in range(len(chromosome)):
            explored.append(chromosome[i])
            numcount = explored.count(chromosome[i])

            operation_list.append(jobs[chromosome[i]].operations[numcount-1])  # changed chromosome[i] to chromosome[i]-1
        return operation_list

    def install_operations(self, jobs):
        for job in jobs:
            job.operations = [Operation(job.job_number) for i in range(self.m)]

    # operation_data = create_operation_data(machine_data,ptime_data, m)

    def assign_data_to_operations(self, jobs, operation_data):
        for job,sublist in zip(jobs, operation_data):
            for operation,i in zip(job.operations, range(self.c)):
                operation.operation_number = i
                operation.workcenter = sublist[i][0]
                operation.Pj = sublist[i][1]
    
    def assign_amrs_to_jobs(self, jobs, amrs, amr_assignments):
        for job, amr_num in zip(jobs, amr_assignments):
            job.amr_number = amr_num
            amrs[job.amr_number].assigned_jobs.append(job.job_number)
            
    def set_travel_time(self, jobs, amrs, distance_matrix):
        for job in jobs:
            for operation in job.operations:
                operation.travel_time = operation.calculate_travel_time(amrs, jobs, distance_matrix, self.enable_travel_time)
                
    def get_integer_list(self, chromosome):    
        ranks = rankdata(chromosome)
        return [int(rank - 1) for rank in ranks]
            
    def get_jobindex_list(self, chromosome):
        new_index = 0
        operation_index_pop = []

        tlist = []
        temp = chromosome
        for j in range(len(chromosome)):
            new_index = (temp[j] % self.n)
            tlist.append(new_index)
        operation_index_pop = tlist
        
        return operation_index_pop 
    
    def get_operation_objects(self, chromosome, jobs):
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
            jobs[chromosome[i]].operations[numcount-1].index = i
            operation_list.append(jobs[chromosome[i]].operations[numcount-1])
        return operation_list   
    
    def get_workcenter_and_time_sequence(self, operation_schedule):
        workcenter_sequence = []
        ptime_sequence = []
        for operation in operation_schedule:
            workcenter_sequence.append(operation.workcenter)
            ptime_sequence.append(operation.Pj)
        return workcenter_sequence , ptime_sequence     
    
    def get_machine_indices_list(self, encoded_list, workcenter_sequence, machine_data, operation_objects):
    
        machine_indices_list = []
        for r, c, operation in zip(encoded_list, workcenter_sequence, operation_objects):
            r = round(r,0)
            machine_no = int(r % machine_data[c])
            machine_indices_list.append(machine_no)
            operation.machine = machine_no
            
        return machine_indices_list  
    
    def get_random_machine_indices_list(self, encoded_list, workcenter_sequence, machine_data, operation_objects):
    
        machine_indices_list = []
        for  c, operation in zip(workcenter_sequence, operation_objects):
            
            machine_no = random.randint(0, machine_data[c] - 1)
            machine_indices_list.append(machine_no)
            operation.machine = machine_no
            
        return machine_indices_list

    def get_amr_assignments(self):
        amr_assignments = []
        for num in range(self.n):
            amr_num = random.randint(0, self.num_amrs - 1)
            amr_assignments.append(amr_num)
            
        return amr_assignments
                
                
    def get_machine_sequence(self, operation_schedule):
        machine_sequence = []
        for operation in operation_schedule:
            machine_sequence.append(operation.machine)
            
        return machine_sequence

    def get_processing_times(self, operation_schedule):
        ptime_sequence = []
        for operation in operation_schedule:
            ptime_sequence.append(operation.Pj)
            
        return ptime_sequence
    
    def calculate_Cj_with_amr(self, operation_schedule, workcenters, jobs, amrs):
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
                            # THE AMR MUST TRAVEL TO FIRST MACHINE BEFORE PROCESSING FIRST OPERATION
                            initial_travel_time = operation.calculate_travel_time(amrs, jobs, self.distance_matrix, self.enable_travel_time, 1)
                            if workcenters[operation.workcenter].machines[operation.machine].finish_operation_time > initial_travel_time:
                                operation.start_time = workcenters[operation.workcenter].machines[operation.machine].finish_operation_time
                            else:
                                operation.start_time = workcenters[operation.workcenter].machines[operation.machine].finish_operation_time + initial_travel_time
                        else:
                            # MAKE SURE THE PREVIOUS JOBS TRAVEL TIME SHOULD BE GIVEN TO NEXT JOB IF M'TH JOB IS HAVING PJ = 0
                            i = 0
                            while jobs[amrs[jobs[operation.job_number].amr_number].completed_jobs[-1]].operations[self.c-i-1].Pj == 0:
                                i+=1   
                            operation.start_time = workcenters[operation.workcenter].machines[operation.machine].finish_operation_time + jobs[amrs[jobs[operation.job_number].amr_number].completed_jobs[-1]].operations[self.c-i-1].travel_time
                            
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
                if operation.operation_number == self.c - 1 and amrs[jobs[operation.job_number].amr_number].current_job == operation.job_number:
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
        # eof while    
 
    def assign_machine_operationlist(self, workcenters, operation_schedule):
        for operation in operation_schedule:
            workcenters[operation.workcenter].machines[operation.machine].operationlist.append(operation)

    def get_Cmax(self, workcenters):
        runtimes = []
        max_runtime = 0
        for workcenter in workcenters:
            for machine in workcenter.machines:
                if machine.finish_operation_time > max_runtime:
                    max_runtime = machine.finish_operation_time
            
        return max_runtime
    
    def get_travel_time(self, jobs, amrs, distance_matrix):
        for job in jobs:
            for operation in job.operations:
                operation.travel_time = operation.calculate_travel_time(amrs, jobs, distance_matrix, self.enable_travel_time)
                
    def process_chromosome(self, encoded_list, amr_assignments, heuristic = 0):
    
        jobs = [Job(number) for number in range(self.n)]
        amrs = [AMR(number) for number in range(self.num_amrs)]
        workcenters = [Workcenter(number) for number in range(self.c)]
        
        self.generate_machines(workcenters, self.machine_data)
        # assign_operations(jobs, operation_data)
        
        encoded_list = self.remove_duplicates(encoded_list)

        ranked_list = self.get_integer_list(encoded_list)

        operation_index_list = self.get_jobindex_list(ranked_list)
        
        
        self.generate_operations(jobs)
        self.assign_data_to_operations(jobs, self.operation_data)

        # [each index replaced with its object]
        operation_objects = self.get_operation_objects(operation_index_list, jobs)   

        self.assign_amrs_to_jobs(jobs, amrs, amr_assignments)
        
        # # get the sequence of machines and ptimes
        workcenter_sequence, ptime_sequence = self.get_workcenter_and_time_sequence(operation_objects)
        
        # use heuristic machine selection for initial population only
        if heuristic:
            machine_sequence = self.get_machine_indices_list(encoded_list, workcenter_sequence, self.machine_data, operation_objects)
        else:
            machine_sequence = self.get_random_machine_indices_list(encoded_list, workcenter_sequence, self.machine_data, operation_objects)
            
        # HERE GOES MACHINE RESTRICTIONS REPAIR FUNCTION
        if self.machine_restriction == 1:
            self.repair_machine_constraints(machine_sequence, operation_objects, jobs)
            
            
        # # SET TRAVEL TIMES FOR EACH JOB
        self.set_travel_time(jobs, amrs, self.distance_matrix)
        
        # calculate_Cj(operation_schedule, machines, jobs)
        self.calculate_Cj_with_amr(operation_objects, workcenters, jobs, amrs)
        self.assign_machine_operationlist(workcenters, operation_objects)
        Cmax = self.get_Cmax(workcenters)
        
            
        chromosome = Chromosome(encoded_list)
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
        chromosome.total_processing_time = sum(self.ptime_data)
        chromosome.total_number_of_machines = sum(self.machine_data)
        chromosome.set_idle_time()
        chromosome.set_wait_time()
        chromosome.set_jobs_completion_time()
        chromosome.set_fitness()
        
        return chromosome
    
    def PlotGanttChart_with_amr_scalable(self, chromosome, workcenter_machine_list):

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
                ytick_labels.append(f'{wc_index}-{machine_num}')  # Workcenter-Machine label
                current_tick += 1

        ax.set_ylim(-0.5, total_machines - 0.5)
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(ytick_labels, minor=False)
        ax.tick_params(axis='y', labelcolor='black', labelsize=10)
        
        ax.set_xlim(0, Cmax + 2)
        ax.tick_params(axis='x', labelcolor='black', labelsize=12)
        ax.grid(True, linestyle='--')

        tmpTitle = f'Scheduling for c={self.c}; n={self.n} and AMRs={self.num_amrs} with Cmax={math.ceil(Cmax)}'
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
                        ax.text(ST + (j.Pj / 2 - 0.3), machine_count + 0.03, '{}'.format(j.job_number), fontsize=10, color='white')

                machine_count += 1  # Move to the next machine in global index

        # Top Gantt chart with custom y-ticks (AMRs)
        top_ax = axs[1]
        top_ax.set_ylabel('AMR', fontweight='bold', loc='top', color='black', fontsize=12)
        top_ax.set_xlabel('time', fontweight='bold', loc='right', color='black', fontsize=12)
        top_ax.set_ylim(-0.5, self.num_amrs - 0.5)
        top_ax.set_yticks(range(self.num_amrs), minor=False)
        top_ax.set_yticklabels(range(0, self.num_amrs), minor=False)
        top_ax.tick_params(axis='y', labelcolor='black', labelsize=10)
        top_ax.set_xlim(0, Cmax + 2)
        top_ax.tick_params(axis='x', labelcolor='black', labelsize=12)
        top_ax.grid(True, linestyle='--')

        # Plot the AMR jobs
        for i in range(self.num_amrs):
            joblen = len(chromosome.amr_list[i].job_objects)
            for k in range(joblen):
                j = chromosome.amr_list[i].job_objects[k]
                ST = j.job_start_time
                duration = j.job_completion_time - j.job_start_time
                if duration != 0:
                    top_ax.broken_barh([(ST, duration)], (-0.3 + i, 0.6), facecolor=colors[j.job_number], linewidth=1, edgecolor='black')
                    top_ax.text(ST + (duration) / 2 , i - 0.2, '{}'.format(j.job_number), fontsize=10, ha='center', color='white')

        plt.tight_layout()
        
        if self.create_txt_file:
            # CHANGE DIRECTORY FOR SAVING FIGURE
            
            folder_name = "run results"
        
            # Check if the folder exists,   if not, create it
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                
                
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = os.path.join(folder_name, self.GetUniqueFileName("GA", "png"))
            
            plt.savefig(filename)
            
    def GetUniqueFileName (self, prefix, ftype):
        timestamp = int (time.time())
        fileName = "{}_m{}_n{}_a{}_{}.{}".format (prefix, self.c, self.n, self.num_amrs, timestamp, ftype)
        return fileName
        

    def tournament(self, population):
        indices2 = [x for x in range(self.N)]
        
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
        
        indices2 = [x for x in range(self.N)]
        
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
    

    def stochastic_universal_sampling(self, population, num_parents):
        # Calculate inverted fitness values
        max_fitness = max(chromosome.fitness for chromosome in population)
        inverted_fitness = [max_fitness - chromosome.fitness for chromosome in population]

        # Calculate total inverted fitness
        total_inverted_fitness = sum(inverted_fitness)

        # Calculate distance between selection pointers
        pointer_distance = total_inverted_fitness / num_parents

        # Randomly choose a starting point for the selection pointers
        start_point = random.uniform(0, pointer_distance)

        # Create selection pointers
        pointers = [start_point + i * pointer_distance for i in range(num_parents)]

        # Initialize selected individuals list
        selected_individuals = []

        # Iterate over selection pointers and select individuals
        cumulative_fitness = 0
        idx = 0
        for pointer in pointers:
            while cumulative_fitness < pointer:
                cumulative_fitness += inverted_fitness[idx]
                idx += 1
            selected_individuals.append(population[idx])

        return selected_individuals
    
    def single_point_crossover(self, chrom1, chrom2, amr_assignments):
        
        parent1 = chrom1.encoded_list
        parent2 = chrom2.encoded_list
        
        r = random.uniform(0,1)
        # r = 0.4
        
        p = random.randint(0,len(parent1))
        if r > self.pc:
            return chrom1 , chrom2
        else:
            offspring1 = parent1[0:p] + parent2[p:]
            offspring2 = parent2[0:p] + parent1[p:]
            # checked_offsp1 = remove_duplicates(offspring1)[:]
            # checked_offsp2 = remove_duplicates(offspring2)[:]
            chrom_out1 = self.process_chromosome(offspring1, amr_assignments)
            chrom_out2 = self.process_chromosome(offspring2, amr_assignments)
        
        return chrom_out1, chrom_out2   
    
    def single_bit_mutation(self, chromosome, amr_assignments):
        
        r = random.uniform(0,1)
        code = chromosome.encoded_list[:]
        
        if r > self.pm:
            return chromosome
        else:
            index = random.randint(0, len(code) - 1)
            code[index] = round(random.uniform(0,self.c*self.n), 2)
            # checked_code = remove_duplicates(code)[:]
            mutated_chromosome = self.process_chromosome(code, amr_assignments)
        
        return mutated_chromosome 
    
    def next_gen_selection(self, parents, offsprings):
        total_population = []
        total_population.extend(parents)
        total_population.extend(offsprings)
        
        sortedGen = []
        sortedGen = sorted(total_population, key = lambda x : x.fitness)
        return sortedGen[:self.N], sortedGen[0]
    
    def next_gen_selection_elitism(self, parents, offsprings, parent_perc, offspring_perc):
        # Use elitism from parent population
        sorted_parent = []
        number = len(parents)  # Your number
        twenty_percent = int(number * parent_perc)
        sorted_parents = sorted(parents, key = lambda  x : x.fitness )
        
        sorted_offsprings = []
        number = len(offsprings)  # Your number
        eighty_percent = number - twenty_percent
        sorted_offsprings = sorted(sorted_offsprings, key = lambda  x : x.fitness )
        
        total_population = sorted_parents[0:twenty_percent] + sorted_offsprings[twenty_percent:]
        sorted_total_population = sorted(total_population, key = lambda  x : x.fitness )
        return total_population, sorted_total_population[0]
    
    
    def swapping(self, chromosome, amr_assignments):
        r = random.uniform(0,1)
        if r >self.pswap:
            return chromosome
        
        code = chromosome.encoded_list[:]
        indexes = [num for num in range(len(code))]
        
        p = random.choice(indexes)
        q = random.choice(indexes)
        while p == q:
            q = random.choice(indexes)
            
        code[p], code[q] = code[q], code[p]
        
        swapped_chromosome = self.process_chromosome(code, amr_assignments)
        return swapped_chromosome
        
    def inversion(self, chromosome, amr_assignments):
        
        r = random.uniform(0,1)
        if r > self.pinv:
            return chromosome
        
        code = chromosome.encoded_list[:]
        indexes = [num for num in range(len(code))]
        p = random.choice(indexes)
        q = random.choice(indexes)
        while p == q:
            q = random.choice(indexes)
            
        
        p, q = min(p, q), max(p, q)
        code[p:q+1] = reversed(code[p:q+1])
        
        inverted_chromosome = self.process_chromosome(code, amr_assignments)
        
        return inverted_chromosome
    
    def SPT_heuristic(self, operation_data):
        operation_index_list = []
        n = len(operation_data[0])  # Number of operations
        m = len(operation_data)     # Number of jobs

        for j in range(n):
            tlist = [(i, operation_data[i][j]) for i in range(m)]
            tlist.sort(key=lambda x: x[1][1])  # Sort based on processing time
            operation_index_list.extend([t[0] for t in tlist])

        return operation_index_list

    def LPT_heuristic(self, operation_data):
        operation_index_list = []
        n = len(operation_data[0])  # Number of operations
        m = len(operation_data)     # Number of jobs

        for j in range(n):
            tlist = [(i, operation_data[i][j]) for i in range(m)]
            tlist.sort(key=lambda x: x[1][1], reverse=True)  # Sort based on processing time
            operation_index_list.extend([t[0] for t in tlist])
            
        return operation_index_list

    def srt_heuristic(self, operation_data):
        rem_time = 0
        job_rem_time = []
        operation_index_list = []
        
        for i in range(self.m):
            job_rem_time = []
            for job in operation_data:
                rem_time = 0
                tjob = job[i:]
                for operation in tjob:
                    rem_time += operation[1]
                job_rem_time.append(rem_time)
            sorted_indices = sorted(range(len(job_rem_time)), key=lambda x: job_rem_time[x])
            operation_index_list.extend(sorted_indices)
        return operation_index_list

    def decode_operations_to_schedule(self, operation_index, num_jobs):
        n = len(operation_index)
        possible_indices = [[(num_jobs * j + op) for j in range(n // num_jobs + 1)] for op in operation_index]
        ranked_list = [0] * n
        used_indices = set()
        is_valid = True
        for i, options in enumerate(possible_indices):
            # Find the smallest available index that hasn't been used yet
            for option in sorted(options):
                if option not in used_indices and option < n:
                    ranked_list[i] = option
                    used_indices.add(option)
                    break
            else:
                # If no valid option is found, note that configuration may be invalid
                is_valid = False
                break

        if not is_valid:
            return None, None  # Indicate that no valid configuration was found
        
        random_numbers = [0] * n
        index_to_number = {rank: i for i, rank in enumerate(ranked_list)}
        for i in range(n):
            random_numbers[index_to_number[i]] = i + 1  # Simple 1-to-n mapping for simplicity

        return ranked_list, random_numbers
    
    
    # not in use
    def generate_population_with_heuristic(self, operation_data, amr_assignments):
        n = self.n
        m = self.m
        N = self.N
        population = []
        number = n*m
        
        if N > 6:
        
            for i in range(2):
                srt_op_seq = self.srt_heuristic(operation_data)
                ranked, code = self.decode_operations_to_schedule(srt_op_seq, n)
                population.append(self.process_chromosome(code, amr_assignments))
            
            for i in range(2):
                spt_op_seq = self.SPT_heuristic(operation_data)
                ranked, code = self.decode_operations_to_schedule(spt_op_seq, n)
                population.append(self.process_chromosome(code, amr_assignments))
                
                
            for i in range(2):
                lpt_op_seq = self.LPT_heuristic(operation_data)
                ranked, code = self.decode_operations_to_schedule(lpt_op_seq, n)
                population.append(self.process_chromosome(code, amr_assignments))
            
            for i in range(N - 6):
                num = [round(random.uniform(0,m*n), 2) for _ in range(n*m)]
                population.append(self.process_chromosome(num, amr_assignments))
            
        else:
            initial_population = self.generate_population(N)
            population = []
            for encoded_list in initial_population:
                # print(f'generated list: {encoded_list}')
                chromosome = self.process_chromosome(encoded_list, amr_assignments)
                population.append(chromosome)
            
        return population

    def get_sequences_in_amr(self, amrs):
        amr_machines = []
        amr_ptimes = []
        glob_amr_machine = []
        glob_amr_ptime = []
        for amr in amrs:
            for j in amr.job_objects:
                for o in j.operations:
                    amr_machines.append(o.machine)
                    amr_ptimes.append(o.Pj)
                amr_machines.extend([-2, -1])
                amr_ptimes.extend([3, 3])
            amr.machine_sequence = amr_machines
            amr.ptime_sequence = amr_ptimes
            glob_amr_machine.append(amr_machines)
            glob_amr_ptime.append(amr_ptimes)
            amr_machines = []
            amr_ptimes = []
            
        return glob_amr_machine, glob_amr_ptime
    
    def create_amr_json(self, machine_sequences, ptime_sequences, output_file):
        # Initialize the structure
        amr_data = {
            "amr_list": [
                {
                    "amr_no": 1,
                    "machine_sequence": machine_sequences[0],
                    "ptime_sequence": ptime_sequences[0]
                },
                {
                    "amr_no": 2,
                    "machine_sequence": machine_sequences[1],
                    "ptime_sequence": ptime_sequences[1]
                }
            ]
        }

        # Write the data to a JSON file
        with open(output_file, 'w') as json_file:
            json.dump(amr_data, json_file, indent=4)
            
    def get_file(self, best_chromosome, processing_time, xpoints, ypoints):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = self.GetUniqueFileName("GA", "txt")

        # Define the folder name
        directory = "run results"  # Folder for saving the file

        # Check if folder exists, create if necessary
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Construct the full file path
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w') as file:
            file.write(f"Welcome to main function at {datetime.now().strftime('%d-%m %H:%M:%S')}.{datetime.now().microsecond}\n")
            file.write(f"Population size: {self.N}\n")
            file.write(f"Number of generations: {self.T}\n")
            file.write(f"Number of AMRs: {self.num_amrs}\n")
            file.write(f"Encoded list: {best_chromosome.encoded_list}\n")
            file.write(f"ranked list: {best_chromosome.ranked_list}\n")
            file.write(f"operation_index list: {best_chromosome.operation_index_list}\n")
            file.write(f"machine_sequence: {best_chromosome.machine_sequence}\n")
            file.write(f"ptime sequence: {best_chromosome.ptime_sequence}\n\n")
            
            # file.write(f"amr_machine_sequences: {best_chromosome.amr_machine_sequences}\n")
            # file.write(f"amr_ptime_sequences: {best_chromosome.amr_ptime_sequences}\n")
            

            file.write(f"Makespan is {best_chromosome.Cmax} time units\n")
            file.write(f"Fitness is {best_chromosome.fitness} \n")
            file.write(f"Problem solved in {round(processing_time, 2)} seconds\n\n")

            file.write("----------------------------------------------------------------------------------------------\n")
            file.write("n \t c\t a\t T \t N \t Pc \t Pm \t Cmax \t WAIT \t IDLE\t Jobs completion \t CPU Time (s) \t Termination value \t Machines in Workcenters\n ")
            file.write("----------------------------------------------------------------------------------------------\n")
            file.write(f" {self.n} \t {self.c} \t {self.num_amrs} \t {self.T} \t {self.N} \t {self.pc} \t {self.pm}  \t {best_chromosome.Cmax} \t {best_chromosome.wait_time} \t {best_chromosome.idle_time} \t {best_chromosome.jobs_completion_time} \t {round(processing_time, 2)} \t {self.stagnation_limit} \t {self.machine_data}\n")
            file.write("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
            
            for i, j in zip(xpoints, ypoints):
                file.write(f'{i} \t {j} \n')
                
                
    def setup_constraints(self):
        # self.M_j = [[0, 0, [1]]]    # [job, operation, [elligible machines]]
        pass
    
    
    # Validate each constraint. Select a machine only from elligible set.
    def repair_machine_constraints(self, machine_sequence, operation_objects, jobs):
        
        for constraint in self.M_j:
            job_index = constraint[0]
            operation_index = constraint[1]
            elligible_machines = constraint[2]  # list of elligible machines
            
            # for i, operation in enumerate(operation_objects):
            #     if job_index == operation.job_number and operation_index == operation.operation_number:
            #         operation.machine = random.choice(elligible_machines)
            #         machine_sequence[i] = operation.machine
            #         break
            
            exact_operation = jobs[job_index].operations[operation_index]
            
            if exact_operation.machine in elligible_machines:
                # do nothine
                continue
            else:
                # access the exact operation and reset machine to elligible machine
                exact_operation.machine = random.choice(elligible_machines)
                # also edit machine sequence list at the correct index
                machine_sequence[exact_operation.index] = exact_operation.machine
            
            
            
        

    def GeneticAlgorithm(self):
        
        start_time = time.time()
        flag = 0
        count = 0
        t = 0
        ypoints = []
        
        # set initial amr assignments
        amr_assignments = self.get_amr_assignments()
        population = self.generate_random_population(self.N, self.n, self.c, amr_assignments)
            
        sorted_population = sorted(population, key = lambda  x : x.fitness )
        
        # get current  best chromosome
        best_chromosome = sorted_population[0]
        
        history = 0
        stagnation = 0
        
        if self.machine_restriction:
            self.setup_constraints()
        
        # start generation loop
        while t < self.T:
                
                new_amr_assignments = self.get_amr_assignments()
                
                # create mating pool
                winners_list = self.tournament(population)
                # winners_list = three_way_tournament(population)
                
                
                # perform crossover on mating pool
                indices = [x for x in range(self.N)]
                offspring_list = winners_list
                while len(indices) != 0:
                    i1 = random.choice(indices)
                    i2 = random.choice(indices)
                    while i1 == i2:
                        i2 = random.choice(indices)
                        
                    rchoice = random.uniform(0,1)
                    if rchoice < 1:
                        offspring1, offspring2 = self.single_point_crossover(winners_list[i1], winners_list[i2], new_amr_assignments)
                    # else:
                    #     # potential bug, skipping job
                    #     offspring1, offspring2 = double_point_crossover(winners_list[i1], winners_list[i2], new_amr_assignments)
                    offspring_list[i1] = offspring1
                    offspring_list[i2] = offspring2
                    
                    indices.remove(i1)
                    indices.remove(i2)
                    
                    
                # REPAIR FUNCTIONS TO GO HERE
                    
                    
                    
                    
                    
                # perform mutation
                enhanced_list = []
                for chromosome in offspring_list:
                    
                    select_mutation = random.randint(1, 3)
                    
                    if select_mutation == 1:
                        mutated_chromosome = self.single_bit_mutation(chromosome, new_amr_assignments)
                    
                    elif select_mutation == 2:
                    # perform swapping operation
                        mutated_chromosome = self.swapping(chromosome, new_amr_assignments)
                    
                    else:
                        # enhanced_list.append(swap_chromosome)
                        mutated_chromosome = self.inversion(chromosome, new_amr_assignments)
                       
                    enhanced_list.append(mutated_chromosome)
                
                # REPAIR FUNCTIONS TO GO HERE
                
                
                
                
                
                
                # EVALUATE FITNESS OF EACH CHROMOSOME ONCE MORE, PENALIZE IF BREAKING CONSTRAINTS
                
                
                
                
                # select next population
                survivors, best_in_gen = self.next_gen_selection_elitism(winners_list, enhanced_list, self.parent_perc, self.offspring_perc)
                
                survivors[-1] = best_in_gen
                #CHECK IF AMR ASSIGNMENT IS BETTER OR WORSE
                if best_in_gen.fitness < best_chromosome.fitness:
                    best_chromosome = best_in_gen
                    amr_assignments = new_amr_assignments
                    
                if best_chromosome.fitness == history and self.activate_termination == 1:
                    stagnation += 1
                else:
                    stagnation = 0
                    
                if stagnation > self.stagnation_limit:
                    elapsed = time.time() - start_time
                    converged_at = elapsed
                    break
                
                
                history = best_chromosome.fitness
                    
                ypoints.append(best_chromosome.fitness)
                winners_list = survivors
                
                if (t + 1) % 25 == 0:
                    print(f'At generation {t + 1}, best fitness :{best_chromosome.fitness}')
                
                
        
                t += 1
                # end of loop
                
                
        xpoints = [x for x in range(1, t+ 1)]
        
        if self.display_convergence:
            plt.plot(xpoints, ypoints,  color= 'b')
        
        # Record the end time
        end_time = time.time()
        processing_time = end_time - start_time
        
        if self.create_txt_file:
            self.get_file(best_chromosome, processing_time, xpoints, ypoints)
        
        
        # print(f'best Cmax = {ypoints[N-1]}')
        print(f'best Cmax = {best_chromosome.fitness}')
        
        print('random generated numbers:',best_chromosome.encoded_list)
        print(f'ranked list : {best_chromosome.ranked_list}\n operation_index :{best_chromosome.operation_index_list},\n operation object{best_chromosome.operation_schedule}\n')
        print(f'machine sequence: {best_chromosome.machine_sequence}\n ptime sequence: {best_chromosome.ptime_sequence}\n Cmax: {best_chromosome.Cmax}\n wait time: {best_chromosome.wait_time}\n idle_time: {best_chromosome.idle_time}\n job_completion times : {best_chromosome.jobs_completion_time} \n fitness: {best_chromosome.fitness}')



        self.PlotGanttChart_with_amr_scalable(best_chromosome, self.machine_data)
        
        
        if self.display_schedule:
            plt.show()
        else:
            plt.close()
        
        # machine_seq_amrs, ptime_seq_amrs = get_sequences_in_amr(best_chromosome.amr_list)
        # print(machine_seq_amrs,'\n',ptime_seq_amrs)   
        
        # if update_json_file:
        #     create_amr_json(machine_seq_amrs, ptime_seq_amrs, 'amr_data.json')

        # plt.show()
        
        print('\n')
        return best_chromosome
    
    
    


if __name__ == '__main__':
    pass