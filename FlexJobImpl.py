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


c = 4
n = 3
num_amrs = 2
N = 50
pc = 0.7
pm = 0.5
pswap = 0.5
pinv = 0.5
T = 100

workcenter_data = benchmarks.pinedo['workcenter_data']
ptime_data = benchmarks.pinedo['ptime_data']
machine_data = benchmarks.pinedo['machine_data']

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

    distance_matrix = distances.four_machine_matrix
else:
    distance_matrix = distances.empty_matrix


'''
initialization code

test_get_amr_assignments(): for testing assignment
create_operation_data(): compiles the input data to a sublist structure which is further used to easily derive operation data 
generate machines and generate_operations create internal machines and operation objects for each class using methods
assign_data_to_operations(): goes through operation data and assigns data to job objects
assign_amrs_to_jobs(): assigns amrs to each job object
set_travel_time(): sets the travel time for each operation based on its operation location and location of next operation


'''


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
    amr_assignments = [0, 1, 1]
    # for num in range(n):
    #     amr_num = random.randint(0,num_amrs - 1)
    #     amr_assignments.append(amr_num)
        
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
get_operation_objects(): takes the operation objects for all jobs and puts them in the list according to job index e.g :    [job2.op1, job2.op2, job1.op1, ..., job2.op4 ]
get_workcenter_and_ptime_sequence(): takes in list sequence of operation objects and appends the workcenter number and Pj of each object.

get_machine_indices_list(): generate a list of randomly selected machine index for each workcenter for each operation  <--- currently working
get_machine_objects_list(): use the list to get machine objects in correct sequence <--- NEEDED?

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

def get_workcenter_and_time_sequence(operation_schedule):
    workcenter_sequence = []
    ptime_sequence = []
    for operation in operation_schedule:
        workcenter_sequence.append(operation.workcenter)
        ptime_sequence.append(operation.Pj)
    return workcenter_sequence , ptime_sequence


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
    # # print(chromosome)
    ranked_list = get_integer_list(chromosome)
    # # print(ranked_list)
    operation_index_list = get_jobindex_list(ranked_list)
    
    # # CASE 1
    operation_index_list = [0, 1, 2, 0, 2, 1,  1, 0, 2, 1, 0, 2]
    
    
    generate_operations(jobs)
    assign_data_to_operations(jobs, operation_data)
    # check_list_length(operation_index_list)
    
    operation_schedule = get_operation_objects(operation_index_list, jobs)   
    # check_list_length(operation_schedule)
    assign_amrs_to_jobs(jobs, amrs, amr_assignments)
    
    # # get the sequence of machines and ptimes
    workcenter_sequence, ptime_sequence= get_workcenter_and_time_sequence(operation_schedule)
    
    
    # # SET TRAVEL TIMES FOR EACH JOB
    set_travel_time(jobs, amrs, distance_matrix)
    
    # # calculate_Cj(operation_schedule, machines, jobs)
    # calculate_Cj_with_amr(operation_schedule, machines, jobs, amrs)
    # assign_machine_operationlist(machines, operation_schedule)
    # Cmax = get_Cmax(machines)
    
    # chromosome = Chromosome(chromosome)
        
    # chromosome.ranked_list = ranked_list
    # chromosome.operation_index_list = operation_index_list
    # chromosome.job_list = jobs
    # chromosome.amr_list = amrs
    # chromosome.operation_schedule = operation_schedule
    # chromosome.machine_sequence = machine_sequence
    # chromosome.machine_list = machines
    # chromosome.ptime_sequence = ptime_sequence
    # chromosome.Cmax = Cmax
    # chromosome.fitness = chromosome.Cmax + chromosome.penalty
    
    # return chromosome

def tests():
    test_chromosome = [0, 1]
    amr_assignments = test_get_amr_assignments()
    
    process_chromosome(test_chromosome, amr_assignments)

    
if __name__ == '__main__':
    tests()