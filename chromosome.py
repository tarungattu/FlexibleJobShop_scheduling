class Chromosome:
    def __init__(self, encoded_list):
        self.encoded_list = encoded_list
        self.ranked_list = []
        self.operation_index_list = []
        self.workcenter_sequence = []
        self.machine_sequence = []
        self.ptime_sequence = []
        
        # List of objects
        self.job_list = []
        self.amr_list = []
        self.workcenter_list = []
        self.machine_list = []
        self.operation_schedule = []
        
        self.Cmax = 99999
        self.penalty = 0
        self.wait_time = 0
        self.total_processing_time = 0
        self.total_number_of_machines = 0
        self.idle_time = 0
        self.jobs_completion_time = 0
        self.fitness = 99999
        
    def set_fitness(self):
        self.fitness = self.idle_time*0 + self.wait_time*0 + self.jobs_completion_time*1 + self.Cmax*0 + self.penalty
        
    def set_wait_time(self):
        wait_time = 0
        amrs = self.amr_list
        for amr in amrs:
            i = 0
            j = 1
            r = 0
            l = -1
            for r in range(len(amr.job_objects)):
                job = amr.job_objects[r]
                for i in range(len(job.operations)):
                    if i < len(job.operations) - 1:
                        if i == 0:
                            if r == 0:
                                wait_time += job.job_start_time                            
                            else:
                                wait_time += job.job_start_time - amr.job_objects[r - 1].job_completion_time 
                                # print(wait_time)
                        
                        wait_time += job.operations[i + 1].start_time - job.operations[i].Cj
                        # print(wait_time)
                    
        self.wait_time = wait_time
                    
    def set_idle_time(self):
        self.idle_time = (self.Cmax*self.total_number_of_machines) - self.total_processing_time
        
    def set_jobs_completion_time(self):
        sum = 0
        for job in self.job_list:
            sum += job.job_completion_time
            
        self.jobs_completion_time = sum
                
                
                
                