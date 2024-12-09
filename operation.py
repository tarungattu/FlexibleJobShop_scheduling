class Operation:
    def __init__(self, job_number):
        index = 0
        self.job_number = job_number
        self.operation_number = 0
        self.workcenter = None  # int
        self.machine = None     # int
        self.Pj = None
        self.start_time = 0
        self.Cj = 0
        self.travel_time = 0
        
    def __str__(self):
        return str(self.job_number)
    
    def __repr__(self):
        return f'(job {self.job_number} operation {self.operation_number})'
        
    def getCj(self):
        self.Cj = self.start_time + self.Pj
        
        
        
    # TURN ON SOURCE AND DISTANCE FOR USING TRAVEL TIME
    def calculate_travel_time(self, amrs, jobs, distance_matrix, en_tt, initial = 0):
        distance = 0
        
        velocity = amrs[jobs[self.job_number].amr_number].velocity
        
        if en_tt:
            
            source = self.workcenter
            
            if self.Pj == 0:
                return 0
            
            # IF NEXT JOB PROCESSING TIME IS 0 PUT THIS JOB AS LAST JOB
            if self.operation_number != len(jobs[self.job_number].operations) - 1:
                if jobs[self.job_number].operations[self.operation_number + 1].Pj == 0:
                    
                    distance = distance_matrix[source][distance_matrix.shape[0]- 1] + distance_matrix[distance_matrix.shape[0] - 1][distance_matrix.shape[0] - 2]  
                    
                    return distance/velocity
            
            # CHECK IF THE JOB IS INITIALLY TRAVELLING FROM LOADING DOCK TO MACHINE
            if initial == 1:
                dest = source
                distance = distance_matrix[distance_matrix.shape[0]- 2][dest]
                return distance/velocity
                    
            if self.operation_number == len(jobs[self.job_number].operations) - 1:
                
                distance = distance_matrix[source][distance_matrix.shape[0]- 1] + distance_matrix[distance_matrix.shape[0] - 1][distance_matrix.shape[0] - 2]
            
            else:
                dest = jobs[self.job_number].operations[self.operation_number + 1].workcenter
                distance = distance_matrix[source][dest]
                
        return distance/velocity
            