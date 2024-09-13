from operation import Operation

# number of jobs

n = 15
c = 4

class Job:
    def __init__(self, job_number):
        # self.joblist = []
        self.job_number = job_number % n
        self.job_start_time = 0
        self.job_completion_time = 0
        # self.operations = tuple([[0 , 0] for _ in range(n)])
        self.operations = None
        self.amr_number = None
        
        
    def generate_operations(self, c):
        self.operations = [Operation(self.job_number) for i in range(c)]
        
