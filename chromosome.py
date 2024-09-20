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
        self.fitness = 99999
        
    def set_fitness(self):
        self.fitness = self.Cmax + self.penalty