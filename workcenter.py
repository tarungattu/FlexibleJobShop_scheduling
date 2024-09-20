c = 4
from machine import Machine

class Workcenter:
    def __init__(self, workcenter_id):
        self.machines = []   # machine objects
        self.workcenter_id = workcenter_id % c # center number
        self.operation_list = []
        self.max_runtime = None
        
    
    def generate_machines(self, qty):
        self.machines = [Machine(id, self.workcenter_id) for id in range(qty)]