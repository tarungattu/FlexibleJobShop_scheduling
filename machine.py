#number of machines
m = 15

class Machine:
    def __init__(self, machine_id, workcenter_id):
        self.operationlist = []
        self.workcenter_id = workcenter_id
        self.machine_id = machine_id % m #machine number
        self.start_operation_time = 0
        self.finish_operation_time = 0
        
        
    def set_workcenter_max_runtime(self, workcenter):
        if self.finish_operation_time > workcenter.max_runtime:
            workcenter.max_runtime = self.finish_operation_time
        
