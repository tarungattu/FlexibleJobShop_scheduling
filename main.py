from FlexibleJobShopScheduler import FlexibleJobShopScheduler
import benchmarks
import distances

def main():
    
    workcenter_data4 = benchmarks.pinedo['workcenter_data']
    machine_data4 = benchmarks.pinedo['machine_data']
    ptime_data4 = benchmarks.pinedo['ptime_data']
    
    workcenter_data6 = benchmarks.ft06c6n6['workcenter_data']
    machine_data6 = benchmarks.ft06c6n6['machine_data']
    ptime_data6 = benchmarks.ft06c6n6['ptime_data'] 
    
    
    workcenter_data5  = benchmarks.la01c5n10['workcenter_data']
    machine_data5  = benchmarks.la01c5n10['machine_data']
    ptime_data5  = benchmarks.la01c5n10['ptime_data']
    
    
    workcenter_data10 = benchmarks.ft10c10n10['workcenter_data']
    machine_data10 = benchmarks.ft10c10n10['machine_data']
    ptime_data10 = benchmarks.ft10c10n10['ptime_data']
    
    ta20_20_workcenter_data = benchmarks.twenty_twenty['workcenter_data']
    ta20_20_machine_data = benchmarks.twenty_twenty['machine_data']
    ta20_20_ptime_data = benchmarks.twenty_twenty['ptime_data']
    ta20_50_machine_data = benchmarks.twenty_fifty['machine_data']
    ta20_50_ptime_data = benchmarks.twenty_fifty['ptime_data']
    
    # la06_data5  = benchmarks.la06['machine_data']
    # la06_data5  = benchmarks.la06['machine_data']
    # la06_ptime5  = benchmarks.la06['ptime_data']
    
    
    # la23_data10 = benchmarks.la23['machine_data']
    # la23_data10 = benchmarks.la23['machine_data']
    # la23_ptime10 = benchmarks.la23['ptime_data']
    
    
    
    
    # scheduler1 = FlexibleJobShopScheduler(4, 3, 2, 6, 0.7, 0.5, 100, workcenter_data4, machine_data4, ptime_data4)    
    # scheduler1.set_distance_matrix(distances.four_machine_matrix)
    
    
    # scheduler1 = FlexibleJobShopScheduler(6, 6, 3, 50, 0.7, 0.5, 350, workcenter_data6, machine_data6, ptime_data6)    
    # scheduler1.set_distance_matrix(distances.six_machine_matrix)
    
    
    # scheduler1 = FlexibleJobShopScheduler(5, 10, 3, 500, 0.7, 0.5, 350, workcenter_data5, machine_data5, ptime_data5)    
    # scheduler1 = FlexibleJobShopScheduler(5, 15, 3, 350, 0.7, 0.5, 450, la06_data5, la06_ptime5)    
    # scheduler1.set_distance_matrix(distances.five_machine_matrix)
    
    
    # scheduler1 = FlexibleJobShopScheduler(10, 10, 4, 100, 0.7, 0.5, 350, workcenter_data10, machine_data10, ptime_data10)    
    # scheduler1 = FlexibleJobShopScheduler(10, 15, 3, 350, 0.7, 0.5, 450, la23_data10, la23_ptime10)    
    # scheduler1.set_distance_matrix(distances.ten_machine_matrix)
    
    scheduler1 = FlexibleJobShopScheduler(20, 20, 3, 50, 0.7, 0.5, 600, ta20_20_workcenter_data, ta20_20_machine_data, ta20_20_ptime_data)    
    # # scheduler1 = JobShopScheduler(20, 50, 7, 100, 0.7, 0.5, 600, ta20_50_machine_data, ta20_50_ptime_data)    
    scheduler1.set_distance_matrix(distances.twenty_machine_matrix)
    
    scheduler1.runs = 1
    scheduler1.display_schedule = 1
    scheduler1.display_convergence = 0
    scheduler1.enable_travel_time = 1
    scheduler1.create_txt_file = 0
    
    scheduler1.stagnation_limit = 60
    scheduler1.activate_termination = 1
    
    print(scheduler1.operation_data)
    for _ in range(scheduler1.runs):
        chromosome1 = scheduler1.GeneticAlgorithm()
        
    # print(f'best Cmax = {chromosome1.fitness}')
    
    # print('random generated numbers:',chromosome1.encoded_list)
    # print(f'ranked list : {chromosome1.ranked_list}\n operation_index :{chromosome1.operation_index_list},\n operation object{chromosome1.operation_schedule}\n')
    # print(f'machine sequence: {chromosome1.machine_sequence}\n ptime sequence: {chromosome1.ptime_sequence}\n Cmax: {chromosome1.Cmax}\n wait time: {chromosome1.wait_time}\n idle_time: {chromosome1.idle_time}\n fitness: {chromosome1.fitness}')
    
    
    
if __name__ == '__main__':
    main()