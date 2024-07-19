from multiprocessing import Pool, Manager
# from multiprocessing import Queue
#  result_queue = multiprocessing.Queue()
class multi_process():
    def __init__(self, num_core, file_list):
        self.num_core = num_core
        len_file = len(file_list)
        self.List_subsets = []
        manager = Manager()
        self.result_queue = manager.Queue()
        for i in range(num_core):
            if i != num_core - 1:
                subset = file_list[(len_file * i) // num_core:(len_file * (i+1)) // num_core]
            else:
                subset = file_list[(len_file * i) // num_core:]
            self.List_subsets.append(subset)
        
    def apply_async(self, single_worker):
        p = Pool(self.num_core)
        for i in range(self.num_core):
            p.apply_async(single_worker, args=(self.List_subsets[i], i), callback=self.result_queue.put)
        p.close()
        p.join()
        results = []
        while not self.result_queue.empty():
            result = self.result_queue.get()
            results.append(result)

        results.sort(key=lambda x: x[1])
        final_results = []
        for result, _ in results:
            final_results += result
        return final_results


    
    

def sub_task(file_list, sub_task_id=0):
    return file_list, sub_task_id

if __name__ == '__main__':
    mp = multi_process(num_core = 4, file_list=list(range(0,24)))
    a = mp.apply_async(sub_task)
    print(a)



