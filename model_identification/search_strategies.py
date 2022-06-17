import torch
import itertools
from utils import torch_random_tensor

class BatchRandomSearch:
    def __init__(self, ranges, batch_size=4096, num_iterations=10):
    
        self.batch_size = batch_size
        self.done = False
        self.num_iterations = num_iterations
        self.completed = 0
        self.candidate_ranges = []
        for _, props in ranges.items():
            repeat = 1
            start, end = props['start'], props['end']
            if 'repeat' in props.keys():
                repeat = props['repeat']
            self.candidate_ranges.append((start, end, (repeat, 1)))
        
    def get_query_points(self):
        query_points = []
        for _ in range(self.batch_size):
            qp = []
            for candidate in self.candidate_ranges:
                qp.append(torch_random_tensor(*candidate[:]))
            query_points.append(torch.cat(qp, axis=0))
        self.completed += 1
        return query_points, self.completed == self.num_iterations

class BatchExhaustiveSearch:
    def __init__(self, ranges, batch_size=4096):
        candidates_per_prop = []        
        for _, props in ranges.items():
            start, end, step = props['start'], props['end'], props['step']
            candidates_per_prop.extend(torch.linspace(start, end, step).repeat(props['repeat']).view(props['repeat'], -1).tolist())

        self.batch_size = batch_size
        self.search_list = iter(itertools.product(*candidates_per_prop))
        self.done = False
        
    def get_query_points(self):
        query_points = []
        for _ in range(self.batch_size):
            try:
                query_points.append(next(self.search_list))
            except StopIteration:
                self.done = True
        return query_points, self.done

# class AdaptiveSearch:
#     pass

# ranges = {
#             'friction': {
#                 'start': 0,
#                 'end': 10,
#                 'step': 64, 
#                 'repeat': 5
#             },
            
#             'mass': {
#                 'start': 0,
#                 'end': 1,
#                 'step': 10, 
#                 'repeat': 1
#             }
#         }

# qps = []
# es = BatchRandomSearch(ranges, iterations=10, batch_size=2)
# k = 0
# while True:
#     qp, done = es.get_query_points()
#     # print(qp)
#     qps.extend(qp)
#     # print(k+1)
#     if done:
#         break
#     k = k+1
    
# print(len(qps))
    
    


