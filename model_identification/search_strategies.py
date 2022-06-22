import torch
import itertools
from utils import torch_random_tensor
from abc import ABC
import numpy as np

class SearchStrategy:
    def __init__(self, strategy_args):
        pass
        
    def get_query_points(self, *args):
        raise NotImplementedError
    


class BatchRandomSearch(SearchStrategy):
    def __init__(self, strategy_args):
        ranges = strategy_args.ranges
        batch_size = getattr(strategy_args, batch_size, 4096)
        num_iterations = getattr(num_iterations, num_iterations, 10)
        
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
        
    def get_query_points(self, *args):
        query_points = []
        for _ in range(self.batch_size):
            qp = []
            for candidate in self.candidate_ranges:
                qp.append(torch_random_tensor(*candidate[:]))
            query_points.append(torch.cat(qp, axis=0))
        self.completed += 1
        return query_points, self.completed == self.num_iterations

class BatchExhaustiveSearch(SearchStrategy):
    def __init__(self, strategy_args):
        ranges = strategy_args.ranges
        batch_size = getattr(strategy_args, "batch_size", 4096)
        candidates_per_prop = []
               
        for _, props in ranges.items():
            for prop in props:
                start, end, step = prop['start'], prop['end'], prop['step']
                candidates_per_prop.extend([torch.linspace(start, end, step).tolist()])

        self.batch_size = batch_size
        self.search_list = iter(itertools.product(*candidates_per_prop))
        self.done = False
        
    def get_query_points(self, *args):
        query_points = []
        for _ in range(self.batch_size):
            try:
                query_points.append(next(self.search_list))
            except StopIteration:
                self.done = True
        return query_points, self.done

class AdaptiveSearch(SearchStrategy):
    def __init__(self, strategy_args):
        self.ranges = strategy_args.ranges
        self.num_iterations = strategy_args.num_iterations
        self.adaptive_step = strategy_args.adaptive_step
        batch_size = getattr(strategy_args, "batch_size", 4096)
        self.zoom_rate = strategy_args.interval_zoom
        self.interval = 1
        candidates_per_prop = []
        for _, props in self.ranges.items():
            for prop in props:
                start, end, step = prop['start'], prop['end'], prop['step']
                candidates_per_prop.extend([torch.linspace(start, end, step).tolist()])

        self.batch_size = batch_size
        self.search_list = iter(itertools.product(*candidates_per_prop))
        self.exhaustive = True
        self.completed = 0
    
    def exhaustive_search(self):
        query_points = []
        for _ in range(self.batch_size):
            try:
                query_points.append(next(self.search_list))
            except StopIteration:
                self.exhaustive = False
        return query_points
    
    def get_query_points(self, *args):
        if self.exhaustive:
            query_points = self.exhaustive_search()
            if len(query_points) != 0:
                return query_points, False
        best_model = args[0]
        
        self.interval *= self.zoom_rate
        # query_points = []
        candidates_per_prop = []
        # TODO: add clipping at boundary values
        for prop in best_model:
            start, end, step = prop - self.interval, prop + self.interval, self.adaptive_step
            
            candidates_per_prop.extend([torch.linspace(start, end, step).tolist()])
        
        self.completed += 1
        self.search_list = iter(itertools.product(*candidates_per_prop))
        self.exhaustive = True
        query_points = self.exhaustive_search()
        
        return query_points, self.completed == self.num_iterations

class CrossEntropySearch(SearchStrategy):
    def __init__(self, strategy_args):
        self.start_mean = strategy_args.start_mean
        self.start_var = strategy_args.start_var
        self.batch_size = strategy_args.batch_size
        self.elite_frac = strategy_args.elite_frac
        self.num_iterations = strategy_args.num_iterations
        self.completed = 0
        
    def get_query_points(self, *args):
        _, qps_error = args
        if qps_error[0] is None:
            
            query_points = np.random.multivariate_normal(self.start_mean, self.start_var, self.batch_size)
                
            print(query_points.shape)            
                
            return query_points, self.completed == self.num_iterations
        
        qps, errors = qps_error
        elite_count = round(self.batch_size * self.elite_frac)
        qps = np.array(qps)
        elite = qps[np.argpartition(errors.cpu().numpy(), elite_count)][:elite_count]
        mean = elite.mean(axis=0)
        cov = np.cov(elite.T)
        print(elite.shape, mean.shape, cov.shape)
        self.completed += 1
        
        return np.random.multivariate_normal(mean, cov, self.batch_size), self.completed == self.num_iterations
        
        
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
    
    


