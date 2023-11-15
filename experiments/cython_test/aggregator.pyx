from functools import reduce

def aggregator2(all_free_intervals, time_tracker_job):
    p = 0.0001
    limits = list(set([round(elem, 6) for elem in sum(sum(all_free_intervals, []), []) if elem >= time_tracker_job - p]))
    limits.sort()
    for start3 in limits:
        if reduce(lambda x, y: x + y, [1 if len([1 for st, et in slots if st-p <= start3 <= et+p]) else 0 for slots in all_free_intervals]) == 4:
            return start3
