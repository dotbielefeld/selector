import instance_sets
import random

set_size = 305 # 282, 492, 50
cut_off = 300

instance_set = [f"i_{c}" for c in range(set_size)]
instance_selector = instance_sets.TimedInstanceSet(instance_set, 4, set_size=set_size, runtime=172900, start_time=0.15, end_time=0.7)
instance_id, instances = instance_selector.get_subset(0, 0, 0)
time = 1
iteration = 0
point_onefive = True
point_seven = True
point_nine = True
for i in range(10000):
    
    if time >= 0.15 * 172900:
        if iteration % 21 == 0:
            time += sum([random.randint((cut_off * 0.1), cut_off) for _ in range(len(instances))]) + 3 * len(instances)
        #else:
            # time += random.randint((cut_off * 0.1), cut_off)
    else:
        if iteration % 21 == 0:
            time += sum([random.randint((cut_off * 0.1), cut_off) for _ in range(len(instances))])
        #else:
            # time += random.randint((cut_off * 0.1), cut_off)

    if time >= 0.15 * 172900 and point_onefive:
        point_onefive = False
        print('\nPoint One Five\n')
    elif time >= 0.7 * 172900 and point_seven:
        point_seven = False
        print('\nPoint Seven\n')
    elif time >= 0.9 * 172900 and point_nine:
        point_nine = False
        print('\nPoint Nine\n')
    iteration += 1  # random.choice([0, 1])
    instance_id, instances = instance_selector.get_subset(instance_id + 1, time, int(instance_id + 1))
    print(iteration, len(instances), time)
    m = iteration % 21
    #print(m)
    if time >= 172900:
        break

print('SO')