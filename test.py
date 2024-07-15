from data.heat_chart import heat_chart

# m = heat_chart(2021, neighbors = 20)
# m.create_heat_chart()
# print(m.is_hit(77, 25))

import queue

# Create a priority queue
pq = queue.PriorityQueue()

# Insert elements into the priority queue
pq.put((1.3, 'task with priority 1'))
pq.put((0.5, 'task with priority 3'))
pq.put((2, 'task with priority 2'))

# Retrieve elements based on priority
while not pq.empty():
    print(pq.get())
