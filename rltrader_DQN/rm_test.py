import Replay_Memory

memory = Replay_Memory.Replay_Memory(100)

for i in range(100):
    memory.push_num(i)

for i in range(10):
    print(memory.random_data(5))
