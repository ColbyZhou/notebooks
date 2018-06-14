import sys
import numpy as np
import random

lr = 0.1
gama = 0.9
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def legal(self, row, col):
        if self.x < 0 or self.y < 0 or self.x >= row or self.y >= col:
            return False
        return True
    def __repr__(self):
        return "x:" + str(self.x) + ', y:' + str(self.y)

def left(pos):
    return Pos(pos.x, pos.y - 1)
def right(pos):
    return Pos(pos.x, pos.y + 1)
def up(pos):
    return Pos(pos.x - 1, pos.y)
def down(pos):
    return Pos(pos.x + 1, pos.y)

# x row , y col
reward_table = np.array([
    [0, 1, 0], # 0, 1, 2
    [2,-10,10] # 3, 4, 5
    ])

row = reward_table.shape[0]
col = reward_table.shape[1]
print(row)
print(col)
state_num = row * col
print(state_num)

action_list = [left, right, up, down]
action_name = ['left', 'right', 'up', 'down']
action_num = len(action_list)
q_table = np.zeros((state_num, action_num))

def exploit(q_table, pos):
    index = pos.x * col + pos.y
    action = np.argmax(q_table[index])
    return action

def update_q(q_table, action, reward, cur_pos, new_pos):
    old_index = cur_pos.x * col + cur_pos.y
    old_q_a = q_table[old_index][action]

    new_index = new_pos.x * col + new_pos.y
    max_new_q = np.max(q_table[new_index])

    # new_q_a = (1 - lr) * old_q_a + lr * (reward + gama * new_q)
    new_q_a = old_q_a + lr * (reward + gama * max_new_q - old_q_a)
    q_table[old_index][action] = new_q_a
    return q_table

def visual_pos(pos, row, col):
    v = np.zeros((row, col))
    v[pos.x][pos.y] = '1'
    print(v)

iter = 0
cur_pos = Pos(0, 0)
print(cur_pos)
while iter < 10000:
    # re-initialize
    if iter % 100 == 0:# or (cur_pos.x == 1 and cur_pos.y == 1):
        cur_pos = Pos(0, 0)

    # 1. choose an action
    seed = random.uniform(0, 1)
    method = 0
    if seed > epsilon:
        # exploit
        action = exploit(q_table, cur_pos)
        method = 1
    else:
        # explore
        action = random.randint(0, action_num - 1)
        method = 2
    # 2. perform action
    act = action_list[action]
    new_pos = act(cur_pos)
    if not new_pos.legal(row, col):
        continue
    if method == 1:
        print('exploit:' + action_name[action])
    else:
        print('explore:' + action_name[action])
    # 3. Measure reward
    reward = reward_table[new_pos.x, new_pos.y]
    # 4. update Q
    q_table = update_q(q_table, action, reward, cur_pos, new_pos)

    print("iter " + str(iter) + ', epsilon: ' + str(epsilon)) 
    print(str(cur_pos) + '=====>' + str(new_pos))

    cur_pos = new_pos
    """
    if iter > 20 and iter % 5 == 0:
        epsilon -= 0.02
        if epsilon < 0:
            epsilon = 0.01
    """
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * iter) 

#    visual_pos(cur_pos, row, col)
    print(q_table)
    print("=========================")
    iter += 1

for q_arr in q_table:
    idx = np.argmax(q_arr)
    name = action_name[idx]
    print(name)


"""
for act in action_list:
    new_pos = act(cur_pos)
    if not new_pos.legal(row, col):
        print('error')
    else:
        print(new_pos)
"""