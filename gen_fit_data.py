'''
Created on May 2, 2024

@author: sander
'''
import shutil
import os
import snake_core as sc
import csv
import numpy as np

def rot_coords(x, y):
    '''assumes square playing field, so sc.FIELD_HEIGHT = sc.FIELD_WIDTH
        rotates positional coords 90 degrees'''
    return (sc.FIELD_HEIGHT - y, x)

def rot_dir(x_c, y_c):
    '''rotates direction of movement by 90 degrees'''
    return (-y_c, x_c)
    
def reachable(x, y, x_c, y_c):
    '''yields the reachable points of snake with head coords (x,y) and
       moving direction (x_c,y_c) not regarding whether it would hit a wall'''    
    if x_c != 0:
        yield (x + x_c, y)
        yield (x, y + 1)
        yield (x, y - 1)
    elif y_c != 0:
        yield (x, y + y_c)
        yield (x + 1, y)
        yield (x - 1, y)
    else:
        yield(x + 1, y)
        yield(x, y + 1)
        yield(x - 1, y)
        yield(x, y - 1)
        
def backray(x, y, x_c, y_c):
    '''yields the ray opposite to the direction of movement'''
    if (x_c, y_c) != (0, 0):
        while (x - x_c <= sc.FIELD_WIDTH - 1 and x - x_c >= 0 and y - y_c <= sc.FIELD_HEIGHT - 1 and y - y_c >= 0):
            x, y = x - x_c, y - y_c
            yield x, y
        
def towards(x1, y1, x2, y2):
    '''yields the directions snake has to take from (x1, y1) to decrease distance to (x2,y2)'''    
    if x1 < x2:
        yield (1, 0)
    elif x1 > x2:
        yield (-1, 0)
    
    if y1 < y2:
        yield (0, 1)
    elif y1 > y2:
        yield (0, -1)
    

def rel_loc(x1, y1, x2, y2):
    '''returns the location of (x2, y2) relative to (x1, y1)
       [LEFT, UP, RIGHT, DOWN]'''
    lst = np.zeros(4, dtype = int)
    if (x1 < x2):
        lst[2] = 1
    elif (x1 > x2):
        lst[0] = 1
    
    if (y1 < y2):
        lst[3] = 1
    elif (y1 > y2):
        lst[1] = 1
    
    return lst

def poss_moves(x_c, y_c):
    '''returns the possible moves the snake can do
       [LEFT, UP, RIGHT, DOWN]'''
    lst = np.zeros(4, dtype = int)
    if x_c < 0:
        lst[0] = lst[1] = lst[3] = 1
    elif x_c > 0:
        lst[1] = lst[2] = lst[3] = 1
    else:
        lst[0] = lst[2] = 1
        if y_c < 0:
            lst[1] = 1
        elif y_c > 0:
            lst[3] = 1
        else:
            lst[1] = lst[3] = 1
    return lst
        
if os.path.exists('./fit_data'):
    if input('Fitness data seems to already exist. Overwrite (Y/n)? ') == 'Y':
        print('Erasing data...')
        shutil.rmtree('./fit_data')
        print('Done erasing.')
    else:
        print('Keeping data. Exiting.')
        exit()
     
os.mkdir('./fit_data/')
with open('./fit_data/test_bench.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for head_x in range(1, sc.FIELD_WIDTH - 1, 3):
        for head_y in range(1, sc.FIELD_HEIGHT - 1, 3):
            for x_c , y_c in ((0,0), (1,0), (0,1), (-1,0), (0, -1)):
                for food_x, food_y in reachable(head_x, head_y, x_c, y_c):
                    lst = [head_x, head_y, x_c, y_c, food_x, food_y]
                    lst.extend(2 * rel_loc(head_x, head_y, food_x, food_y))
                    writer.writerow(lst)
    
    for j in range(0,250):                
        head_x = np.random.randint(1, sc.FIELD_WIDTH - 1)
        head_y = np.random.randint(1, sc.FIELD_HEIGHT - 1)
        x_c , y_c = [(0,0), (1,0), (0,1), (-1,0), (0, -1)][np.random.randint(0,5)]
        
        food_x = np.random.randint(1, sc.FIELD_WIDTH - 1)
        food_y = np.random.randint(1, sc.FIELD_HEIGHT - 1)
        while ((food_x, food_y) == (head_x, head_y) or (food_x, food_y) in reachable(head_x, head_y, x_c, y_c) or 
               (food_x, food_y) in backray(head_x, head_y, x_c, y_c)):
            food_x = np.random.randint(1, sc.FIELD_WIDTH - 1)
            food_y = np.random.randint(1, sc.FIELD_HEIGHT - 1)
        for _ in range(0,4):
            head_x, head_y = rot_coords(head_x, head_y)
            food_x, food_y = rot_coords(food_x, food_y)
            x_c, y_c = rot_dir(x_c, y_c)
            lst = [head_x, head_y, x_c, y_c, food_x, food_y]
            lst.extend(rel_loc(head_x, head_y, food_x, food_y) * poss_moves(x_c, y_c))
            writer.writerow(lst)
                        
print('Done generating fitness data.')
