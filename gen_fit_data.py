'''
Created on May 2, 2024

@author: sander
'''
import shutil
import os
import snake_core as sc
import csv
import numpy as np

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

def location(x1, y1, x2, y2):
    '''assumes (x2, y2) is reachable from (x1, y1) in one move
       returns what key to press to reach (x2, y2)
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
                    lst.extend(2*location(head_x, head_y, food_x, food_y))
                    writer.writerow(lst)

print('Done generating fitness data.')
