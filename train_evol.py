'''
Created on May 11, 2022

@author: sander
'''

import os
import shutil
import random
import numpy as np
import snake_core as sc
import csv

'''
To do:
1. perfect game notification
2. What if (OFFSPRING_SIZE + 1) * BEST_SIZE <= POP_SIZE does not hold?
3. Best lists values initialization with -1 is bad
'''
#size of each population
POP_SIZE = 10

#amount of individuals that get to reproduce
BEST_SIZE = 2

#amount of individuals each NN can create as offspring
OFFSPRING_SIZE = 3

#generation 1 is created randomly
#from generation 2 onwards, the current generation consists of:    a) the BEST_SIZE best from the previous generation
#                                                                  b) BEST_SIZE * OFFSPRING_SIZE slightly modified versions of a)
#                                                                  c) randomly generated individuals
#note that we must have (OFFSPRING_SIZE + 1) * BEST_SIZE <= POP_SIZE

#number of rounds each NN plays
MAX_ROUNDS = 20

#maximal number of generations
MAX_GENS = 40

#the mutation rate controls how big the alteration of the weight matrices is
MUT_RATE = 0.05

if not os.path.exists('./fit_data/test_bench.csv'):
    if input('No test bench data found for computing fitness. Generate? (Y/n) ') == 'Y':
        import gen_fit_data
    else:
        print('Can\'t proceed without test bench data. Exiting.')
        exit()

test_bench = []
#for calculating maximal possible score
max_score = 0
with open('./fit_data/test_bench.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            invec = np.array(row[0:6], dtype=int)
            pos_score = np.array(row[6:10], dtype=int)
            max_score += pos_score.max()
            test_bench.append((invec, pos_score))
    
#check whether there is data from a previous run
if os.path.exists('./generations'):
    if input('Data from a previous run seems to exist in ./generations/. Erase this folder? (Y/n) ') == 'Y':
        print('Erasing data...')
        shutil.rmtree('./generations')
        print('Done.')
    else:
        print('Keeping data. Exiting.')
        exit()

#create folder for saving data
os.mkdir('./generations')

with open('./generations/stats.txt', 'a') as f:
    f.write(f'POP_SIZE = {POP_SIZE}\n')
    f.write(f'BEST_SIZE = {BEST_SIZE}\n')
    f.write(f'OFFSPRING_SIZE = {OFFSPRING_SIZE}\n')
    f.write(f'MAX_ROUNDS = {MAX_ROUNDS}\n')
    f.write(f'MAX_GENS = {MAX_GENS}\n')
    f.write(f'MUT_RATE = {MUT_RATE}\n\n')

#alters the entries of an array
def mutate(array, rate):
    mutated = np.copy(array)
    with np.nditer(mutated, op_flags=['readwrite']) as it:
        for x in it:
            x[...] += np.random.normal(0, rate)
    return mutated

def get_fitness(test_bench, W1, b1, W2, b2, W3, b3):
    fitness = 0
    for input_data, pos_score in test_bench:

        #inputvector = np.array([head_x_n, head_y_n, tail_x_n, tail_y_n, snake_length_n, head_x_change_n, head_y_change_n, food_x_n, food_y_n, 1]).reshape(10,1)
        inputvector = sc.normalize(input_data)

        #multiply W1 with inputvector and apply activation function to each entry
        layer1_output = np.tanh(np.matmul(W1, inputvector) + b1)

        layer2_output = np.tanh(np.matmul(W2, layer1_output) + b2)

        outputvector = sc.vsigmoid(np.matmul(W3, layer2_output) + b3)

        #rescale outputvector to get probabilities
        total_sum = outputvector[0] + outputvector[1] + outputvector[2] + outputvector[3]
        outputvector = 1/total_sum * outputvector
        
        direction = np.zeros(4, dtype = int)
        direction[np.argmax(outputvector)] = 1
        
        fitness += np.dot(direction, pos_score)
    return fitness/max_score
                    
                
def get_fitness_old(W1, b1, W2, b2, W3, b3):
    fitness = 0
    for _ in range(0, MAX_ROUNDS):
        #Let the NN play MAX_ROUNDS rounds
        game_over = False
        #set starting position
        head_x = random.randrange(sc.FIELD_WIDTH)
        head_y = random.randrange(sc.FIELD_HEIGHT)
        snake_head = [head_x, head_y]
        head_x_change = 0
        head_y_change = 0

        snake_list = [snake_head]
        snake_length = 1

        #create random block for first food position and map it onto field blocks without snake blocks (which is just the head)
        #the following code enumerates <food_block> blocks excluding the ones belonging to the snake
        #starting from the top left corner
        food_block = random.randrange(sc.FIELD_WIDTH * sc.FIELD_HEIGHT - snake_length) + 1
        i = -1
        for _ in range(food_block):
            i += 1
            while [i % sc.FIELD_WIDTH, i // sc.FIELD_WIDTH] in snake_list:
                i += 1
            food_x = i % sc.FIELD_WIDTH
            food_y = i // sc.FIELD_WIDTH
        
        #initial_distance = abs(head_x - food_x) + abs(head_y - food_y)
        timeout = 0
        moves = 0
        while not game_over:
            #set input vector for NN
            #(head_x,head_y) = coords of head
            #(snake_list[0][0], snake_list[0][1]) = coords of head
            #snake_length = well...
            #(head_x_change, head_y_change) = direction of movement
            #(food_x, food_y) = coords of food
    
            #first we normalize the inputvector
            #all points (x,y) are mapped in the square [-1, 1) x (-1, 1]
            #the asymmetry in the half open intervals is due to pyGame's coordinate system having the origin in the top left corner
            head_x_n = 2/sc.FIELD_WIDTH * head_x - 1   #maps is to the range [-1, 1)
            head_y_n = 1 - 2/sc.FIELD_HEIGHT * head_y  #maps it to the range (-1, 1]
            #tail_x_n = 2/sc.FIELD_WIDTH * snake_list[0][0] - 1     #maps is to the range [-1, 1)
            #tail_y_n = 1 - 2/sc.FIELD_HEIGHT * snake_list[0][1]    #maps it to the range (-1, 1]
            #snake_length_n = snake_length / (sc.FIELD_WIDTH * sc.FIELD_HEIGHT)
            head_x_change_n = head_x_change
            head_y_change_n = head_y_change
            food_x_n = 2/sc.FIELD_WIDTH * food_x - 1   #maps is to the range [-1, 1)
            food_y_n = 1 - 2/sc.FIELD_HEIGHT * food_y  #maps it to the range (-1, 1]
    
            #inputvector = np.array([head_x_n, head_y_n, tail_x_n, tail_y_n, snake_length_n, head_x_change_n, head_y_change_n, food_x_n, food_y_n, 1]).reshape(10,1)
            inputvector = np.array([head_x_n, head_y_n, head_x_change_n, head_y_change_n, food_x_n, food_y_n])
            
            #multiply W1 with inputvector and apply activation function to each entry
            layer1_output = np.tanh(np.matmul(W1, inputvector) + b1)
    
            layer2_output = np.tanh(np.matmul(W2, layer1_output) + b2)
    
            outputvector = sc.vsigmoid(np.matmul(W3, layer2_output) + b3)
    
            #rescale outputvector to get probabilities
            total_sum = outputvector[0] + outputvector[1] + outputvector[2] + outputvector[3]
            outputvector = 1/total_sum * outputvector
    
            #check at what index maximal value is contained
            key = np.argmax(outputvector)
            if (key == 0):
                #move left
                if (head_x_change == 0):
                    head_x_change = -1
                    head_y_change = 0
            elif (key == 1):
                #move up
                if (head_y_change == 0):
                    head_y_change = -1
                    head_x_change = 0
            elif (key == 2):
                #move right
                if (head_x_change == 0):
                    head_x_change = 1
                    head_y_change = 0
            elif (key == 3):
                #move down
                if (head_y_change == 0):
                    head_y_change = 1
                    head_x_change = 0
    
            #change snake coordinates
            head_x += head_x_change
            head_y += head_y_change
    
            #check whether snake hits border
            if head_x >= sc.FIELD_WIDTH  or head_x < 0 \
            or head_y >= sc.FIELD_HEIGHT or head_y < 0:
                game_over = True
    
            snake_head = [head_x, head_y]
            snake_list.append(snake_head)
            if len(snake_list) > snake_length:
                del snake_list[0]
    
            for x in snake_list[:-1]:
                if x == snake_head:
                    game_over = True
    
            timeout += 1
            moves += 1
            #check whether snake ate food
            if (head_x == food_x) and (head_y == food_y):
                if sc.FIELD_WIDTH * sc.FIELD_HEIGHT - snake_length == 0:
                    #perfect game
                    print('Perfect Game!')
                    game_over = True
                else:
                    #create random block for new food position and map it onto field blocks without snake blocks
                    #the following code enumerates <food_block> blocks excluding the ones belonging to the snake
                    #starting from the top left corner
                    food_block = random.randrange(sc.FIELD_WIDTH * sc.FIELD_HEIGHT - snake_length) + 1
                    i = -1
                    for _ in range(food_block):
                        i += 1
                        while [i % sc.FIELD_WIDTH, i // sc.FIELD_WIDTH] in snake_list:
                            i += 1
                    food_x = i % sc.FIELD_WIDTH
                    food_y = i // sc.FIELD_WIDTH
                    #initial_distance = abs(head_x - food_x) + abs(head_y - food_y)
    
                snake_length += 1
                timeout = 0
    
            #to prevent the NN from running in circles we implement a timeout
            if timeout == 100:
                #NN made 100 moves without finding food
                game_over = True
            
            #distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        #if distance < initial_distance:
        #    fitness+=snake_length - 1 + distance/(sc.FIELD_HEIGHT + sc.FIELD_WIDTH) - moves/100
        #elif distance > initial_distance:
        #    fitness+=snake_length - 1 - distance/(sc.FIELD_HEIGHT + sc.FIELD_WIDTH) - moves/100
        #else:
        #    fitness+=snake_length - 1
        fitness += snake_length - 1 + moves/(moves + 10)
        #print('(fitness, score, moves) in round ' + str(_) + ' = (' + str(fitness) + ', ' + str(snake_length-1) + ', ' + str(moves) + ')')
        #input()
    fitness/=MAX_ROUNDS    
    return fitness

#stores the best weight matrices of the last generation along with their total scores
#initially it stores dummy scores of -1
#data will be stored in the format [fitness, W1, W2, W3]
best_list_data = [[-1]]*BEST_SIZE

#define format strings for nice output later
gen_fstr = '{:=' + str(1+int(np.log10(MAX_GENS))) + '}'
nncnt_fstr = '{:=' + str(int(np.log10(POP_SIZE))) + '}'

#stores the BEST_SIZE best fitness values of this generation
#initially it stores dummy scores of -1 such that the first fitness values achieved automatically get in the list
best_list = [-1]*BEST_SIZE

#stores the best weight matrices of this generation along with their fitness values
#initially it stores dummy fitness values of -1
#data will be stored in the format [fitness, W1, b1, W2, b2, W3, b3, idx]
best_list_data = [[-1]]*BEST_SIZE

for gen in range(1, MAX_GENS + 1):    
    cnt = 0
    while cnt < POP_SIZE:
        if gen == 1:            
            #in the first generation create weight matrices randomly
            #one input layer with 6 neurons
            #two hidden layers with 8 neurons each
            #one output layer with 4 neurons
            W1 = np.random.uniform(-1, 1, (8,6))
            b1 = np.zeros(8)
            W2 = np.random.uniform(-1, 1, (8,8))
            b2 = np.zeros(8)
            W3 = np.random.uniform(-1, 1, (4,8))
            b3 = np.zeros(4)
            idx = f'gen_{gen}-cnt_{cnt}'
        else:
            #from generation 2 onwards
            if cnt == 0:
                for cnt in range(0, BEST_SIZE):
                    #take the unmodified versions of the previous generation
                    best_list_data[cnt][7] += '-U'
                cnt += 1
            if cnt in range(BEST_SIZE, (OFFSPRING_SIZE+1)*BEST_SIZE):
                #subsequently take slightly altered offspring
                W1 = mutate(best_list_data[cnt%BEST_SIZE][1], MUT_RATE)
                b1 = mutate(best_list_data[cnt%BEST_SIZE][2], MUT_RATE)
                W2 = mutate(best_list_data[cnt%BEST_SIZE][3], MUT_RATE)
                b2 = mutate(best_list_data[cnt%BEST_SIZE][4], MUT_RATE)
                W3 = mutate(best_list_data[cnt%BEST_SIZE][5], MUT_RATE)
                b3 = mutate(best_list_data[cnt%BEST_SIZE][6], MUT_RATE)

                idx = best_list_data[cnt%BEST_SIZE][7] + f'-{cnt//BEST_SIZE}'
            else:
                #generate the rest randomly
                W1 = np.random.uniform(-1, 1, (8,6))
                b1 = np.zeros(8)
                W2 = np.random.uniform(-1, 1, (8,8))
                b2 = np.zeros(8)
                W3 = np.random.uniform(-1, 1, (4,8))
                b3 = np.zeros(4)
                idx = f'gen_{gen}-cnt_{cnt}' 
        fitness = get_fitness(test_bench, W1, b1, W2, b2, W3, b3)
        if fitness > best_list[0]:
            #NN is under the BEST_SIZE best so far
            #replace the worst with the current one	
            for j in range(len(best_list_data)):
                if best_list_data[j][0] == best_list[0]:
                    #now we have found a worst candidate and replace it
                    best_list_data[j] = [fitness, W1, b1, W2, b2, W3, b3, idx]
                    break
            #replace a worst score with the current one
            best_list[0] = fitness
            #sort the list again
            best_list.sort()
        print('Generation ' + gen_fstr.format(gen) + '/' + str(MAX_GENS) + ' NN #' + nncnt_fstr.format(cnt) + ' | Min = ' + '{:+7.5f}'.format(best_list[0]) + ' Median = ' + '{:+7.5f}'.format(best_list[int(BEST_SIZE/2)]) + ' Max = ' + '{:+7.5f}'.format(best_list[BEST_SIZE-1]), end='\r')
        cnt+=1
    
    print()

    #save the best BEST_SIZE weight matrices
    #first create folder for the generation
    os.mkdir(f'./generations/gen_{gen}')
    for j in range(len(best_list_data)):
        np.savez(f'./generations/gen_{gen}/{best_list_data[j][7]}_fit={best_list_data[j][0]}', W1=best_list_data[j][1], b1=best_list_data[j][2],
                 W2=best_list_data[j][3], b2=best_list_data[j][4], W3=best_list_data[j][5], b3=best_list_data[j][6])
    
    with open('./generations/stats.txt', 'a') as f:
        f.write(f'Generation = {gen:03d}\tMin = {best_list[0]:7.5f}\tMedian = {best_list[int(BEST_SIZE/2)]:7.5f}\tMax = {best_list[BEST_SIZE-1]:7.5f}\n')
