'''
Created on May 11, 2022

@author: sander
'''

import os
import shutil
import random
import numpy as np

'''
To do:
1. perfect game notification
'''
#size of each population
POP_SIZE = 1000

#amount of individuals that get to reproduce
BEST_SIZE = 200

#amount of individuals each NN can create as offspring
OFFSPRING_SIZE = 3

#generation 1 is created randomly
#from generation 2 onwards, the generation consists of:    a) the BEST_SIZE best from the previous generation
#                                                          b) BEST_SIZE * OFFSPRING_SIZE slightly modified versions of a)
#                                                          c) randomly generated individuals
#note that we must have (OFFSPRING_SIZE + 1) * BEST_SIZE <= POP_SIZE

#number of rounds each NN plays
MAX_ROUNDS = 20

#maximal number of generations
MAX_GENS = 40

#the mutation rate controls how big the alteration of the weight matrices is
MUT_RATE = 0.001

#dimensions of playing field in blocks
FIELD_WIDTH = 20
FIELD_HEIGHT = 20

#check whether there is data from a previous run
if os.path.exists("./generations"):
    usr_input = input("Data from a previous run seems to exist in ./generations/. Erase this folder? (Y/n)\n")
    if usr_input == "Y":
        print("Erasing data...")
        shutil.rmtree("./generations")
        print("Done.")
    else:
        print("Keeping data. Exiting.")
        exit()

#create folder for saving data
os.mkdir("./generations")

f = open("./generations/stats.txt", "a")
f.write("POP_SIZE = " + str(POP_SIZE) + "\n")
f.write("BEST_SIZE = " + str(BEST_SIZE) + "\n")
f.write("OFFSPRING_SIZE = " + str(OFFSPRING_SIZE) + "\n")
f.write("MAX_ROUNDS = " + str(MAX_ROUNDS) + "\n")
f.write("MAX_GENS = " + str(MAX_GENS) + "\n")
f.write("MUT_RATE = " + str(MUT_RATE) + "\n\n")
f.close()

def sigmoid(x):
    return 1/(1+np.exp(-x))

#create version of sigmoid for vectors and matrices
vsigmoid = np.vectorize(sigmoid)

#alters the entries of a matrix
def mutate(matrix, rate):
    for i in range(0,matrix.shape[0]):
        for j in range(0,matrix.shape[1]):
            matrix[i,j] += matrix[i,j] * np.random.normal(0, rate)
            
def get_fitness(W1, W2, W3):
    fitness = 0
    for _ in range(0, MAX_ROUNDS):
        #Let the NN play MAX_ROUNDS rounds
        game_over = False
        #set starting position
        head_x = random.randrange(FIELD_WIDTH)
        head_y = random.randrange(FIELD_HEIGHT)
        snake_head = [head_x, head_y]
        head_x_change = 0
        head_y_change = 0

        snake_list = [snake_head]
        snake_length = 1

        #create random block for first food position and map it onto field blocks without snake blocks (which is just the head)
        #the following code enumerates <food_block> blocks excluding the ones belonging to the snake
        #starting from the top left corner
        food_block = random.randrange(FIELD_WIDTH * FIELD_HEIGHT - snake_length) + 1
        i = -1
        for j in range(food_block):
            i += 1
            while [i % FIELD_WIDTH, i // FIELD_WIDTH] in snake_list:
                i += 1
            food_x = i % FIELD_WIDTH
            food_y = i // FIELD_WIDTH
        
        initial_distance = abs(head_x - food_x) + abs(head_y - food_y)
        timeout = 0
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
            head_x_n = 2/FIELD_WIDTH * head_x - 1   #maps is to the range [-1, 1)
            head_y_n = 1 - 2/FIELD_HEIGHT * head_y  #maps it to the range (-1, 1]
            tail_x_n = 2/FIELD_WIDTH * snake_list[0][0] - 1     #maps is to the range [-1, 1)
            tail_y_n = 1 - 2/FIELD_HEIGHT * snake_list[0][1]    #maps it to the range (-1, 1]
            snake_length_n = snake_length / (FIELD_WIDTH * FIELD_HEIGHT)
            head_x_change_n = head_x_change
            head_y_change_n = head_y_change
            food_x_n = 2/FIELD_WIDTH * food_x - 1   #maps is to the range [-1, 1)
            food_y_n = 1 - 2/FIELD_HEIGHT * food_y  #maps it to the range (-1, 1]
    
            inputvector = np.array([head_x_n, head_y_n, tail_x_n, tail_y_n, snake_length_n, head_x_change_n, head_y_change_n, food_x_n, food_y_n, 1]).reshape(10,1)
    
            #multiply W1 with inputvector and apply activation function to each entry
            layer1_output = np.tanh(np.matmul(W1, inputvector))
    
            #append a one to the vector for the bias
            layer1_output = np.append(layer1_output, 1)
            layer1_output = layer1_output.reshape(9,1)
    
            #apply activation function to each entry
            layer2_output = np.tanh(np.matmul(W2, layer1_output))
    
            #append a one to the vector for the bias
            layer2_output = np.append(layer2_output, 1)
            layer2_output = layer2_output.reshape(9,1)
    
            outputvector = vsigmoid(np.matmul(W3, layer2_output))
    
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
            if head_x >= FIELD_WIDTH  or head_x < 0 \
            or head_y >= FIELD_HEIGHT or head_y < 0:
                game_over = True
    
            snake_head = [head_x, head_y]
            snake_list.append(snake_head)
            if len(snake_list) > snake_length:
                del snake_list[0]
    
            for x in snake_list[:-1]:
                if x == snake_head:
                    game_over = True
    
            timeout += 1
            #check whether snake ate food
            if (head_x == food_x) and (head_y == food_y):
                if FIELD_WIDTH * FIELD_HEIGHT - snake_length == 0:
                    #perfect game
                    print("Perfect Game!")
                    game_over = True
                else:
                    #create random block for new food position and map it onto field blocks without snake blocks
                    #the following code enumerates <food_block> blocks excluding the ones belonging to the snake
                    #starting from the top left corner
                    food_block = random.randrange(FIELD_WIDTH * FIELD_HEIGHT - snake_length) + 1
                    i = -1
                    for j in range(food_block):
                        i += 1
                        while [i % FIELD_WIDTH, i // FIELD_WIDTH] in snake_list:
                            i += 1
                    food_x = i % FIELD_WIDTH
                    food_y = i // FIELD_WIDTH
                    initial_distance = abs(head_x - food_x) + abs(head_y - food_y)
    
                snake_length += 1
                timeout = 0
    
            #to prevent the NN from running in circles we implement a timeout
            if timeout == 100:
                #NN made 100 moves without finding food
                game_over = True
            
            distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        if distance < initial_distance:
            fitness+=snake_length - 1 + distance/(FIELD_HEIGHT + FIELD_WIDTH)
        elif distance > initial_distance:
            fitness+=snake_length - 1 - distance/(FIELD_HEIGHT + FIELD_WIDTH)
        else:
            fitness+=snake_length - 1
    fitness/=MAX_ROUNDS    
    return fitness

#stores the best weight matrices of the last generation along with their total scores
#initially it stores dummy scores of -1
#data will be stored in the format [fitness, W1, W2, W3]
best_list_data = [[-1]]*BEST_SIZE

#define format strings for nice output later
gen_fstr = "{:=" + str(1+int(np.log10(MAX_GENS))) + "}"
nncnt_fstr = "{:=" + str(int(np.log10(POP_SIZE))) + "}"

for gen in range(1,MAX_GENS+1):
    #stores the BEST_SIZE best total scores of this generation
    #initially it stores dummy scores of -1 such that the first scores reached automatically get in the list
    best_list = [-1]*BEST_SIZE

    #stores the best weight matrices of this generation along with their total scores
    #initially it stores dummy scores of -1
    #data will be stored in the format [fitness, W1, W2, W3]
    tmp_best_list_data = [[-1]]*BEST_SIZE
    
    cnt = 0
    while cnt < POP_SIZE:
        if gen == 1:
            #in the first generation create weight matrices randomly
            #one input layer with 9 neurons
            #two hidden layers with 8 neurons each
            #one output layer with 4 neurons
            #last column of each weight matrix is for the bias
            W1 = np.random.uniform(-2, 2, (8,10))
            W2 = np.random.uniform(-2, 2, (8,9))
            W3 = np.random.uniform(-2, 2, (4,9))
        else:
            #from generation 2 onwards
            if cnt in range(0, BEST_SIZE):
                #take the unmodified versions of the previous generation
                W1 = best_list_data[cnt][1]
                W2 = best_list_data[cnt][2]
                W3 = best_list_data[cnt][3]
            elif cnt in range(BEST_SIZE, (OFFSPRING_SIZE+1)*BEST_SIZE):
                #subsequently take slightly altered offspring
                W1 = best_list_data[cnt%BEST_SIZE][1]
                mutate(W1, MUT_RATE)
                W2 = best_list_data[cnt%BEST_SIZE][2]
                mutate(W2, MUT_RATE)
                W3 = best_list_data[cnt%BEST_SIZE][3]
                mutate(W3, MUT_RATE)
            else:
                #generate the rest randomly
                W1 = np.random.uniform(-1, 1, (8,10))
                W2 = np.random.uniform(-1, 1, (8,9))
                W3 = np.random.uniform(-1, 1, (4,9))
                
        fitness = get_fitness(W1, W2, W3)
        if fitness > best_list[0]:
            #NN is under the BEST_SIZE best so far
            #replace the worst with the current one	
            for j in range(len(tmp_best_list_data)):
                if tmp_best_list_data[j][0] == best_list[0]:
                    #now we have found a worst candidate and replace it
                    tmp_best_list_data[j] = [fitness, W1, W2, W3, str(cnt)]
                    break
            #replace a worst score with the current one
            best_list[0] = fitness
            #sort the list again
            best_list.sort()
        print("Generation " + gen_fstr.format(gen) + "/" + str(MAX_GENS) + " NN #" + nncnt_fstr.format(cnt) + " | Min = " + "{:+7.5f}".format(best_list[0]) + " Median = " + "{:+7.5f}".format(best_list[int(BEST_SIZE/2)]) + " Max = " + "{:+7.5f}".format(best_list[BEST_SIZE-1]), end='\r')
        cnt+=1
    
    print()
    best_list_data = tmp_best_list_data

    #save the best BEST_SIZE weight matrices
    #first create folder for the generation
    os.mkdir("./generations/Gen_" + str(gen))
    for j in range(len(tmp_best_list_data)):
        np.savez("./generations/Gen_" + str(gen) + "/" + str(best_list_data[j][4]) + "_fit=" + str(best_list_data[j][0]), W1=best_list_data[j][1], W2=best_list_data[j][2], W3=best_list_data[j][3])
    f = open("./generations/stats.txt", "a")
    f.write("Generation = " + str("{:03d}".format(gen)) + "\tMin = " + str("{:7.5f}".format(best_list[0])) + "\tMedian = " + str("{:7.5f}".format(best_list[int(BEST_SIZE/2)])) + "\tMax = " + str("{:7.5f}".format(best_list[BEST_SIZE-1])) + "\n")
    f.close()
