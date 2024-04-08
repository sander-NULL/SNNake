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
BEST_SIZE = 100

#amount of individuals each ANN can create as offspring
OFFSPRING_SIZE = 9

#generation 1 is created randomly
#from generation 2 onwards, the generation consists of:    a) the BEST_SIZE best from the previous generation
#                                                          b) BEST_SIZE * OFFSPRING_SIZE slightly modified versions of a)
#                                                          c) randomly generated individuals
#note that we must have (OFFSPRING_SIZE + 1) * BEST_SIZE <= POP_SIZE

#number of rounds each ANN plays
MAX_ROUNDS = 20

#maximal number of generations
MAX_GENS = 40

#the mutation rate controls how big the alteration of the weight matrices is
MUT_RATE = 0.001

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

#create folder for saving the generations
os.mkdir("./generations")

f = open("./generations/stats.txt", "a")
f.write("POP_SIZE = " + str(POP_SIZE) + "\n")
f.write("BEST_SIZE = " + str(BEST_SIZE) + "\n")
f.write("OFFSPRING_SIZE = " + str(OFFSPRING_SIZE) + "\n")
f.write("MAX_ROUNDS = " + str(MAX_ROUNDS) + "\n")
f.write("MAX_GENS = " + str(MAX_GENS) + "\n")
f.write("MUT_RATE = " + str(MUT_RATE) + "\n\n")
f.close()


#dimension of unit block in pixels
snake_block = 10

#offsets of playing field in pixels
offset_x = 20
offset_y = 35

#dimensions of playing field in blocks
field_width = 20
field_height = 20

def sigmoid(x):
    return 1/(1+np.exp(-x))

#create version of sigmoid for vectors and matrices
vsigmoid = np.vectorize(sigmoid)

#alters the entries of a matrix
def mutate(matrix, rate):
    for i in range(0,matrix.shape[0]):
        for j in range(0,matrix.shape[1]):
            matrix[i,j] += matrix[i,j] * np.random.normal(0, rate)
            
def get_score_moves(W1, W2, W3):
    game_over = False

    #set starting position and speed
    x1 = random.randrange(field_width) * snake_block + offset_x
    y1 = random.randrange(field_height) * snake_block + offset_y
    snake_Head = [x1, y1]
    x1_change = 0
    y1_change = 0
    usr_x1_change = 0
    usr_y1_change = 0

    snake_List = [snake_Head]
    length_of_snake = 1

    #create random block for first food position and map it onto field blocks without snake blocks (which is just the head)
    #the following code enumerates <food_block> blocks excluding the ones belonging to the snake
    #starting from the top left corner
    food_block = random.randrange(field_width * field_height - length_of_snake) + 1
    i = -1
    for j in range(food_block):
        i += 1
        while [offset_x + (i % field_width) * snake_block, offset_y + (i // field_width) * snake_block] in snake_List:
            i += 1
        foodx = offset_x + (i % field_width) * snake_block
        foody = offset_y + (i // field_width) * snake_block

    moves = 0
    timeout = 0
    while not game_over:
        #set input vector for ANN
        #(x1,y1) = coords of head
        #(snake_List[0][0], snake_List[0][1]) = coords of head
        #length_of_snake = well...
        #(x1_change, y1_change) = direction of movement
        #(foodx, foody) = coords of food

        #first we normalize the inputvector
        #all points (x,y) are mapped in the square [-1, 1) x (-1, 1]
        #the asymmetry in the half open intervals is due to pyGame's coordinate system having the origin in the top left corner
        head_x_normalized = 2/field_width * (x1-offset_x)/snake_block - 1   #maps is to the range [-1, 1)
        head_y_normalized = 1 - 2/field_height * (y1-offset_y)/snake_block  #maps it to the range (-1, 1]
        tail_x_normalized = 2/field_width * (snake_List[0][0]-offset_x)/snake_block - 1     #maps is to the range [-1, 1)
        tail_y_normalized = 1 - 2/field_height * (snake_List[0][1]-offset_y)/snake_block    #maps it to the range (-1, 1]
        los_normalized = length_of_snake / (field_width*field_height)
        x1_change_normalized = x1_change/snake_block
        y1_change_normalized = y1_change/snake_block
        food_x_normalized = 2/field_width * (foodx-offset_x)/snake_block - 1    #maps is to the range [-1, 1)
        food_y_normalized = 1 - 2/field_height * (foody-offset_y)/snake_block   #maps it to the range (-1, 1]

        inputvector = np.array([head_x_normalized, head_y_normalized, tail_x_normalized, tail_y_normalized, los_normalized, x1_change_normalized, y1_change_normalized, food_x_normalized, food_y_normalized, 1]).reshape(10,1)

        #apply activation function to each entry
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

        #see what index contains maximal value
        key = np.argmax(outputvector)
        if (key == 0):
            #move left
            if (x1_change == 0):
                usr_x1_change = -snake_block
                usr_y1_change = 0
        elif (key == 1):
            #move up
            if (y1_change == 0):
                usr_y1_change = -snake_block
                usr_x1_change = 0
        elif (key == 2):
            #move right
            if (x1_change == 0):
                usr_x1_change = snake_block
                usr_y1_change = 0
        elif (key == 3):
            #move down
            if (y1_change == 0):
                usr_y1_change = snake_block
                usr_x1_change = 0

        #change snake coordinates
        #change x1_change and y1_change only once per tick to prevent snake from instantly moving in opposite direction and hitting itself
        x1_change = usr_x1_change
        y1_change = usr_y1_change    
        x1 += x1_change
        y1 += y1_change

        #check whether snake hits border
        if x1 >= offset_x + field_width * snake_block or x1 < offset_x \
        or y1 >= offset_y + field_height * snake_block or y1 < offset_y:
            game_over = True

        snake_Head = [x1, y1]
        snake_List.append(snake_Head)
        if len(snake_List) > length_of_snake:
            del snake_List[0]

        for x in snake_List[:-1]:
            if x == snake_Head:
                game_over = True

        moves += 1
        timeout += 1
        #check whether snake ate food
        if (x1 == foodx) and (y1 == foody):
            if field_width * field_height - length_of_snake == 0:
                #perfect game
                print("Perfect Game!")
                game_over = True
            else:
                #create random block for new food position and map it onto field blocks without snake blocks
                #the following code enumerates <food_block> blocks excluding the ones belonging to the snake
                #starting from the top left corner
                food_block = random.randrange(field_width * field_height - length_of_snake) + 1
                i = -1
                for j in range(food_block):
                    i += 1
                    while [offset_x + (i % field_width) * snake_block, offset_y + (i // field_width) * snake_block] in snake_List:
                        i += 1
                foodx = offset_x + (i % field_width) * snake_block
                foody = offset_y + (i // field_width) * snake_block

            length_of_snake += 1
            timeout = 0

        #to prevent the ANN from running in circles we implement a timeout
        if timeout == 100:
            #ANN made 100 moves without finding food
            game_over = True

    return [length_of_snake-1, moves]    

#stores the best weight matrices of the last generation along with their total scores
#initially it stores dummy scores of -1
#data will be stored in the format [fitness, W1, W2, W3]
best_list_data = [[-1]]*BEST_SIZE

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
            W1 = np.random.uniform(-1, 1, (8,10))
            W2 = np.random.uniform(-1, 1, (8,9))
            W3 = np.random.uniform(-1, 1, (4,9))
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
                

        fitness = 0

        for game_round in range(0, MAX_ROUNDS):
            #Let the ANN play MAX_ROUNDS rounds
            pair = get_score_moves(W1, W2, W3)
            fitness += pair[0]
        
        fitness /= MAX_ROUNDS

        if fitness > best_list[0]:
            #ANN is under the BEST_SIZE best so far
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
            print("New best list: Min = " + str("{:7.5f}".format(best_list[0])) + "\tMedian = " + str("{:7.5f}".format(best_list[int(BEST_SIZE/2)])) + "\tMax = " + str("{:7.5f}".format(best_list[BEST_SIZE-1])))
        print("Generation " + str(gen) + " ANN # " + str(cnt))
        cnt+=1

    best_list_data = tmp_best_list_data

    #save the best BEST_SIZE weight matrices
    #first create folder for the generation
    os.mkdir("./generations/Gen_" + str(gen))
    for j in range(len(tmp_best_list_data)):
        np.savez("./generations/Gen_" + str(gen) + "/" + str(best_list_data[j][4]) + "_fit=" + str(best_list_data[j][0]), W1=best_list_data[j][1], W2=best_list_data[j][2], W3=best_list_data[j][3])
    f = open("./generations/stats.txt", "a")
    f.write("Generation = " + str("{:03d}".format(gen)) + "\tMin = " + str("{:7.5f}".format(best_list[0])) + "\tMedian = " + str("{:7.5f}".format(best_list[int(BEST_SIZE/2)])) + "\tMax = " + str("{:7.5f}".format(best_list[BEST_SIZE-1])) + "\n")
    f.close()
