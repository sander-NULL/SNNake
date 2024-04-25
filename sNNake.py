'''
Created on May 11, 2022

@author: sander
'''

import pygame
import itertools
import random
import numpy as np

'''
To do:
1. perfect game notification
'''

pygame.init()
 
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
grey = (150,150,150)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

#dimension of unit block
snake_block = 10

#speed of the snake
snake_speed = 5

#offsets of playing field in pixels
offset_x = 20
offset_y = 35

#dimensions of playing field in blocks
FIELD_WIDTH = 20
FIELD_HEIGHT = 20

#dimensions of display
dis_width = FIELD_WIDTH * snake_block + 4 + 2 * offset_x + 400
dis_height = FIELD_HEIGHT * snake_block + 4 + offset_y + 10
 
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('SNNake')
 
clock = pygame.time.Clock()
 
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)
block_font = pygame.font.SysFont("comicsansms", 13) 

def sigmoid(x):
    return 1/(1+np.exp(-x))

# create version of sigmoid for vectors and matrices
vsigmoid = np.vectorize(sigmoid)

def draw_score(score):
    value = score_font.render("Score: " + str(score), True, white)
    dis.blit(value, [offset_x, 0])


def draw_snake(snake_block, snake_list):
    i = 230
    sign = -1
    cnt = 0    
    for x in itertools.islice(reversed(snake_list), 1, None):
        pygame.draw.rect(dis, (i,i,i), [x[0], x[1], snake_block-1,snake_block-1])
        i += sign * 25
        if i <= 25 or i >= 230:
            sign *= -1
        cnt += 1
        foo = block_font.render(str(cnt), True, red)
        dis.blit(foo, [x[0],x[1]])
    for x in itertools.islice(reversed(snake_list),1):
        pygame.draw.rect(dis, white, [x[0], x[1], snake_block-1, snake_block-1]) 
 
def game():
    game_over = False
    game_close = False
 
    #create weight matrices
    #one input layer with 9 neurons
    #two hidden layers with 8 neurons
    #one output layer with 4 neurons
    '''W1 = np.random.uniform(-1, 1, (8,9))
    W2 = np.random.uniform(-1, 1, (8,8))
    W3 = np.random.uniform(-1, 1, (4,8))'''
    
    npzfile = np.load("model_weights.npz")
    W1 = npzfile['W1']
    b1 = npzfile['b1']
    W2 = npzfile['W2']
    b2 = npzfile['b2']
    W3 = npzfile['W3']
    b3 = npzfile['b3']
    
    # set starting position and speed
    head_x = random.randrange(FIELD_WIDTH) * snake_block + offset_x
    head_y = random.randrange(FIELD_HEIGHT) * snake_block + offset_y
    
    snake_Head = [head_x, head_y]
    head_x_change = 0
    head_y_change = 0
 
    snake_list = [snake_Head]
    snake_length = 1
 
    # create random block for first food position and map it onto field blocks without snake blocks (which is just the head)
    # the following code enumerates <food_block> blocks excluding the ones belonging to the snake
    # starting from the top left corner
    food_block = random.randrange(FIELD_WIDTH * FIELD_HEIGHT - snake_length) + 1
    i = -1
    for _ in range(food_block):
        i += 1
        while [offset_x + (i % FIELD_WIDTH) * snake_block, offset_y + (i // FIELD_WIDTH) * snake_block] in snake_list:
            i += 1
        food_x = offset_x + (i % FIELD_WIDTH) * snake_block
        food_y = offset_y + (i // FIELD_WIDTH) * snake_block
        
    while not game_close:
        #snake hits border check
        if head_x >= offset_x + FIELD_WIDTH * snake_block or head_x < offset_x \
        or head_y >= offset_y + FIELD_HEIGHT * snake_block or head_y < offset_y:
            game_over = True
        
        snake_Head = [head_x, head_y]
        snake_list.append(snake_Head)
        if len(snake_list) > snake_length:
            del snake_list[0]
 
        for x in snake_list[:-1]:
            if x == snake_Head:
                game_over = True
        
        #check if snake ate food
        if head_x == food_x and head_y == food_y:
            if FIELD_WIDTH * FIELD_HEIGHT == snake_length:
                #Perfect game?
                game_over = True
            else:
                #create random block for new food position and map it onto field blocks without snake blocks
                #the following code enumerates <food_block> blocks excluding the ones belonging to the snake
                #starting from the top left corner
                food_block = random.randrange(FIELD_WIDTH * FIELD_HEIGHT - snake_length) + 1
                i = -1
                for _ in range(food_block):
                    i += 1
                    while [offset_x + (i % FIELD_WIDTH) * snake_block, offset_y + (i // FIELD_WIDTH) * snake_block] in snake_list:
                        i += 1
                food_x = offset_x + (i % FIELD_WIDTH) * snake_block
                food_y = offset_y + (i // FIELD_WIDTH) * snake_block
            snake_length += 1
                 
        #draw background, border and food
        dis.fill(black)
        pygame.draw.rect(dis, white, [offset_x-3, offset_y-3, FIELD_WIDTH * snake_block + 5, FIELD_HEIGHT * snake_block + 5] , 2)
        pygame.draw.rect(dis, green, [food_x, food_y, snake_block-1, snake_block-1])
 
        #draw snake and score
        draw_snake(snake_block, snake_list)
        draw_score(snake_length - 1)
 
        #update screen
        pygame.display.update()
        
        while game_over == True:
            mesg = font_style.render("Game over! Press 'C' to play again or 'Q' to quit", True, white)
            pygame.display.update(dis.blit(mesg, [150, 0]))
 
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_close = True
                        game_over = False
                    if event.key == pygame.K_c:
                        game()

        #set input vector for ANN
        #(head_x,head_y) = coords of head
        #(snake_list[0][0], snake_list[0][1]) = coords of head
        #snake_length = well...
        #(head_x_change, head_y_change) = direction of movement
        #(food_x, food_y) = coords of food

        #first we normalize the inputvector
        #all 2d points (x,y) are mapped in the square [-1, 1) x (-1, 1]
        #the asymmetry in the half open intervals is due to pyGame's coordinate system having the origin in the top left corner
        head_x_n = 2/FIELD_WIDTH * (head_x-offset_x)/snake_block - 1   #maps is to the range [-1, 1)
        head_y_n = 1 - 2/FIELD_HEIGHT * (head_y-offset_y)/snake_block  #maps it to the range (-1, 1]
        tail_x_n = 2/FIELD_WIDTH * (snake_list[0][0]-offset_x)/snake_block - 1     #maps is to the range [-1, 1)
        tail_y_n = 1 - 2/FIELD_HEIGHT * (snake_list[0][1]-offset_y)/snake_block    #maps it to the range (-1, 1]
        snake_length_n = snake_length / (FIELD_WIDTH * FIELD_HEIGHT)
        head_x_change_n = head_x_change/snake_block
        head_y_change_n = head_y_change/snake_block
        food_x_n = 2/FIELD_WIDTH * (food_x-offset_x)/snake_block - 1    #maps is to the range [-1, 1)
        food_y_n = 1 - 2/FIELD_HEIGHT * (food_y-offset_y)/snake_block   #maps it to the range (-1, 1]
        
        #inputvector = np.array([head_x_n, head_y_n, tail_x_n, tail_y_n, snake_length_n, head_x_change_n, head_y_change_n, food_x_n, food_y_n, 1]).reshape(10,1)
        inputvector = np.array([head_x_n, head_y_n, head_x_change_n, head_y_change_n, food_x_n, food_y_n])
        print("input = ")
        print(inputvector)

        #compute output of first hidden layer
        layer1_output = np.tanh(np.matmul(W1, inputvector) + b1)

        #compute output of second hidden layer
        layer2_output = np.tanh(np.matmul(W2, layer1_output) + b2)

        #compute output
        outputvector = vsigmoid(np.matmul(W3, layer2_output) + b3)

        #rescale outputvector to get probabilities
        total_sum = outputvector[0] + outputvector[1] + outputvector[2] + outputvector[3]
        outputvector = 1/total_sum * outputvector
        print("output = ")
        print(outputvector)
        
        #see what index contains maximal value
        key = np.argmax(outputvector)
        if (key == 0):
            #move left
            print('LEFT')
            if (head_x_change == 0):
                head_x_change = -snake_block
                head_y_change = 0
        elif (key == 1):
            #move up
            print('UP')
            if (head_y_change == 0):
                head_y_change = -snake_block
                head_x_change = 0
        elif (key == 2):
            #move right
            print('RIGHT')
            if (head_x_change == 0):
                head_x_change = snake_block
                head_y_change = 0
        elif (key == 3):
            #move down
            print('DOWN')
            if (head_y_change == 0):
                head_y_change = snake_block
                head_x_change = 0
        
        #change snake coordinates   
        head_x += head_x_change
        head_y += head_y_change
        clock.tick(snake_speed)
    pygame.quit()
    quit()

game()
