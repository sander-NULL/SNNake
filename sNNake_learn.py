'''
Created on May 11, 2022

@author: sander
'''

#import itertools
import random
import numpy as np

'''
To do:
1. perfect game notification
'''

#number of round each ANN plays
max_rounds = 100

#dimension of unit block
snake_block = 10

#offsets of playing field in pixels
offset_x = 20
offset_y = 35

#dimensions of playing field in blocks
field_width = 20
field_height = 20

def sigmoid(x):
	return 1/(1+np.exp(-x))
	
def mutate(matrix, rate):
	for i in range(0,matrix.shape[0]):
		for j in range(0,matrix.shape[1]):
			matrix[i,j] += matrix[i,j] * np.random.normal(0, rate)

# create version of sigmoid for vectors and matrices
vsigmoid = np.vectorize(sigmoid)

#stores the best weight matrices of the last generation along with their total scores
#initially it stores dummy scores of -1
#data will be stored in the format [total_score, W1, W2, W3]
best_list_data = [[-1]]*100
	
for gen in range(1,41):
	#just stores the 100 best total scores of this generation
	best_list = [-1]*100

	#stores the best weight matrices of this generation along with their total scores
	#initially it stores dummy scores of -1
	#data will be stored in the format [total_score, W1, W2, W3]
	tmp_best_list_data = [[-1]]*100

	for ann_cnt in range(100):
		for offspring_cnt in range(10):
			
			if gen == 1:
				#in the first generation create weight matrices randomly
				#one input layer with 9 neurons
				#two hidden layers with 8 neurons each
				#one output layer with 4 neurons
				W1 = np.random.uniform(-1, 1, (8,9))
				W2 = np.random.uniform(-1, 1, (8,8))
				W3 = np.random.uniform(-1, 1, (4,8))
				#W1 = np.load("25outof100.npz")['W1']
				#W2 = np.load("25outof100.npz")['W2']
			else:				
				W1 = best_list_data[ann_cnt][1]
				mutate(W1, 0.001)
				W2 = best_list_data[ann_cnt][2]
				mutate(W2, 0.001)
				W3 = best_list_data[ann_cnt][3]
				mutate(W3, 0.001)
			
			total_score = 0

			for game_round in range(0, max_rounds):
				game_over = False
				# set starting position and speed
				x1 = random.randrange(field_width) * snake_block + offset_x
				y1 = random.randrange(field_height) * snake_block + offset_y
				snake_Head = [x1, y1]
				x1_change = 0
				y1_change = 0
				usr_x1_change = 0
				usr_y1_change = 0

				snake_List = [snake_Head]
				Length_of_snake = 1

				# create random block for first food position and map it onto field blocks without snake blocks (which is just the head)
				# the following code enumerates <food_block> blocks excluding the ones belonging to the snake
				# starting from the top left corner
				food_block = random.randrange(field_width * field_height - Length_of_snake) + 1
				i = -1
				for j in range(food_block):
					i += 1
					while [offset_x + (i % field_width) * snake_block, offset_y + (i // field_width) * snake_block] in snake_List:
						i += 1
					foodx = offset_x + (i % field_width) * snake_block
					foody = offset_y + (i // field_width) * snake_block
				#foodx = random.randrange(field_width) * snake_block + offset_x
				#foody = random.randrange(field_height) * snake_block + offset_y
				timeout = 0
				while not game_over:
					#set input vector for ANN, snake_List[0] contains coordinates of tail
					inputvector = np.array([x1, y1, snake_List[0][0], snake_List[0][1], Length_of_snake, x1_change, y1_change, foodx, foody]).reshape(9,1)
					#print("input = " + str(inputvector))
					
					layer1 = np.tanh(np.matmul(W1, inputvector))
					layer2 = np.tanh(np.matmul(W2, layer1))
					outputvector = vsigmoid(np.matmul(W3, layer2))
				 	
				 	#rescale outputvector to get probabilities
					total_sum = outputvector[0] + outputvector[1] + outputvector[2] + outputvector[3]
					outputvector = 1/total_sum * outputvector
					#print("output = " + str(outputvector))
					
					#see what index contains maximal value
					key = np.argmax(outputvector)
					if (key == 0):
						#move left
						#print('LEFT')
						if (x1_change == 0):
							usr_x1_change = -snake_block
							usr_y1_change = 0
					elif (key == 1):
						#move up
						#print('UP')
						if (y1_change == 0):
							usr_y1_change = -snake_block
							usr_x1_change = 0
					elif (key == 2):
						#move right
						#print('RIGHT')
						if (x1_change == 0):
							usr_x1_change = snake_block
							usr_y1_change = 0
					elif (key == 3):
						#move down
						#print('DOWN')
						if (y1_change == 0):
							usr_y1_change = snake_block
							usr_x1_change = 0
					
					# change snake coordinates
					#change x1_change and y1_change only once per tick to prevent snake from instantly moving in opposite direction and hitting itself
					x1_change = usr_x1_change
					y1_change = usr_y1_change    
					x1 += x1_change
					y1 += y1_change
					
					# snake hits border check
					if x1 >= offset_x + field_width * snake_block or x1 < offset_x \
					or y1 >= offset_y + field_height * snake_block or y1 < offset_y:
						game_over = True
					
					snake_Head = [x1, y1]
					snake_List.append(snake_Head)
					if len(snake_List) > Length_of_snake:
						del snake_List[0]
				 
					for x in snake_List[:-1]:
						if x == snake_Head:
							game_over = True
					
					timeout += 1
					# check if snake ate food
					if (x1 == foodx) and (y1 == foody):
						if field_width * field_height - Length_of_snake == 0:
							#perfect game
							print("Perfect Game!")
							game_over = True
						else:
							# create random block for new food position and map it onto field blocks without snake blocks
							# the following code enumerates <food_block> blocks excluding the ones belonging to the snake
							# starting from the top left corner
							food_block = random.randrange(field_width * field_height - Length_of_snake) + 1
							i = -1
							for j in range(food_block):
								i += 1
								while [offset_x + (i % field_width) * snake_block, offset_y + (i // field_width) * snake_block] in snake_List:
								    i += 1
							foodx = offset_x + (i % field_width) * snake_block
							foody = offset_y + (i // field_width) * snake_block

						Length_of_snake += 1
						timeout = 0
					
					#to prevent the AI from running in circles we implement a timeout
					if timeout == 100:
						#AI made 100 steps without finding food
						game_over = True

				total_score += Length_of_snake-1

			if total_score > best_list[0]:
				#AI is under the 100 best so far
				#generate its id				
				if gen == 1:
					ann_id = str("{:03d}".format(ann_cnt*10 + offspring_cnt))
				else:
					ann_id = best_list_data[ann_cnt][4] + "-" + str(offspring_cnt)
				#replace the worst with the current one	
				for j in range(len(tmp_best_list_data)):
					if tmp_best_list_data[j][0] == best_list[0]:
						#now we have found a worst candidate and replace it
						tmp_best_list_data[j] = [total_score, W1, W2, W3, ann_id]
						break
				#replace a worst score with the current one
				best_list[0] = total_score
				#sort the list again
				best_list.sort()
				print("New best list: Min = " + str(best_list[0]) + "\tMedian = " + str(best_list[50]) + "\tMax = " + str(best_list[99]))
		print("Generation " + str(gen) + " ANN # " + str(ann_cnt))

	best_list_data = tmp_best_list_data

	#save the best 100 weight matrices
	for j in range(len(tmp_best_list_data)):
		np.savez("./best/Gen_" + str(gen) + "/" + str(best_list_data[j][4]) + "_" + str(best_list_data[j][0]), W1=best_list_data[j][1], W2=best_list_data[j][2], W3=best_list_data[j][3])
	f = open("./best/stats.txt", "a")
	f.write("Generation = " + str("{:03d}".format(gen)) + "\tMin = " + str("{:02d}".format(best_list[0])) + "\tMedian = " + str("{:02d}".format(best_list[50])) + "\tMax = " + str("{:02d}".format(best_list[99])) + "\n")
	f.close()
