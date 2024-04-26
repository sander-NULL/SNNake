'''
Created on Apr 18, 2024

@author: sander
'''
import shutil
import os
import csv
import snake_core as sc

def write_data(path, start = 0, stepsize = 2):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for head_x in range(start, sc.FIELD_WIDTH, stepsize):
            for head_y in range(start, sc.FIELD_HEIGHT, stepsize):
                for food_x in range(start, sc.FIELD_WIDTH, stepsize):
                    for food_y in range(start, sc.FIELD_HEIGHT, stepsize):
                        for (head_x_change, head_y_change) in ((0,0), (-1,0), (1,0), (0,-1), (0,1)):
                            if (head_x, head_y) != (food_x, food_y):
                                if (head_x_change, head_y_change) == (0, 0):
                                    #snake is not moving, beginning of the game
                                    if head_x < food_x:
                                        key = "RIGHT"
                                    elif head_x > food_x:
                                        key = "LEFT"
                                    else:
                                        if head_y < food_y:
                                            key = "DOWN"
                                        else:
                                            #here food_x == head_x, hence food_y == head_y is impossible
                                            key = "UP"
                                            
                                elif (head_x_change, head_y_change) == (-1, 0):
                                    #snake is moving to the left
                                    if head_x < food_x:
                                        #need to make u-turn
                                        if head_y < food_y:
                                            key = "DOWN"
                                        elif head_y > food_y:
                                            key = "UP"
                                        else:
                                            #head and food have same y-coordinate
                                            if head_y < sc.FIELD_HEIGHT/2:
                                                #head is in upper half
                                                key = "DOWN"
                                            else:
                                                #head is in lower half
                                                key = "UP"
                                            
                                    elif head_x > food_x:
                                        key = "LEFT"
                                    else:
                                        if head_y < food_y:
                                            key = "DOWN"
                                        else:
                                            #here food_x == head_x, hence food_y == head_y is impossible
                                            key = "UP"
                                            
                                elif (head_x_change, head_y_change) == (1, 0):
                                    #snake is moving to the right
                                    if head_x < food_x:
                                        key = "RIGHT"
                                    elif head_x > food_x:
                                        #need to make u-turn
                                        if head_y < food_y:
                                            key = "DOWN"
                                        elif head_y > food_y:
                                            key = "UP"
                                        else:
                                            #head and food have same y-coordinate
                                            if head_y < sc.FIELD_HEIGHT/2:
                                                #head is in upper half
                                                key = "DOWN"
                                            else:
                                                #head is in lower half
                                                key = "UP"
                                    else:
                                        if head_y < food_y:
                                            key = "DOWN"
                                        else:
                                            #here food_x == head_x, hence food_y == head_y is impossible
                                            key = "UP"
                                
                                elif (head_x_change, head_y_change) == (0, -1): 
                                    #snake is moving up
                                    if head_x < food_x:
                                        key = "RIGHT"
                                    elif head_x > food_x:
                                        key = "LEFT"
                                    else:
                                        if head_y < food_y:
                                            #need to make u-turn
                                            if head_x < food_x:
                                                key = "RIGHT"
                                            elif head_x > food_x:
                                                key = "LEFT"
                                            else:
                                                #head and food have same x-coordinate
                                                if head_x < sc.FIELD_WIDTH/2:
                                                    #head is in left half
                                                    key = "RIGHT"
                                                else:
                                                    key = "LEFT"
                                        else:
                                            #here food_x == head_x, hence food_y == head_y is impossible
                                            key = "UP"
                                            
                                elif (head_x_change, head_y_change) == (0, 1): 
                                    #snake is moving down
                                    if head_x < food_x:
                                        key = "RIGHT"
                                    elif head_x > food_x:
                                        key = "LEFT"
                                    else:
                                        if head_y < food_y:
                                            key = "DOWN"
                                        else:
                                            #here food_x == head_x, hence head_y > food_y is impossible
                                            #need to make u-turn
                                            if head_x < food_x:
                                                key = "RIGHT"
                                            elif head_x > food_x:
                                                key = "LEFT"
                                            else:
                                                #head and food have same x-coordinate
                                                if head_x < sc.FIELD_WIDTH/2:
                                                    #head is in left half
                                                    key = "RIGHT"
                                                else:
                                                    key = "LEFT"
                                                                                                                        
                                writer.writerow([head_x, head_y, head_x_change, head_y_change, food_x, food_y, key])
    
if os.path.exists("./tt_data"):
    if input("Train and test data seems to already exist. Overwrite (Y/n)? ") == "Y":
        print("Erasing data...")
        shutil.rmtree("./tt_data")
        print("Done erasing.")
    else:
        print("Keeping data. Exiting.")
        exit()
        
os.mkdir("./tt_data/")
write_data("./tt_data/train_annotations.csv")
write_data("./tt_data/test_annotations.csv", start=1, stepsize=3)
print("Done generating train and test data.")

    
