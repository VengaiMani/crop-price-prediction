import random

crops=[]

def set(c):
    global crops
    crops=c

def get(crop,price):
    a=random.randint(0,len(crops))
    crop=crops[a]
    return [crop,str(random.randint(25,350))];
