# this is just a stupid Rock, Paper, Scissors game because
# I'm bored and in quarantine.

import random
import re

wins = 0
while True:
    print("\n")
    print("Rock, Paper, Scissors!")
    print("\n")
    print("Starting game...")
    myChoice = input("Input [R]ock, [P]aper, or [S]cissors:")
    if not re.match("[RrPpSs]", myChoice):
        print("Please choose between R, P or S.")
        continue
    print("You chose " + myChoice)
    choices = ["R", "P", "S"]
    cpu_choice = random.choice(choices)
    print("\n")
    print("I chose " + cpu_choice)
    
    if cpu_choice == str.upper(myChoice):
        print("Tie!")
        
    elif cpu_choice == "R" and myChoice.upper() == "S":
        print("I win!")
        
    elif cpu_choice == "P" and myChoice.upper() == "R":
        print("I win!")
        
    elif cpu_choice == "S" and myChoice.upper() == "P":
        print("I win!")
        
    else:
        wins = wins + 1
        print(" You win!")
        print("\n")
        print("Total wins: " + str(wins))
