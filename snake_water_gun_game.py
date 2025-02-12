import random

tools = ["gun", "water", "snake"]
computer = random.choice(tools)

toolsDict = {
    "g":"gun", 
    "s":"snake",
    "w":"water"
}

user = input("Type 's' for Snake, 'w' for Water, or 'g' for Gun : ").lower()

if user not in toolsDict:
    print("Invalid input! please enter 's', 'w' or 'g'")
    exit()
else:
    user_choice = toolsDict[user]
    print(f"You Chose : {user_choice}")
    print(f"Computer chose : {computer}")

if(computer == user_choice):
    print("It's a draw...")
else:
    if(computer == 'snake' and toolsDict[user] == 'gun'):
        print("You won! 😊🎉")
    elif(computer == 'gun' and toolsDict[user] == 'snake'):
        print("You lose! 😢💔")
    elif(computer == 'water' and toolsDict[user] == 'snake'):
        print("You won! 😊🎉")
    elif(computer == 'snake' and toolsDict[user] == 'water'):
        print("You lose!  😢💔")
    elif(computer == 'gun' and toolsDict[user] == 'water'):
        print("You won! 😊🎉")
    elif(computer == 'water' and toolsDict[user] == 'gun'):
        print("You lose!  😢💔")
    else:
        print("Something went wrong!")