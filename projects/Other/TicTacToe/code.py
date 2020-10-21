### WARNING: STILL WORK IN PROGRESS
### This is a TicTacToe game I'm creating to practise on MinMax algorythms.

# this should be intended to play with a numpad, for now

# create board
board = [['.','.','.'],
          ['.','.','.'],
          ['.','.','.']]

def make_move():
  # I want to use numpad keys as coordinates for X and O
  
  move = input("Input number from 1 to 9: ")
  if move == str(1):
    board[0][0] = 'X'
  elif move == str(2):
    board[0][1] = 'X'
  elif move == str(3):
    board[0][2] = 'X'
  elif move == str(4):
    board[1][0] = 'X'
  elif move == str(5):
    board[1][1] = 'X'
  elif move == str(6):
    board[1][2] = 'X'
  elif move == str(7):
    board[2][0] = 'X'
  elif move == str(8):
    board[2][1] = 'X'
  elif move == str(9):
    board[2][2] = 'X'
  print(str(board[0]) + " \n" + str(board[1]) + " \n" + str(board[2]) + " \n")
