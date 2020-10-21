import random

### WARNING: STILL WORK IN PROGRESS
### This is a TicTacToe game I'm creating to practise on MinMax algorythms.

# this should be intended to play with a numpad, for now

# create board
BoardList = [' '] * 10

def draw_board(board):
  ### draw the board in an easy and intuitive way
  print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3] + "\n")
  print('------------')
  print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6] + "\n")
  print('------------')
  print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9] + "\n")

def isSpaceFree(board, move):
  ### returns True if the space is free
  return board[move] == ' '

def make_move(board, move):
  ### i will probably merge this function and the next in the future because I don't like this design.
  board[move] = 'O'

def player_move(board):
  ### checks for right input and space taken
  move = ' '
  while move not in '1 2 3 4 5 6 7 8 9'.split() or not isSpaceFree(board, int(move)):
    move = input("Make your next move. Select a number on the numpad.")
  return int(move)

def computer_move(board):
  ### this function is just a test, I don't plan on leaving the AI as it is.
    move = ' '
    move = random.randint(1, 9)
    if isSpaceFree(board, int(move)):
      board[move] = 'X'


while True:
  ### as of now I managed to make a semi-functional game player vs computer,
  ### still there is no victory and the AI is just a random function,
  ### so it's still very primitive.
  BoardList = [' '] * 10
  while True:
    draw_board(BoardList)
    move = player_move(BoardList)
    make_move(BoardList, move)
    computer_move(BoardList)
