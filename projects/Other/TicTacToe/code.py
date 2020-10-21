### WARNING: STILL WORK IN PROGRESS
### This is a TicTacToe game I'm creating to practise on MinMax algorythms.

# this should be intended to play with a numpad, for now

# create board
BoardList = [' '] * 10

def draw_board(board):
  print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3] + "\n")
  print('------------')
  print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6] + "\n")
  print('------------')
  print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9] + "\n")



draw_board(BoardList)
