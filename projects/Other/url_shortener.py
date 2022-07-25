# I hate long links.

import pyshorteners as sh

link = input('Paste the link you want to short.')

# Printing the link to be more clear.
print(link)

s = sh.Shortener()
print(s.tinyurl.short(link))
