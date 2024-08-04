book = open("data/tolstoy/book-war-and-peace", 'r')
chapno = 1
chapter = open("data/tolstoy/Chapter_1.txt", 'w')
line1 = book.readline() # Skip first line

for line in book:
    line = line.rstrip()
    if (line.startswith("CHAPTER")):
        chapter.close()
        print("Closing Chapter " + str(chapno))
        chapno += 1
        chapter = open("data/tolstoy/Chapter_" + str(chapno) + ".txt", 'w')
    else:
        chapter.write(line + ' ')
    # if (chapno == 10): break
  
chapter.close()
book.close()