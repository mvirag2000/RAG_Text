##
## Break book into Chapter files for use as RAG metadata
##
# directory = "data/eliot/" 
# source = directory + "Middlemarch"
directory = "data/eliot/" 
source = directory + "Middlemarch"

book = open(source, 'r')
chapno = 1
chapter = open(directory + "Chapter_1.txt", 'w')
line1 = book.readline() # Skip first CHAPTER heading 

for line in book:
    line = line.rstrip()
    if (line.startswith("CHAPTER")):
        chapter.close()
        print("Closing Chapter " + str(chapno))
        chapno += 1
        chapter = open(directory + "Chapter_" + str(chapno) + ".txt", 'w')
    else:
        chapter.write(line + ' ')
    # if (chapno == 10): break
  
chapter.close()
book.close()