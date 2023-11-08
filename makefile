
#LINUX
COMMON=-O3 -I../include -L../lib -pthread -Wl,-no-as-needed -Wl,-rpath,'$$ORIGIN'/../lib
LIBS = -lmujoco -lglfw -lm
CC = gcc


ROOT = main

all:
	$(CC) $(COMMON) main.c $(LIBS) -o $(ROOT)

main.o:
	$(CC) $(COMMON) -c main.c

clean:
	rm *.o $(ROOT)
