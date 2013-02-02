INCLUDEDIR = ../include
LIBDIR = ../lib
LIBML = $(LIBDIR)/libml.a

CFLAGS = -Wall

CC= cc
AR = ar

SRC =  alloc.c get_line.c read_data.c parser.c sort.c sampling.c unique.c dist.c nn.c tree.c svm.c matrix.c mlg.c rsfn.c rn.c trrn.c ttest.c

OBJS = $(SRC:.c=.o)

all: compile $(LIBML)

compile: $(SRC)
	$(CC) $(CFLAGS) -I$(INCLUDEDIR) -c $(SRC)

$(LIBML): $(OBJS) 
	$(AR) rvu $(LIBML) $(OBJS)
	ranlib $@ 

clean:
	-rm *.o
