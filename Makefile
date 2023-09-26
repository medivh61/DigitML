:WARNINGS = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wformat=2\
 -Winit-self -Wmissing-declarations -Wredundant-decls -Wshadow\
 -Wstrict-overflow=5 -Wswitch-default -Wundef

FLAGS = $(WARNINGS) -std=c++11

SRC = src/main.cpp

sigmoid: 
	g++ $(FLAGS) -Ofast $(SRC) -I include -o main

isrlu:
	g++ $(FLAGS) -DPERS -Ofast $(SRC) -I include -o main

test:
	g++ $(FLAGS) -DTESTS -Ofast $(SRC) -I include -o main -lgtest

all: sigmoid

debug:
	g++ $(FLAGS) -DDEBUG $(SRC) -o main
	./main
