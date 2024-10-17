CXX      = g++
CXX_FILE = $(wildcard *.cpp)
TARGET   = $(patsubst %.cpp,%,$(CXX_FILE))
CXXFLAGS = -g -Wall -Werror -pedantic-errors -fmessage-length=0
TARGETFLAGS = -lsfml-graphics -lsfml-window -lsfml-system

all:
	$(CXX) $(CXXFLAGS) $(CXX_FILE) -o $(TARGET) $(TARGETFLAGS)
clean:
	rm -f $(TARGET) $(TARGET).exe
