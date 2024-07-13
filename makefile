flags:=
Obj= program.o calc.o file.o mnist.o holders.o
args= version10.mn1 img.idx3 lbl.idx1 0 0.5 50


all:program


program: $(Obj)
	nvcc $(flags) $^ -o program

%.o:%.cu
	nvcc $(flags) $^ -c -o $@

holders.o:holders.cpp
	nvcc $(flags) $^ -c -o $@


clean:
	rm -rf program *.o


run:program
	./program $(args)