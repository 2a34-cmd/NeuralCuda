flags:=
Obj= ./object/program.o ./object/calc.o ./object/file.o ./object/mnist.o ./object/holders.o
args= mn1/RandomPy.mn1 mnist/img.idx3 mnist/lbl.idx1 0 3 1 100
NeuralTrainer = ./build/NeuralTrainer

all: $(NeuralTrainer)


$(NeuralTrainer): $(Obj)
	nvcc $(flags) $^ -o build/NeuralTrainer
# %.o:%.cu
# 	nvcc $(flags) $^ -c -o $@

./object/holders.o : ./code/holders.cpp
	nvcc $(flags) $^ -c -o $@


./object/%.o : ./code/%.cu
	nvcc $(flags) $^ -c -o $@


clean:
	rm -rf build/NeuralTrainer $(Obj)


run:$(NeuralTrainer)
	$(NeuralTrainer) $(args)