import random
def takeList():
    li = []
    cache = input()
    while(cache != ""):
        if(cache.isdigit()):
            li.append(int(cache))
        else:
            print("give integers only")
        cache = input()
    print(li)
    return li
def makeNeuralNetwork(Filepath, Nums,activeFunc):
    f = open(Filepath, 'a')
    f.write("s\n")
    f.write(f"net0:{activeFunc};\n")
    leng = len(Nums)
    for m in range(len(Nums)):
        f.write(f"l{m};\n")
        for n in range(Nums[m]):
            f.write(f"nu {n} : {random.uniform(-2.0,2.0):.2f};\n")
    f.write("c\n")
    for i in range(len(Nums)):
        if i== 0:
            continue
        for A in range(Nums[i]):
            for B in range(Nums[i-1]):
                f.write(f"[{i-1}, {B}][{i}, {A}] {random.uniform(-2.0,2.0):.2f} : 0;\n")
    f.close()
    return



# main entry
print("hello to neural network randomizer")
print("to begin with, give the number of neurons in each layer")
V = takeList()
print("excellent!, now give the index of activation function")
print("{Hyperbolic tangent ,1} {Sigmoid ,2} {Rectfied linear unit ,3} {Linear ,4}")
F = 0
while(True):
    Func = input()
    if(Func.isdigit()):
        F = int(Func)
        if(F > 0 and F < 5):
            break
    print("write either 1,2,3, or 4")
print("now give the file path")
P = input()
makeNeuralNetwork(P,V,F)
print("the file is made!")