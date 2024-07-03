#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
using namespace std;
//mnist is data set includes a bunch of randomly sorted images with 28*28 pixels in grayscale
//      from 0 which donates black to 255 which donates white with
//      also labels which indicates the number inside images.
//      it meant to be used to train machine learning models and neural networks are just an example

//struct below defines input and output of neural network
typedef struct
{
    // list pointer of input unsigned chars 
    unsigned char *grayscale;
    // the output number
    // however, it needs to translated into list with size 10 which is mostly 0 except for the element
    //      with the index the same as label
    unsigned char label;
} image;

//a little function to change endianess of integers
int swap(int d)
{
   int a;
   unsigned char *dst = (unsigned char *)&a;
   unsigned char *src = (unsigned char *)&d;

   dst[0] = src[3];
   dst[1] = src[2];
   dst[2] = src[1];
   dst[3] = src[0];

   return a;
}


/// @brief
/// @param ImagePAth
/// @param LabelPath
/// @param startingPos
/// @return

// this method takes both image path which has the inputs and label path which has the ouputs
// in return of list pointer of image struct
image *Mnist(string ImagePAth, string LabelPath, int startingPos)
{
    // mnist image files are written with unsigned chars where the first 16 are
    // (4 for each integer) 1- a number for checking integrty of data called magic number
    //                      2- number of images in file (60,000)
    //                      3&4- width and hieght of images (28,28)
    // mnist label files are written where the first 8 unsigned chars are (4)magic number and (4)number of labels
    ifstream ImageF(ImagePAth);
    ifstream LabelF(LabelPath);
    image *imgPtr;
    unsigned char uc;
    int magicNum, NumOfIm, Width, hight, magicLabel, numOfLabel;

    ImageF >> magicNum >> NumOfIm >> Width >> hight;
    LabelF >> magicLabel >> numOfLabel;

    // imgPtr = (image *)malloc((NumOfIm - startingPos) * sizeof(image));
    cudaMallocManaged((void**)&imgPtr,(NumOfIm - startingPos) * sizeof(image));
    ImageF.seekg(startingPos * Width * hight + 1, ios_base::cur);
    LabelF.seekg(startingPos, ios_base::cur);
    for (size_t i = startingPos; i < NumOfIm; i++)
    {
        // imgPtr[i].grayscale = (unsigned char *)malloc(sizeof(unsigned char) * Width * hight);
        cudaMallocManaged((void**)&(imgPtr[i].grayscale),sizeof(unsigned char) * Width * hight);
        LabelF >> uc;
        imgPtr[i].label = uc;
        ImageF.read(reinterpret_cast<char *>(imgPtr[i].grayscale), sizeof(unsigned char) * Width * hight);
    }
    LabelF.close();
    ImageF.close();
    return imgPtr;
}

unsigned char** InputsToNN(unsigned char**X, string ImagePath, int startingPos)
{
    ifstream ImageF(ImagePath,ios::binary| ios::in);
    if(!ImageF.is_open()){
        cout << "There are problems";
        return X;
    }
    int magicNum, NumOfIm, Width, hight;
    ImageF.read((char*)&magicNum,sizeof(magicNum));
    ImageF.read((char*)&NumOfIm,sizeof(NumOfIm));
    ImageF.read((char*)&Width,sizeof(Width));
    ImageF.read((char*)&hight,sizeof(hight));
    magicNum = swap(magicNum);
    NumOfIm = swap(NumOfIm);
    Width = swap(Width);
    hight = swap(hight);

    unsigned char** Arr;
    cudaMallocManaged((void**)&Arr,sizeof(unsigned char*)*(NumOfIm-startingPos));
    X = Arr;

    for (size_t i = startingPos; i < NumOfIm; i++)
    {
        // Arr[i] = (unsigned char*)calloc(Width*hight,sizeof(unsigned char));
        cudaMallocManaged((void**)&Arr[i],Width*hight,sizeof(unsigned char));
       
        ImageF.read((char*)&(Arr[i][0]),sizeof(Arr[i][0])*Width*hight);
        
        X[i] = Arr[i];
    }


    ImageF.close();
    return Arr;


    // unsigned char **Arr;
    
    // ifstream ImageF(ImagePath);
    // int magicNum, NumOfIm, Width, hight;
    // ImageF >> magicNum >> NumOfIm >> Width >> hight;


    // // Arr is normal list pointer to cuda unified memory pointers
    // Arr = (unsigned char**)malloc(sizeof(unsigned char*)*(NumOfIm-startingPos));

    // // it doesn't work when passing adress of unsigned char** (which makes it of type unsigned char***) as argument
    // //that's why the commented line below isn't used

    // // cuMemAllocManaged(&Arr , sizeof(unsigned char*) * (NumOfIm-startingPos));

    // ImageF.seekg(startingPos * Width * hight + 1, ios_base::cur);
    // for (size_t i = startingPos; i < NumOfIm; i++)
    // {
    //     //the commented line makes Arr[i] normal pointers

    //     // Arr[i] = (unsigned char *)malloc(sizeof(unsigned char) * Width * hight);

    //     //Arr[i] being cuda unified memory pointers
    //     cudaMallocManaged(&Arr[i], sizeof(unsigned char) * Width * hight);
    //     ImageF.read(reinterpret_cast<char *>(Arr[i]), sizeof(unsigned char) * Width * hight);
    // }
    // ImageF.close();
    // return Arr;
}
unsigned char** ExpectedFromNN(unsigned char** X,string LabelPath, int startingPos)
{
    unsigned char** Arr;
    ifstream LabelF(LabelPath,ios::in|ios::binary);
    unsigned char uc;
    int magicLabel, numOfLabel;

    if(!LabelF.is_open()){
        cout << "There are problems";
        return X;
    }
    LabelF.read((char*)&magicLabel,sizeof(magicLabel));
    LabelF.read((char*)&numOfLabel,sizeof(numOfLabel));
    magicLabel = swap(magicLabel);
    numOfLabel = swap(numOfLabel);
    //Arr here will work as normal list pointer to unified memory pointers
    // Arr = (unsigned char**)malloc((numOfLabel - startingPos) * sizeof(unsigned char*));
    cudaMallocManaged((void**)&Arr,(numOfLabel - startingPos) * sizeof(unsigned char*));
    X = Arr;

    // LabelF.seekg(startingPos, ios_base::cur);

    for (size_t i = startingPos; i < numOfLabel; i++)
    {
        LabelF.read((char*)&uc,sizeof(uc));
        // the commented code does work when using normal pointer

        // Arr[i] = (unsigned char*)calloc(10,sizeof(unsigned char));


        //However, we want Arr[i] to be cuda pointers
        cudaMallocManaged( (void**)&Arr[i], sizeof(unsigned char)* 10 );
        X[i] = Arr[i];
        //couldn't find a smarter way to initialize values of Arr[i] with 0 
        // keep in mind that Arr[i] is (abstract) unified memory pointer
        for(size_t j =0; j <10;j++){
            Arr[i][j] = (unsigned char)0;
        }
        switch (uc)
        {
            case 0:
                Arr[i][0] = (unsigned char)1;
                break;
            case 1:
                Arr[i][1] = (unsigned char)1;
                break;
            case 2:
                Arr[i][2] = (unsigned char)1;
                break;
            case 3:
                Arr[i][3] = (unsigned char)1;
                break;
            case 4:
                Arr[i][4] = (unsigned char)1;
                break;
            case 5:
                Arr[i][5] = (unsigned char)1;
                break;
            case 6:
                Arr[i][6] = (unsigned char)1;
                break;
            case 7:
                Arr[i][7] = (unsigned char)1;
                break;
            case 8:
                Arr[i][8] = (unsigned char)1;
                break;
            case 9:
                Arr[i][9] = (unsigned char)1;
                break;
            default:
                cout << "There's error";
                break;
        }
    }
    LabelF.close();
    return Arr;
    // unsigned char** Arr;
    // ifstream LabelF(LabelPath);
    // unsigned char uc;
    // int magicLabel, numOfLabel;

    // LabelF >> magicLabel >> numOfLabel;
    // //Arr here will work as normal list pointer to unified memory pointers
    // Arr = (unsigned char**)malloc((numOfLabel - startingPos) * sizeof(unsigned char*));

    // LabelF.seekg(startingPos, ios_base::cur);

    // for (size_t i = startingPos; i < numOfLabel; i++)
    // {
    //     LabelF >> uc;
    //     // the commented code does work when using normal pointer

    //     // Arr[i] = (unsigned char*)calloc(10,sizeof(unsigned char));


    //     //However, we want Arr[i] to be cuda pointers
    //     cudaMallocManaged( (void**)&Arr[i], sizeof(unsigned char)* 10 );
    //     //couldn't find a smarter way to initialize values of Arr[i] with 0 
    //     // keep in mind that Arr[i] is (abstract) unified memory pointer
    //     for(size_t j =0; j <10;j++){
    //         Arr[i][j] = (unsigned char)0;
    //     }
    //     switch (uc)
    //     {
    //         case 0:
    //             Arr[i][0] = (unsigned char)1;
    //             break;
    //         case 1:
    //             Arr[i][1] = (unsigned char)1;
    //             break;
    //         case 2:
    //             Arr[i][2] = (unsigned char)1;
    //             break;
    //         case 3:
    //             Arr[i][3] = (unsigned char)1;
    //             break;
    //         case 4:
    //             Arr[i][4] = (unsigned char)1;
    //             break;
    //         case 5:
    //             Arr[i][5] = (unsigned char)1;
    //             break;
    //         case 6:
    //             Arr[i][6] = (unsigned char)1;
    //             break;
    //         case 7:
    //             Arr[i][7] = (unsigned char)1;
    //             break;
    //         case 8:
    //             Arr[i][8] = (unsigned char)1;
    //             break;
    //         case 9:
    //             Arr[i][9] = (unsigned char)1;
    //             break;
    //         default:
    //             cout << "There's error";
    //             break;
    //     }
    // }
    // LabelF.close();
    // return Arr;
}