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

//[deprecated]
//
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

void InputsToNN(unsigned char***X, string ImagePath, int startingPos,int finishingPos)
{
    ifstream ImageF(ImagePath,ios::binary| ios::in);
    if(!ImageF.is_open()){
        cout << "There are problems";
        return;
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
    cudaMallocManaged((void**)&Arr,sizeof(unsigned char*)*(finishingPos-startingPos +1));
    *X = Arr;

    for (size_t i = startingPos; i <= finishingPos; i++)
    {
        // Arr[i] = (unsigned char*)calloc(Width*hight,sizeof(unsigned char));
        cudaMallocManaged((void**)&Arr[i],Width*hight,sizeof(unsigned char));
       
        ImageF.read((char*)&(Arr[i][0]),sizeof(Arr[i][0])*Width*hight);
        
        (*X)[i] = Arr[i];
    }


    ImageF.close();
    return;

}
void ExpectedFromNN(unsigned char*** X,string LabelPath, int startingPos, int finishingPos)
{
    unsigned char** Arr;
    ifstream LabelF(LabelPath,ios::in|ios::binary);
    unsigned char uc;
    int magicLabel, numOfLabel;

    if(!LabelF.is_open()){
        cout << "There are problems";
        return;
    }
    LabelF.read((char*)&magicLabel,sizeof(magicLabel));
    LabelF.read((char*)&numOfLabel,sizeof(numOfLabel));
    magicLabel = swap(magicLabel);
    numOfLabel = swap(numOfLabel);
    //Arr here will work as normal list pointer to unified memory pointers
    // Arr = (unsigned char**)malloc((numOfLabel - startingPos) * sizeof(unsigned char*));
    cudaMallocManaged((void**)&Arr,(finishingPos - startingPos +1) * sizeof(unsigned char*));
    *X = Arr;

    // LabelF.seekg(startingPos, ios_base::cur);

    for (size_t i = startingPos; i <= finishingPos; i++)
    {
        LabelF.read((char*)&uc,sizeof(uc));
        // the commented code does work when using normal pointer

        // Arr[i] = (unsigned char*)calloc(10,sizeof(unsigned char));


        //However, we want Arr[i] to be cuda pointers
        cudaMallocManaged( (void**)&Arr[i], sizeof(unsigned char)* 10 );
        
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
        (*X)[i] = Arr[i];
    }
    LabelF.close();
    return;
}




void InputsToNN(double***X, string ImagePath, int startingPos,int finishingPos)
{
    ifstream ImageF(ImagePath,ios::binary| ios::in);
    if(!ImageF.is_open()){
        cout << "There are problems";
        return;
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

    unsigned char Arr[Width * hight]; //intermediate pointer (array of char arrays)
    cudaMallocManaged((void**)X,sizeof(double*)*(finishingPos -startingPos +1));//init ITNN
    memset(Arr,0,sizeof(Arr));
    for (size_t i = startingPos; i <= finishingPos; i++)
    {
        // Arr[i] = (unsigned char*)calloc(Width*hight,sizeof(unsigned char));
        cudaMallocManaged((void**)&(X[0][i]),Width*hight*sizeof(double));// init ITNN[i]
        ImageF.read((char*)Arr,sizeof(Arr[0])*Width*hight);
        for(int j=0;j<784;j++){
            X[0][i][j] = (double)Arr[j]/128 -1.00;//normalizing so (0,255) becomes (-1,1)
        }
    }

    ImageF.close();
    return;

}
void ExpectedFromNN(double*** X,string LabelPath, int startingPos, int finishingPos)
{
    ifstream LabelF(LabelPath,ios::in|ios::binary);
    char uc;
    int magicLabel, numOfLabel;

    if(!LabelF.is_open()){
        cout << "There are problems";
        return;
    }
    LabelF.read((char*)&magicLabel,sizeof(magicLabel));
    LabelF.read((char*)&numOfLabel,sizeof(numOfLabel));
    magicLabel = swap(magicLabel);
    numOfLabel = swap(numOfLabel);
    //Arr here will work as normal list pointer to unified memory pointers
    // Arr = (unsigned char**)malloc((numOfLabel - startingPos) * sizeof(unsigned char*));
    cudaMallocManaged((void**)X,sizeof(double)*(finishingPos-startingPos +1));//init EFNN

    // LabelF.seekg(startingPos, ios_base::cur);

    for (size_t i = startingPos; i <= finishingPos; i++)
    {
        LabelF.read((char*)&uc,sizeof(uc));
        // the commented code does work when using normal pointer

        // Arr[i] = (unsigned char*)calloc(10,sizeof(unsigned char));


        //However, we want Arr[i] to be cuda pointers
        cudaMallocManaged((void**)&(X[0][i]), sizeof(double) * 10);// init EFNN[i]
        //couldn't find a smarter way to initialize values of Arr[i] with 0 
        // keep in mind that Arr[i] is (abstract) unified memory pointer
        for(size_t j =0; j <10;j++){
            (*X)[i][j] = 0;
        }
        switch (uc)
        {
            case 0:
                (*X)[i][0] = 1;
                break;
            case 1:
               (*X)[i][1] = 1;
                break;
            case 2:
               (*X)[i][2] = 1;
                break;
            case 3:
               (*X)[i][3] = 1;
                break;
            case 4:
               (*X)[i][4] = 1;
                break;
            case 5:
               (*X)[i][5] = 1;
                break;
            case 6:
               (*X)[i][6] = 1;
                break;
            case 7:
               (*X)[i][7] = 1;
                break;
            case 8:
               (*X)[i][8] = 1;
                break;
            case 9:
               (*X)[i][9] = 1;
                break;
            default:
                cout << "There's error";
                break;
        }
    }
    LabelF.close();
    return;
}