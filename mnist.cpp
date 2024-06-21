#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;
//mnist is data set includes a bunch of randomly sorted images with 28*28 pixels in grayscale
//      from 0 which donates black to 255 which donates white with
//      also labels which indicates the number inside images.
//      it meant to be used to train machine learning models and neural networks are just an example

//struct below defines input and output of neural network
typedef struct
{
    // list pointer of input bytes 
    byte *grayscale;
    // the output number
    // however, it needs to translated into list with size 10 which is mostly 0 except for the element
    //      with the index the same as label
    byte label;
} image;

/// @brief
/// @param ImagePAth
/// @param LabelPath
/// @param startingPos
/// @return

// this method takes both image path which has the inputs and label path which has the ouputs
// in return of list pointer of image struct
image *Mnist(string ImagePAth, string LabelPath, int startingPos)
{
    // mnist image files are written with bytes where the first 16 are
    // (4 for each integer) 1- a number for checking integrty of data called magic number
    //                      2- number of images in file (60,000)
    //                      3&4- width and hieght of images (28,28)
    // mnist label files are written where the first 8 bytes are (4)magic number and (4)number of labels
    ifstream ImageF(ImagePAth);
    ifstream LabelF(LabelPath);
    image *imgPtr;
    char *ptr;
    unsigned char uc;
    int magicNum, NumOfIm, Width, hight, magicLabel, numOfLabel;

    ImageF >> magicNum >> NumOfIm >> Width >> hight;
    LabelF >> magicLabel >> numOfLabel;

    imgPtr = (image *)malloc((NumOfIm - startingPos) * sizeof(image));

    ImageF.seekg(startingPos * Width * hight + 1, ios_base::cur);
    LabelF.seekg(startingPos, ios_base::cur);
    for (size_t i = startingPos; i < NumOfIm; i++)
    {
        imgPtr[i].grayscale = (byte *)malloc(sizeof(byte) * Width * hight);
        LabelF >> uc;
        imgPtr[i].label = (std::byte)uc;
        ImageF.read(reinterpret_cast<char *>(imgPtr[i].grayscale), sizeof(byte) * Width * hight);
    }
    LabelF.close();
    ImageF.close();
    return imgPtr;
}

byte** InputsToNN(string ImagePath, int startingPos)
{
    byte **Arr;
    
    ifstream ImageF(ImagePath);
    int magicNum, NumOfIm, Width, hight;
    ImageF >> magicNum >> NumOfIm >> Width >> hight;
    char *ptr;

    Arr = (byte**)malloc(sizeof(byte*)*(NumOfIm-startingPos));
    ImageF.seekg(startingPos * Width * hight + 1, ios_base::cur);
    for (size_t i = startingPos; i < NumOfIm; i++)
    {
        Arr[i] = (byte *)malloc(sizeof(byte) * Width * hight);
        ImageF.read(reinterpret_cast<char *>(Arr[i]), sizeof(byte) * Width * hight);
    }
    ImageF.close();
    return Arr;
}
byte** ExpectedFromNN(string LabelPath, int startingPos)
{
    byte** Arr;
    ifstream LabelF(LabelPath);
    char *ptr;
    unsigned char uc;
    int magicLabel, numOfLabel;

    LabelF >> magicLabel >> numOfLabel;

    Arr = (byte**)malloc((numOfLabel - startingPos) * sizeof(byte*));

    LabelF.seekg(startingPos, ios_base::cur);

    for (size_t i = startingPos; i < numOfLabel; i++)
    {
        LabelF >> uc;
        Arr[i] = (byte*)calloc(10,sizeof(byte));
        switch (uc)
        {
            case 0:
                Arr[i][0] = (byte)1;
                break;
            case 1:
                Arr[i][1] = (byte)1;
                break;
            case 2:
                Arr[i][2] = (byte)1;
                break;
            case 3:
                Arr[i][3] = (byte)1;
                break;
            case 4:
                Arr[i][4] = (byte)1;
                break;
            case 5:
                Arr[i][5] = (byte)1;
                break;
            case 6:
                Arr[i][6] = (byte)1;
                break;
            case 7:
                Arr[i][7] = (byte)1;
                break;
            case 8:
                Arr[i][8] = (byte)1;
                break;
            case 9:
                Arr[i][9] = (byte)1;
                break;
            default:
                cout << "There's error";
                break;
        }
    }
    LabelF.close();
    return Arr;
}