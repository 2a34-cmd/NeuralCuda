#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

typedef struct
{
    byte *grayscale;
    byte label;
} image;

/// @brief
/// @param ImagePAth
/// @param LabelPath
/// @param startingPos
/// @return
image *Mnist(string ImagePAth, string LabelPath, int startingPos)
{

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
    return Arr;
}