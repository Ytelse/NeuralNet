import numpy as np
import struct

def createNetwork(params):
    numberOfLayers = len(allparams.keys()) / 6
    matrices = []
    biases = []
    matrixCount = 0
    biasCount = 0
    for i in range(numberOfLayers):
        matrices.append(params['arr_'+str(matrixCount)])
        biases.append(params['arr_'+str(biasCount)])
        matrixCount += 6
        biasCount += 6

    return (matrices,biases)

w = []
c = []

def printMatrices():
    size = len(c)
    print(size)
    for i in range(size):
        width = len(w[i])
        height = len(w[i][0])
        print(width)
        print(height)
        for j in range(width):
            for k in range(height):
                print(w[i][j][k])
        vecSize = len(c[i])
        print(vecSize)
        for j in range(vecSize):
            print(c[i][j])
        

def generateNetwork(params):
    global c,d,w
    
    numberOfLayers = len(allparams.keys()) / 6

    weightlist = [np.sign(params['arr_' + str(index * 6)]) for index in range(numberOfLayers)]
    betalist = [params['arr_' + str((index * 6) + 2)] for index in range(numberOfLayers)]
    gammalist = [params['arr_' + str((index * 6) + 3)] for index in range(numberOfLayers)]
    meanlist = [params['arr_' + str((index * 6) + 4)] for index in range(numberOfLayers)]
    inv_stddevlist = [params['arr_' + str((index * 6) + 5)] for index in range(numberOfLayers)]

    for i in range(numberOfLayers):
        a = gammalist[i] * inv_stddevlist[i]
        b = -(gammalist[i] * inv_stddevlist[i] * meanlist[i]) + betalist[i]
        c.append(b/a)
        
    w = weightlist

def getImageValFromNetwork(vector):

    numberOfLayers = len(w)
    normalizedInput = np.sign(vector - (256 / 2))
    
    outNeuron = normalizedInput

    for i in range(numberOfLayers):

        temp = w[i].T.dot(outNeuron)
        batchnorm = temp + c[i]
        outNeuron = np.sign(batchnorm)# * c[i])

    return np.argmax(outNeuron)


def createInput(bytelist,currentByte,imageSize):
    vec = np.zeros(imageSize)

    for i in range(imageSize):
        vec[i] = ord(bytelist[currentByte])
        currentByte += 1
    
    return vec

def readInt(bytelist):
    num = 0
    count = 0
    for i in range((len(bytelist) - 1) * 8,-8,-8):
        num |= (ord(bytelist[count]) << i)
        count += 1

    return num

def getCorrectValues(bytelist):
    correctList = []
    magicNumber = readInt(bytelist[0:4])
    numberOfElements = readInt(bytelist[4:8])

    currentByte = 8
    for i in range(numberOfElements):
        correctList.append(ord(bytelist[currentByte]))
        currentByte += 1
    return correctList

def readImageHeader(bytelist):
    magicNumber = readInt(bytelist[0:4])
    numberOfImages = readInt(bytelist[4:8])
    imageWidth = readInt(bytelist[8:12])
    imageHeight = readInt(bytelist[12:16])
    return (numberOfImages,imageWidth,imageHeight)

def readImages(bytelist,header,correctValues):
    currentByte = 16
    numberOfImages = header[0]
    imageWidth = header[1]
    imageHeight = header[2]

    valueList = [0 for i in range(10)]
    correctCount = 0
    falseCount = 0
    
    imageSize = imageWidth * imageHeight
    for i in range(min(numberOfImages,10000000)):
        image = createInput(bytelist,currentByte,imageSize)
        currentByte += imageSize
        imageVal = getImageValFromNetwork(image)

        valueList[imageVal] += 1

        if(imageVal == correctValues[i]):
            correctCount += 1
        else:
            falseCount += 1
        
        print("%d: correct: %d, guessed: %d"%(i,correctValues[i],imageVal))

    print("Correct:",correctCount,"False:",falseCount,"Guessed:",valueList)
    print("Guessed correct %.2f%%d of the time"%(correctCount * 100.0 / (correctCount + falseCount)))

f = open('../image_and_label_sets/train-images.idx3-ubyte','rb')
byteFile = f.read()
f.close()
f = open('../image_and_label_sets/train-labels.idx1-ubyte')
correctFile = f.read()
f.close()

correctValues = getCorrectValues(correctFile)

allparams = np.load("../networks/256.npz")
generateNetwork(allparams)

header = readImageHeader(byteFile)
readImages(byteFile,header,correctValues)
#printMatrices()
