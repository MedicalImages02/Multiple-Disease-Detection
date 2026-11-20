import os
import cv2 as cv
import nibabel as nib
import pandas as pd
from numpy import matlib
from AFOA import AFOA
from Global_Vars import Global_Vars
from ImageResults import Sample_Images
from Model_CNN import Model_CNN
from Model_EfficientNet import Model_EfficientNet
from Model_ResNet import Model_ResNet18
from Model_VAutoEn import Model_VAutoEn
from ObjFun import objfun
from Plot_Results import *
from Proposed import Proposed
from TBA import TBA
from TDO import TDO
from WOA import WOA

No_of_Dataset = 4

# Read Dataset 1
an = 0
if an == 1:
    dir = './Dataset/Dataset1/Task01_BrainTumour/'
    direct = os.listdir(dir)
    images = []
    target = []
    for i in range(len(direct)):
        file = dir + direct[i] + '/'
        if 'imagesTr' in file:
            direct1 = os.listdir(file)
            data1 = []
            for j in range(len(direct1)):
                file1 = file + direct1[j]
                if '._' in direct1[j]:
                    data1.append(direct1[j])
                else:
                    break
            for j in range(len(data1), len(data1) + 4):
                file1 = file + direct1[j]
                read = nifti_img = nib.load(file1)
                for k in range(read.shape[2]):
                    image = nifti_img.get_fdata()[:, :, k].astype('uint8')
                    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                    images.append(image)
        elif 'labelsTr' in file:
            direct1 = os.listdir(file)
            data1 = []
            for j in range(len(direct1)):
                file1 = file + direct1[j]
                if '._' in direct1[j]:
                    data1.append(direct1[j])
                else:
                    break
            for j in range(len(data1), len(data1) + 4):
                file1 = file + direct1[j]
                read = nifti_img = nib.load(file1)
                for k in range(read.shape[2]):
                    tar = np.where(nifti_img.get_fdata()[:, :, k].astype('uint8') > 0)
                    if len(tar[1]) > 0:
                        targ = 1
                    else:
                        targ = 0
                    target.append(targ)
        else:
            continue
    Target = np.reshape(target, (-1, 1))
    index = np.arange(len(images))
    np.random.shuffle(index)
    Org_Img = np.asarray(images)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = Target[index]
    np.save('Index_1.npy', index)
    np.save('Image_1.npy', Shuffled_Datas)
    np.save('Target_1.npy', Shuffled_Target)

# Raed Dataset 2
an = 0
if an == 1:
    dir = './Dataset/Dataset2/Task06_Lung/'
    direct = os.listdir(dir)
    images = []
    target = []
    for i in range(len(direct)):
        file = dir + direct[i] + '/'
        if 'imagesTr' in file:
            direct1 = os.listdir(file)
            data1 = []
            for j in range(len(direct1)):
                file1 = file + direct1[j]
                if '._' in direct1[j]:
                    data1.append(direct1[j])
                else:
                    break
            for j in range(len(data1), len(data1) + 4):
                file1 = file + direct1[j]
                read = nifti_img = nib.load(file1)
                for k in range(read.shape[2]):
                    image = nifti_img.get_fdata()[:, :, k].astype('uint8')
                    images.append(image)
        elif 'labelsTr' in file:
            direct1 = os.listdir(file)
            data1 = []
            for j in range(len(direct1)):
                file1 = file + direct1[j]
                if '._' in direct1[j]:
                    data1.append(direct1[j])
                else:
                    break
            for j in range(len(data1), len(data1) + 4):
                file1 = file + direct1[j]
                read = nifti_img = nib.load(file1)
                for k in range(read.shape[2]):
                    tar = np.where(nifti_img.get_fdata()[:, :, k].astype('uint8') > 0)
                    if len(tar[1]) > 0:
                        targ = 1
                    else:
                        targ = 0
                    target.append(targ)
        else:
            continue
    Target = np.reshape(target, (-1, 1))
    index = np.arange(len(images))
    np.random.shuffle(index)
    Org_Img = np.asarray(images)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = Target[index]
    np.save('Index_2.npy', index)
    np.save('Image_2.npy', Shuffled_Datas)
    np.save('Target_2.npy', Shuffled_Target)

# Read Dataset 3
an = 0
if an == 1:
    Targets = []
    Images = []
    Fold_Count = [100, 100, 50]
    count = 0
    Directory = './Dataset/Dataset3/MINI-DDSM-Complete-JPEG-8/'
    fold = os.listdir(Directory)
    for i in range(len(fold)):
        file = Directory + fold[i] + '/'
        if os.path.isdir(file):
            if fold[i] == ('Benign' or 'Cancer'):
                fold1 = os.listdir(file)
                for k in range(Fold_Count[count]):
                    file1 = file + fold1[k]
                    fold2 = os.listdir(file1)
                    for l in range(len(fold2)):
                        print(file, k, l, count)
                        file3 = file1 + '/' + fold2[l]
                        if '_Mask.jpg' in file3:
                            image = cv.resize(cv.imread(file3[:-9] + '.jpg'), [512, 512])
                            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                            Target = count
                            Targets.append(Target)
                            Images.append(image.astype('uint8'))
            else:
                fold1 = os.listdir(file)
                for k in range(Fold_Count[count]):
                    file1 = file + fold1[k]
                    fold2 = os.listdir(file1)
                    for l in range(len(fold2)):
                        print(file, k, l, count)
                        file3 = file1 + '/' + fold2[l]
                        if '.jpg' in file3:
                            image = cv.resize(cv.imread(file3), [512, 512])
                            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                            Target = count
                            Targets.append(Target)
                            Images.append(image.astype('uint8'))
            count = count + 1
    uni = np.unique(Targets)
    Target = np.zeros((len(Targets), len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where(np.asarray(Targets) == uni[i])
        Target[ind[0], i] = 1

    index = np.arange(len(Images))
    np.random.shuffle(index)
    Org_Img = np.asarray(Images)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = Target[index]
    np.save('Index_3.npy', index)
    np.save('Image_3.npy', Shuffled_Datas)
    np.save('Target_3.npy', Shuffled_Target)

# Read Dataset 4
an = 0
if an == 1:
    directory = './Dataset/Dataset4/'
    dir = os.listdir(directory)
    images = []
    filename = []
    for i in range(len(dir)):
        file = directory + dir[i]
        if '.jpg' in file:
            read = cv.imread(file)
            read = cv.resize(read, [128, 128])
            images.append(read)
            filename.append(dir[i])
        else:
            pass
    filename = np.asarray(filename)
    file1 = directory + 'HAM10000_metadata.csv'
    read_csv = pd.read_csv(file1)
    read_csv = read_csv.values
    Targets = []
    for j in range(len(filename)):
        ind = np.where(filename[j][:-4] == read_csv[:, 1])
        Targets.append(read_csv[ind[0][0], 2])
    Targets = np.asarray(Targets)
    uni = np.unique(Targets)
    Target = np.zeros((len(Targets), len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where(np.asarray(Targets) == uni[i])
        Target[ind[0], i] = 1
    index = np.arange(len(images))
    np.random.shuffle(index)
    Org_Img = np.asarray(images)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = Target[index]
    np.save('Index_4.npy', index)
    np.save('Image_4.npy', Shuffled_Datas)
    np.save('Target_4.npy', Shuffled_Target)

# Optimization for Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden neuron count, Learning Rate, No of Epochs
        xmin = matlib.repmat([5, 0.01, 5], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 50], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("TDO...")
        [bestfit1, fitness1, bestsol1, time1] = TDO(initsol, fname, xmin, xmax, Max_iter)

        print("AFOA...")
        [bestfit2, fitness2, bestsol2, time2] = AFOA(initsol, fname, xmin, xmax, Max_iter)

        print("WOA...")
        [bestfit3, fitness3, bestsol3, time3] = WOA(initsol, fname, xmin, xmax, Max_iter)

        print("TBA...")
        [bestfit4, fitness4, bestsol4, time4] = TBA(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(No_of_Dataset):
        Data = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        LearnPer = [35, 45, 55, 65, 75]
        for act in range(len(LearnPer)):
            learnperc = round(Data.shape[0] * LearnPer[act])  # Split Training and Testing Datas
            Train_Data = Data[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Data[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_VAutoEn(Data, Target, sol)  # With optimization
            Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred2 = Model_ResNet18(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred3 = Model_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred4 = Model_VAutoEn(Data, Target)  # Without optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Evaluate.npy', Eval_all)  # Save Eval all

plotConvResults()
Plot_ROC_Curve()
Plots_Mod_Results()
PlotAlgResults()
Proposed_PlotResults()
Table()
Sample_Images()
