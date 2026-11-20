import numpy as np
import warnings
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from matplotlib.patches import Polygon

warnings.filterwarnings("ignore")

No_of_Dataset = 4


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'TDO-ACA-PEVGAN-ELF', 'AFOA-ACA-PEVGAN-ELF', 'WOA-ACA-PEVGAN-ELF', 'TBA-ACA-PEVGAN-ELF', 'RUBTO-ACA-PEVGAN-ELF']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 3
    cls = ['CNN', 'ResNet', 'EfficientNet', 'CA-PEVGAN-ELF', 'ACA-PEVGAN-ELF']
    for a in range(No_of_Dataset):
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        Actual = Actual[lenper:, :]
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(a + 1) + ' ROC Curve')
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel("True Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
        plt.title("ROC Curve")
        plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 12})
        path = "./Results/ROC_%s.png" % (a + 1)
        plt.savefig(path)
        plt.show()


def Table():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Algorithm = ['Activation Function', 'TDO-ACA-PEVGAN-ELF', 'AFOA-ACA-PEVGAN-ELF', 'WOA-ACA-PEVGAN-ELF', 'TBA-ACA-PEVGAN-ELF', 'RUBTO-ACA-PEVGAN-ELF']
    Classifier = ['Activation Function', 'CNN', 'ResNet', 'EfficientNet', 'CA-PEVGAN-ELF', 'ACA-PEVGAN-ELF']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    Table_Terms = [0, 2, 4, 6, 9, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Activation_Fun = ['Linear', 'Relu', 'Tanh', 'Softmax', 'Sigmoid']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Activation_Fun)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('----------------------------------Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Activation_Fun)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('---------------------------------Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plots_Mod_Results():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 3, 4, 8, 9, 12, 13, 14]
    colors = ['#8f2d56', '#b388eb', '#219ebc', '#f77f00', '#8ac926']
    Classifier = ['CNN', 'ResNet', 'EfficientNet', 'CA-PEVGAN-ELF', 'ACA-PEVGAN-ELF']
    LearningPercentage = [35, 45, 65, 75]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title(
                'Dataset - ' + str(i + 1) +  Terms[j] + ' Method Comparison of LearningPercentage')
            X = np.arange(len(LearningPercentage))
            plt.bar(X + 0.00, Graph[:, 5], color='#8f2d56', width=0.15, hatch='*', label=Classifier[0])
            plt.bar(X + 0.15, Graph[:, 6], color='#b388eb', width=0.15, hatch='X*', label=Classifier[1])
            plt.bar(X + 0.30, Graph[:, 7], color='#219ebc', width=0.15, hatch='O.', label=Classifier[2])
            plt.bar(X + 0.45, Graph[:, 8], color='#f77f00', width=0.15, hatch='/o', label=Classifier[3])
            plt.bar(X + 0.60, Graph[:, 4], color='#8ac926', width=0.15, hatch='*-', label=Classifier[4])
            plt.xticks(X + 0.30, ['35', '45', '65', '75'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.xlabel('Learning Percentage', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fancybox=False, shadow=False,
                       prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_mod_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def PlotAlgResults():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 3, 4, 8, 9, 12, 13, 14]
    Algorithm = ['TDO-ACA-PEVGAN-ELF', 'AFOA-ACA-PEVGAN-ELF', 'WOA-ACA-PEVGAN-ELF', 'TBA-ACA-PEVGAN-ELF', 'RUBTO-ACA-PEVGAN-ELF']
    BatchSize = ['35', '45', '65', '75']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title(
                'Dataset - ' + str(i + 1) +  Terms[j] + ' Algorithm Comparison of LearningPercentage')
            X = np.arange(len(BatchSize))
            Color = ['#E08A8A', '#E3CD7F', '#8DDC8D', '#C490DB', '#B5B5B5']
            plt.bar(X + 0.00, Graph[:, 0], color='None', edgecolor=Color[0], width=0.15, linewidth=4,
                    label=Algorithm[0])
            plt.bar(X + 0.15, Graph[:, 1], color='None', edgecolor=Color[1], width=0.15, linewidth=4,
                    label=Algorithm[1])
            plt.bar(X + 0.30, Graph[:, 2], color='None', edgecolor=Color[2], width=0.15, linewidth=4,
                    label=Algorithm[2])
            plt.bar(X + 0.45, Graph[:, 3], color='None', edgecolor=Color[3], width=0.15, linewidth=4,
                    label=Algorithm[3])
            plt.bar(X + 0.60, Graph[:, 4], color='None', edgecolor=Color[4], width=0.15, linewidth=4,
                    label=Algorithm[4])
            plt.xticks(X + 0.30, ['35', '45', '65', '75'], fontsize=14, fontname="Arial",
                       fontweight='bold', color='k')
            plt.xlabel('Learning Percentage', fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in Color]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.13), fontsize=12,
                       frameon=False, ncol=2, prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Alg_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Proposed_PlotResults():
    eval = np.load('Evaluates.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC',
             'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Term = [0, 1, 2, 3, 12]
    Epochs = ['100', '200', '300', '400', '500']
    colors = ['#2ca6a4', '#e46d75', '#9ac23c', '#f28c28', '#d43832']
    bar_width = 0.6
    side_depth = 0.25
    top_depth = 0.20

    for i in range(eval.shape[0]):
        for j, metric_idx in enumerate(Graph_Term):
            fig = plt.figure(figsize=(10, 5))
            fig.canvas.manager.set_window_title(
                'Dataset - ' + str(i + 1) +  Terms[j] + ' Step Per Epochs vs Proposed')

            # Plot bars
            for lp_idx, lp in enumerate(Epochs):
                x = lp_idx
                v = eval[i, lp_idx, 4, metric_idx + 4]
                color = colors[lp_idx % len(colors)]

                # FRONT FACE
                front = Polygon([[x, 0], [x + bar_width, 0], [x + bar_width, v], [x, v]], closed=True, facecolor=color,
                                edgecolor='none')
                plt.gca().add_patch(front)

                # RIGHT SIDE FACE
                right = Polygon([[x + bar_width, 0], [x + bar_width + side_depth, side_depth + 2.5],
                                 [x + bar_width + side_depth, v + side_depth + 2.5], [x + bar_width, v]],
                                closed=True, facecolor=color, alpha=0.9, edgecolor='none')
                plt.gca().add_patch(right)

                # TOP FACE
                top = Polygon([[x, v], [x + bar_width, v], [x + bar_width + side_depth, v + top_depth + 2.5],
                               [x + side_depth, v + 2.5]], closed=True, facecolor=color, alpha=0.9, edgecolor='none')
                plt.gca().add_patch(top)
            plt.xlabel('Step Per Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[metric_idx], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            x_positions = [x + bar_width / 2 for x in range(len(Epochs))]
            plt.xticks(x_positions, Epochs, fontsize=12, fontweight='bold', color='k')
            max_value = max([eval[i, lp_idx, 4, metric_idx + 4] for lp_idx in range(len(Epochs))])
            y_step = max(max_value // 5, 1)  # Ensure step is at least 1
            plt.yticks(range(0, int(max_value + y_step + 1), int(y_step)), fontsize=12, fontweight='bold', color='k')

            for s in ['top', 'right']:
                plt.gca().spines[s].set_visible(False)
            plt.xlim(-0.5, len(Epochs))
            plt.ylim(0, max_value + 10)
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Proposed_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path, dpi=300,
                        bbox_inches='tight')
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plot_ROC_Curve()
    Plots_Mod_Results()
    PlotAlgResults()
    Proposed_PlotResults()
    Table()
