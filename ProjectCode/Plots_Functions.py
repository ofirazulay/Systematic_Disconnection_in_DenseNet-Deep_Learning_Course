import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import statistics

def graph_with_subplots(ArchitecturesName,Values_Array,Ylable,Titel):
    num_architectures = len(Values_Array)
    rows = int(num_architectures / 2) + (num_architectures % 2 > 0)
    cols = min(num_architectures, 2)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        if i < num_architectures:
            P_values = range(1, len(Values_Array[i]) + 1)  # Shifted P values
            ax.plot(P_values, Values_Array[i], marker='o')
            ax.set_xlabel('P', fontsize=12)
            ax.set_ylabel(Ylable)
            ax.set_title(ArchitecturesName[i], fontweight='bold')
            ax.grid(True)

        else:
            ax.axis('off')  # Hide empty subplots

    plt.suptitle(Titel, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()


def graph_union(ArchitecturesName,Values_Array, Ylable, Titel):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, parameters in enumerate(Values_Array):
        x = np.arange(1, len(parameters) + 1)
        ax.plot(x, parameters, marker='o', label=ArchitecturesName[i])

    ax.set_xlabel('P',fontweight='bold', fontsize=12)
    ax.set_ylabel(Ylable,fontweight='bold', fontsize=12)
    ax.set_title(Titel, fontweight='bold',fontsize=14)
    ax.legend()
    ax.grid(True)

    # Format y-axis tick labels
    plt.ticklabel_format(axis='y', style='plain')
    plt.show()

def graph_union_zoom(ArchitecturesName, Values_Array, Ylable, Titel):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, parameters in enumerate(Values_Array):
        x = np.arange(1, len(parameters) + 1)
        ax.plot(x, parameters, marker='o', label=ArchitecturesName[i])

    ax.set_xlabel('P')
    ax.set_ylabel(Ylable)
    ax.set_title(Titel, fontweight='bold',fontsize=18)
    ax.legend()
    ax.grid(True)

    # Format y-axis tick labels
    plt.ticklabel_format(axis='y', style='plain')
    plt.ylim(0.96, 1)
    plt.show()




def graph_for_Accuracy(ArchitecturesName,accuracy_Dict,Titel):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    for i, (architecture, name) in enumerate(zip(accuracy_Dict, ArchitecturesName)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        ax.set_title(name, fontweight='bold')

        for p, accuracies in architecture.items():
            epochs = range(1, len(accuracies) + 1)
            ax.plot(epochs, accuracies, label=f"P={p}")

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.legend(loc='lower right', fontsize='small')

    plt.suptitle(Titel, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()




ArchitecturesName = ["L=6,K=6", "L=6 K=12", "L=9 K=6","L=9 K=12", "L=12 K=6", "L=12 K=12"]

#To generate charts, store in the following arrays the outputs of each run
# (each run tests a different architecture among 6 Architectures)


#for each tested architecture:
#in "runTime=[]" array store "runtimes" array output
#in "Num_of_Parameters=[]" array store "total_params_array" array output
#in "accuracy_Dict=[]" array store "accuracy_values" dictionery output
#then run the plots




#####-------------------------------results----------------------
#L=6 K=6
runTime_6_6=[]
Num_of_Parameters_6_6=[]
accuracy_Dict_6_6={}

#L=6 K=12
runtime_6_12 = []
Num_of_Parameters_6_12=[]
accuracy_Dict_6_12 ={}

#L=9 K=6
runtime_9_6=[]
Num_of_Parameters_9_6=[]
accuracy_Dict_9_6 ={}

#L=9 K=12
runtime_9_12=[]
Num_of_Parameters_9_12=[]
accuracy_Dict_9_12 ={}

#L=12 K=6
runtime_12_6=[]
Num_of_Parameters_12_6=[]
accuracy_Dict_12_6={}

#L=12 K=12
runtime_12_12=[]
Num_of_Parameters_12_12=[]
accuracy_Dict_12_12={}



#plots
runTime_Different_Architectures=[runTime_6_6,runtime_6_12,runtime_9_6,runtime_9_12,runtime_12_6,runtime_12_12]
Num_of_Parameters_Different_Architectures=[Num_of_Parameters_6_6,Num_of_Parameters_6_12,Num_of_Parameters_9_6,Num_of_Parameters_9_12,Num_of_Parameters_12_6,Num_of_Parameters_12_12]
accuracy_Dict_Different_Architectures=[accuracy_Dict_6_6,accuracy_Dict_6_12,accuracy_Dict_9_6,accuracy_Dict_9_12,accuracy_Dict_12_6,accuracy_Dict_12_12]



graph_with_subplots(ArchitecturesName,runTime_Different_Architectures,'Run Time (sec)',"Run Time by P Value for 6 Different Architectures")
graph_with_subplots(ArchitecturesName,Num_of_Parameters_Different_Architectures,'Number of_Parameters',"Number of_Parameters by P Value for 6 Different Architectures")

graph_union(ArchitecturesName,runTime_Different_Architectures,'Run Time (sec)',"Run Time by P Value for 6 Different Architectures")
graph_union(ArchitecturesName,Num_of_Parameters_Different_Architectures,'Number of_Parameters',"Number of_Parameters by P Value for 6 Different Architectures")

graph_for_Accuracy(ArchitecturesName,accuracy_Dict_Different_Architectures,'validation Accuracy per Epoch by each model (different P) for 6 Different Architectures ')



max_values_array = []
for architecture in accuracy_Dict_Different_Architectures:
    max_values = []
    for p, accuracies in architecture.items():
        max_value = max(accuracies)
        max_values.append(max_value)
    max_values_array.append(max_values)



std_values_array = []
for architecture in accuracy_Dict_Different_Architectures:
    std_values = []
    for p, accuracies in architecture.items():
        std_value = statistics.stdev(accuracies)
        std_values.append(std_value)
    std_values_array.append(std_values)

graph_union(ArchitecturesName,max_values_array,'Best Validation Accuracy',"Best Validation Accuracy by P Value for 6 Different Architectures")
graph_union(ArchitecturesName,std_values_array,'Standard deviation',"Standard deviation of Validation Accuracy by P Value for 6 Different Architectures")
graph_union_zoom(ArchitecturesName,max_values_array,'Best Validation Accuracy',"Best Validation Accuracy by P Value for 6 Different Architectures")




