import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

###########Draw confusion_matrix###########
###########Using cmap to change the color#########
labels = ['correct move', 'errorneous move']
tick_marks = np.array(range(len(labels))) + 0.5
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.get_cmap('Blues')):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_test=np.loadtxt('C:\\Users\\Pursue_Lee\\PycharmProjects\\Error realted potentials\\y_test.txt')
y_true=np.loadtxt('C:\\Users\\Pursue_Lee\\PycharmProjects\\Error realted potentials\\y_true.txt')
cm = confusion_matrix(y_test,y_true)
print(cm)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  #Normalize
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

#######Set fontsize, precision and fontcolor etc.######
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=12, va='center', ha='center')

plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.show()
