import matplotlib.pyplot as plt
import os

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
root_dir = "Log/unet_6/combined/"

f = open(os.path.join(root_dir, "combined_log.csv"), 'r')
=======
root_dir = "Log/stl10_2/classifier"

f = open(os.path.join(root_dir, "classifier_log.csv"), 'r')
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
root_dir = "Log/stl10_2/classifier"

f = open(os.path.join(root_dir, "classifier_log.csv"), 'r')
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
root_dir = "Log/stl10_2/classifier"

f = open(os.path.join(root_dir, "classifier_log.csv"), 'r')
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
root_dir = "Log/stl10_2/classifier"

f = open(os.path.join(root_dir, "classifier_log.csv"), 'r')
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
#print(f.read())
texts = f.read().split('\n')
texts = texts[1:]
range = 1000
#print(texts)

plots = {'Accuracy': 1, 'Loss': 2}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
plot = 'Loss'
=======
plot = 'Accuracy'
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
plot = 'Accuracy'
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
plot = 'Accuracy'
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
plot = 'Accuracy'
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377

x = []
y = []

for text in texts:
    values = text.split(';')
    if(len(values) >= 2):
        #print(values[0], values[1])
        x.append(float(values[0]))
        y.append(float(values[plots[plot]]))
        #x.append(int(values[0])+1)
        #y.append(float(values[1]))
        #print(int(values[0])+1, float(values[1]))
#plt.xticks(np.arange(0, range, range/10))
#plt.yticks(np.arange(0, range, range/50))
plt.xlabel("Epoch")
plt.ylabel(plot)
#plt.yticks(np.arange(y.min(), y.max(), 0.005))
plt.plot(x,y)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
plt.savefig('combined_'+plot)
#plt.savefig(root_dir + plot + '_Loss' + str(len(x)))
#plt.show()
=======
plt.savefig(root_dir + plot + '_Loss' + str(len(x)))
plt.show()
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
plt.savefig(root_dir + plot + '_Loss' + str(len(x)))
plt.show()
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
plt.savefig(root_dir + plot + '_Loss' + str(len(x)))
plt.show()
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
=======
plt.savefig(root_dir + plot + '_Loss' + str(len(x)))
plt.show()
>>>>>>> 0cde423e319a5c313280a0b772cbab6ad1f81377
