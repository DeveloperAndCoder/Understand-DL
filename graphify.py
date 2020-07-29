import matplotlib.pyplot as plt
import os

# 'Autoencoder', 'classifier', 'combined'
# LogOf = 'Autoencoder'
#LogOf = 'classifier'
LogOf = 'combined'
#LogOf = 'unet'

# LogOf = 'before_classifier'
# LogOf = 'after_classifier'

#rootDir = 'classifier'
rootDir = 'combined'

root_dir = "Log/unet_9/" + rootDir

f = open(os.path.join(root_dir, LogOf + "_log.csv"), 'r')
#print(f.read())
texts = f.read().split('\n')

offset = 1    # should be at least 1

texts = texts[offset:]
range = 1000
#print(texts)

plots = {'Accuracy': 1, 'Mean Squared Error': 1, 'Loss': 2, 'validation_accuracy': 3, 'validation_loss': 4}
plot = 'Accuracy'

x = []
y = []

for text in texts:
    values = text.split(';')
    if(len(values) >= 2):
        #print(values[0], values[1])
        x.append(float(values[0]))
        y.append(float(values[plots[plot]]))
        #print(int(values[0])+1, float(values[1]))
#plt.xticks(np.arange(0, range, range/10))
#plt.yticks(np.arange(0, range, range/50))
plt.xlabel("Epoch")
plt.ylabel(plot)
#plt.yticks(np.arange(y.min(), y.max(), 0.005))
plt.plot(x,y)
plt.savefig(root_dir + plot + str(len(x)))
plt.show()
