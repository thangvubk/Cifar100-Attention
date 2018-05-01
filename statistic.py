import json
import os
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODELS = ['resnet', 'ca_resnet', 'sa_resnet', 'ja_resnet']
val_accs = {}
for model in MODELS:
    log_path = os.path.join('checkpoint', model, 'log.json')
    val_accs[model] = json.load(open(log_path))[u'Validation acc']
    val_accs[model] = np.array(val_accs[model]) 
    #print(val_accs[model]['validation loss')

index = np.arange(200)
print(val_accs['resnet'][1, 1])
plt.xlim(0,200)
plt.ylim(0,100)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.tick_params(direction='in', top='on', right='on')
plt.plot(val_accs['resnet'][:, 1], val_accs['resnet'][:, 2], label='ResNet')
plt.plot(val_accs['ca_resnet'][:, 1], val_accs['ca_resnet'][:, 2], label='CAResNet')
plt.plot(val_accs['sa_resnet'][:, 1], val_accs['sa_resnet'][:, 2], label='SAResNet')
plt.plot(val_accs['ja_resnet'][:, 1], val_accs['ja_resnet'][:, 2], label='JAResNet')
plt.legend(loc='lower right')
plt.grid()

plt.savefig('abc.pdf', format='pdf')
