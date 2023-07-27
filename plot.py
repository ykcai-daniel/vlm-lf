import matplotlib.pyplot as plt
import json
import numpy as np

if __name__=='__main__':
    with open('hong_kong_airport_3.json','r') as f:
        results=json.load(f)
    print(results)
    results_np={ name:np.array(logits) for name,logits in results.items()}
    for name,result in results_np.items():
        plt.plot(result,label=name)
    plt.legend(loc="upper left")
    plt.savefig('results.jpg')