import matplotlib.pyplot as plt
import pandas as pd
import sys

file_location = sys.argv[1].strip()
data3 = pd.read_csv(file_location, header=None)

plt.style.use('ggplot')
plt.plot(data3[0], data3[1], 'go')
plt.xlabel('distance traveled')
plt.ylabel('carbon emission')

plt.show()