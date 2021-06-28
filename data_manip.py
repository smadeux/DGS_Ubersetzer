import pandas as pd
import matplotlib.pyplot as plt

columns_list_x = ['x4_0', 'x8_0', 'x12_0', 'x16_0', 'x20_0']
columns_list_y = ['y4_0', 'y8_0', 'y12_0', 'y16_0', 'y20_0']

df = pd.read_csv('test logs/Full Static Alphabet CSV/coords_norm_norot.csv')

df_plot_x = df[columns_list_x]
df_plot_y = df[columns_list_y]

df_plot_x.cumsum()
df_plot_y.cumsum()

for l in columns_list_x:
    print('X Mean Distance for {}: {}'.format(l, df_plot_x[l].mean()))
    print('X Standard Deviation for {}: {}'.format(l, df_plot_x[l].std()))

# df_plot_x.plot(subplots=True)
# plt.show()