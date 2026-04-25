import matplotlib.pyplot as plt

class Plot:
    
    def subplot(self, n_rows, n_cols, index):
        # Cria uma grade de subplots (n_rows x n_cols)
        # 'index' indica qual posição do gráfico será usada
        plt.subplot(n_rows, n_cols, index)

    def _apply_labels(self, x_label, y_label, title):
        plt.xlabel(x_label)   # Define o nome do eixo X
        plt.ylabel(y_label)   # Define o nome do eixo Y
        plt.title(title)      # Define o título do gráfico

    def plot(self, x, y, color_type = "k", x_label="", y_label="", title=""):
        # Cria um gráfico
        plt.plot(x, y, color_type)
        self._apply_labels(x_label, y_label, title)

    def grid(self):
        # Ativa a grade no gráfico atual
        plt.grid()    

    def show(self):
        # Exibe todos os gráficos criados
        plt.show()
