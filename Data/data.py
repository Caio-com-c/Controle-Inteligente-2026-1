import pandas as pd

class Data:
    #Caso já exista um banco de dados pré-existente 
    Caminho = r"C:\Users\joyce\OneDrive\Documentos\SCILAB\saida.csv"
    def __init__(self, arquivo_csv= Caminho):
        # Lê o arquivo CSV
        try:
            self.dados = pd.read_csv(arquivo_csv, sep=None, engine='python', header=None)
            # Definindo e atribuindo nomes das colunas
            self.nomes_colunas = ['t', 'Qin', 'H']
            self.dados.columns = self.nomes_colunas

            # Atributos que o gráfico vai acessar
            self.tempo = self.dados['t']
            self.entrada = self.dados['Qin']
            self.saida = self.dados['H']

        except Exception as e:
            print(f"Erro ao ler o CSV: {e}")
            exit()

    def t(self):
        return self.tempo
    def Qin(self):
        return self.entrada
    def H(self):
        return self.saida

# MostrarDados = Data()
# print(MostrarDados.t())
# print(MostrarDados.Qin())
# print(MostrarDados.H())
