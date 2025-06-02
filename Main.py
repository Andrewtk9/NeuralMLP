import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


# === CONFIGS ===
LEARNING_RATE = 0.1
PRECISION = 1e-6
epocas_personalizadas = [10000, 100000, 1000000, 1000000,1000000]

class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, output_size=1):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # ativação apenas para camada oculta

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))  # oculta com sigmoid
        x = self.output(x)                # saída linear (sem ativação)
        return x




def reset_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_network(model, x_train, y_train, seed, max_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.apply(reset_weights)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    errors = []

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        errors.append(loss.item())
        loss.backward()
        optimizer.step()

    return model, errors, max_epochs

def validate(model, x_test, y_test):
    with torch.no_grad():
        predictions = model(x_test)
        rel_errors = torch.abs((predictions - y_test) / y_test) * 100
        mean_error = torch.mean(rel_errors).item()
        variance = torch.var(rel_errors).item()
        return mean_error, variance

# === LEITURA DO ARQUIVO ===
df = pd.read_excel("DadosProjeto01RNA.xlsx", engine="openpyxl")
df.columns = df.columns.str.strip().str.lower().str.replace('//', '').str.replace(' ', '')
X = df[['x1', 'x2', 'x3']].values
Y = df[['d']].values

# === SEPARAÇÃO EM TREINO/TESTE ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
x_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(Y_train, dtype=torch.float32)
x_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(Y_test, dtype=torch.float32)

# === TREINAMENTO E COLETA ===
results = []
errors_all = []
epochs_all = []
models = []

for i in range(5):
    epocas_desejadas = epocas_personalizadas[i]
    model = MLP()

    start_time = time.time()
    trained_model, errors, epochs = train_network(model, x_train, y_train, seed=i, max_epochs=epocas_desejadas)
    elapsed_time = time.time() - start_time  # tempo em segundos

    mean_error, variance = validate(trained_model, x_test, y_test)

    print(f"Treinamento T{i+1} concluído em {elapsed_time:.2f} segundos")

    results.append({
        "Treinamento": f"T{i+1}",
        "EQM_Final": errors[-1],
        "Épocas_Executadas": epochs,
        "Erro_Relativo_Médio_%": mean_error,
        "Variância_%": variance,
        "Tempo_Treinamento_s": round(elapsed_time, 2)
    })
    
    errors_all.append(errors)
    epochs_all.append(epochs)
    models.append(trained_model)


# === APLICAÇÃO DOS 5 MODELOS TREINADOS NA TABELA 2 ===
df_tabela2 = pd.read_csv("tabela2_teste.csv")
x_manual = torch.tensor(df_tabela2[['x1', 'x2', 'x3']].values, dtype=torch.float32)
y_manual = df_tabela2['d'].values

resultados_manual = df_tabela2.copy()
erro_relativo_por_modelo = []

for i, modelo in enumerate(models):  # models = lista de modelos treinados no loop anterior
    with torch.no_grad():
        y_pred = modelo(x_manual).squeeze().numpy()
        resultados_manual[f'y(T{i+1})'] = y_pred.round(4)
        erro_rel = abs((y_pred - y_manual) / y_manual) * 100
        erro_relativo_por_modelo.append(erro_rel)

# === Cálculo erro médio e variância ===
linha_erro_medio = {'Amostra': 'Erro relativo médio (%)'}
linha_variancia = {'Amostra': 'Variância (%)'}

for i, erros in enumerate(erro_relativo_por_modelo):
    linha_erro_medio[f'y(T{i+1})'] = round(np.mean(erros), 2)
    linha_variancia[f'y(T{i+1})'] = round(np.var(erros), 2)

# === Adiciona ao final da tabela
resultados_manual = pd.concat([resultados_manual, pd.DataFrame([linha_erro_medio, linha_variancia])], ignore_index=True)

# === Salva arquivo
resultados_manual.to_csv("validacao_tabela2_resultado.csv", index=False)
print("Arquivo 'validacao_tabela2_resultado.csv' gerado com sucesso.")

# === SALVAR CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv("resultados_treinamento.csv", index=False)

# === PLOTAR OS 2 PIORES CASOS (MAIS ÉPOCAS) ===
maiores = sorted(enumerate(epochs_all), key=lambda x: x[1], reverse=True)[:2]
plt.figure(figsize=(10, 5))
for idx, _ in maiores:
    plt.plot(errors_all[idx], label=f"T{idx+1} ({epochs_all[idx]} épocas)")
plt.title('Erro Quadrático Médio por Época')
plt.xlabel('Épocas')
plt.ylabel('EQM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_eqm_epocas.png")
plt.show()
