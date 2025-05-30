import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === CONFIGURAÇÕES ===
LEARNING_RATE = 0.1
PRECISION = 1e-6
MAX_EPOCHS = 10000

# === CLASSE MLP ===
class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=5, output_size=1):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

def reset_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_network(model, x_train, y_train, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.apply(reset_weights)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    errors = []

    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        errors.append(loss.item())
        loss.backward()
        optimizer.step()
        if loss.item() < PRECISION:
            break

    return model, errors, epoch + 1

def validate(model, x_test, y_test):
    with torch.no_grad():
        predictions = model(x_test)
        rel_errors = torch.abs((predictions - y_test) / y_test) * 100
        mean_error = torch.mean(rel_errors).item()
        variance = torch.var(rel_errors).item()
        return predictions, mean_error, variance

# === LEITURA DO ARQUIVO XLSX ===
df = pd.read_excel("DadosProjeto01RNA.xlsx", engine="openpyxl")
print("Colunas carregadas:", df.columns.tolist())
# Separação em treino e teste (80/20)
X = df[['x1 ', 'x2 ', 'x3 ']].values
Y = df[['d ']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

x_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(Y_train, dtype=torch.float32)
x_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(Y_test, dtype=torch.float32)

# === TREINAMENTO ===
results = []
models = []
errors_all = []

for i in range(5):
    model = MLP()
    trained_model, errors, epochs = train_network(model, x_train, y_train, seed=i)
    results.append((errors[-1], epochs))
    models.append(trained_model)
    errors_all.append(errors)

# === VALIDAÇÃO ===
for i, model in enumerate(models):
    preds, mean_error, var = validate(model, x_test, y_test)
    print(f"T{i+1}: Erro Relativo Médio = {mean_error:.2f}%, Variância = {var:.4f}")

# === PLOT DOS 2 COM MAIS ÉPOCAS ===
sorted_by_epochs = sorted(enumerate(results), key=lambda x: x[1][1], reverse=True)[:2]
plt.figure(figsize=(10, 5))
for idx, (mse, epochs) in sorted_by_epochs:
    plt.plot(errors_all[idx], label=f'T{idx+1} ({epochs} épocas)')
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Médio')
plt.title('Treinamentos com maior número de épocas')
plt.legend()
plt.grid()
plt.show()
