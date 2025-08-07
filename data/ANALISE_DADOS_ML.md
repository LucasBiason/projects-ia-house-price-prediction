# 📊 Análise de Dados e Machine Learning: Predição de Preços de Imóveis

## 🎯 Objetivo

Este documento demonstra o processo completo de análise de dados e construção de um modelo de machine learning para predizer preços de imóveis. Vamos aprender os conceitos fundamentais de limpeza de dados, pré-processamento e regressão linear.

## 📚 Conceitos que vamos aprender

- **Limpeza de Dados**: Preparação e organização dos dados
- **Pré-processamento**: Transformação de dados para machine learning
- **Feature Engineering**: Criação de características para o modelo
- **Regressão Linear**: Modelo de predição de valores contínuos
- **Validação de Modelo**: Verificação da qualidade das predições

---

## 🔧 Importação das Bibliotecas

Primeiro, vamos importar as bibliotecas necessárias. Cada uma tem um papel específico:

- **pandas**: Manipulação e análise de dados
- **numpy**: Computação numérica
- **matplotlib/seaborn**: Visualização de dados
- **sklearn**: Algoritmos de machine learning
- **scipy**: Estatísticas e testes
- **statsmodels**: Análise estatística avançada

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
```

---

## 📖 Carregamento e Exploração dos Dados

### O que é Limpeza de Dados?

Limpeza de dados é o processo de identificar e corrigir problemas nos dados antes de usar para análise ou machine learning. Inclui:

- **Dados ausentes**: Valores que não existem
- **Dados duplicados**: Registros repetidos
- **Dados inconsistentes**: Valores que não fazem sentido
- **Formato inadequado**: Dados no formato errado

### Por que é importante?

Dados sujos = Modelos ruins! A qualidade do modelo depende diretamente da qualidade dos dados.

```python
# Carregando os dados do arquivo CSV
base = pd.read_csv('house_prices.csv', sep='|')

# Visualizando as primeiras linhas
print("Primeiras 5 linhas dos dados:")
print(base.head())

# Informações sobre o dataset
print(f"\nDimensões: {base.shape}")
print(f"Colunas: {list(base.columns)}")
print(f"\nTipos de dados:")
print(base.dtypes)
```

---

## 🔄 Pré-processamento de Dados

### O que é Pré-processamento?

Pré-processamento é a transformação dos dados brutos em um formato que os algoritmos de machine learning possam entender e usar eficientemente.

### Tipos de Dados e suas Transformações:

#### 1. **Dados Numéricos** (tamanho, quantidade_quartos, price)
- **Problema**: Diferentes escalas (tamanho em m² vs preço em R$)
- **Solução**: **StandardScaler** - Normaliza os dados para média 0 e desvio padrão 1
- **Fórmula**: z = (x - μ) / σ

#### 2. **Dados Categóricos** (localizacao)
- **Problema**: Algoritmos não entendem texto
- **Solução**: **OneHotEncoder** - Converte categorias em colunas binárias
- **Exemplo**: 'Centro' → [1, 0, 0], 'Suburbio' → [0, 1, 0], 'Vila' → [0, 0, 1]

### Por que fazer isso?
- **Evita viés**: Sem normalização, variáveis com valores maiores dominam o modelo
- **Melhora performance**: Algoritmos convergem mais rápido
- **Facilita interpretação**: Coeficientes ficam comparáveis

```python
# Criando o pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['tamanho', 'quantidade_quartos', 'price']), # Dados numéricos
        ('cat', OneHotEncoder(), ['localizacao']) # Dados categóricos
    ])

# Aplicando as transformações aos dados
base_transformed = preprocessor.fit_transform(base)

# Convertendo de volta para DataFrame para visualização
base_transformed_df = pd.DataFrame(
    base_transformed, 
    columns=['price_scaled', 'tamanho_scaled', 'quantidade_quartos_scaled', 
             'localizacao_Centro', 'localizacao_Suburbio', 'localizacao_Vila']
)

print("Dados transformados com sucesso!")
print("\nPrimeiras 5 linhas dos dados transformados:")
print(base_transformed_df.head())
```

---

## 🔍 Análise de Correlação

### O que é Correlação?

Correlação mede a força e direção da relação linear entre duas variáveis.

- **Correlação = +1**: Relação linear positiva perfeita
- **Correlação = -1**: Relação linear negativa perfeita
- **Correlação = 0**: Nenhuma relação linear

### Por que analisar correlação?
- **Identificar padrões**: Quais variáveis estão relacionadas
- **Detectar multicolinearidade**: Variáveis muito correlacionadas podem causar problemas
- **Feature selection**: Variáveis com baixa correlação com o target podem ser removidas

```python
# Calculando a matriz de correlação
corr = base_transformed_df.corr()

# Criando um heatmap para visualizar as correlações
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', center=0)
plt.title('Matriz de Correlação - Dados Transformados')
plt.show()

print("\nInterpretação das correlações:")
print("• Valores próximos de +1: Correlação positiva forte")
print("• Valores próximos de -1: Correlação negativa forte")
print("• Valores próximos de 0: Baixa correlação")
```

---

## 🤖 Construção do Modelo de Machine Learning

### O que é Regressão Linear?

Regressão linear é um algoritmo que encontra a melhor linha reta para prever uma variável contínua (preço) baseada em outras variáveis (características).

### Fórmula Matemática:
**y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε**

Onde:
- **y**: Variável que queremos prever (preço)
- **β₀**: Intercepto (valor base)
- **βᵢ**: Coeficientes (peso de cada característica)
- **xᵢ**: Características (tamanho, quartos, localização)
- **ε**: Erro (diferença entre predição e valor real)

### Por que Regressão Linear?
- **Simples e interpretável**: Fácil de entender
- **Rápido**: Computacionalmente eficiente
- **Baseline**: Bom ponto de partida para comparar outros modelos
- **Estável**: Menos propenso a overfitting

```python
# Preparando os dados para o modelo
X = base_transformed_df.drop('price_scaled', axis=1)  # Features
y = base_transformed_df['price_scaled']               # Target

print(f"Dimensões dos dados:")
print(f"X (features): {X.shape}")
print(f"y (target): {y.shape}")
print(f"\nFeatures utilizadas: {list(X.columns)}")

# Criando o pipeline do modelo
pipeline = Pipeline(steps=[('model', LinearRegression())])

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDivisão dos dados:")
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# Treinando o modelo
pipeline.fit(X_train, y_train)
print("\nModelo treinado com sucesso!")
```

---

## 📊 Análise de Resíduos

### O que são Resíduos?

Resíduos são a diferença entre os valores reais e os valores preditos pelo modelo:
**Resíduo = Valor Real - Valor Predito**

### Por que analisar resíduos?
- **Verificar qualidade do modelo**: Resíduos pequenos = bom modelo
- **Detectar padrões**: Resíduos com padrão indicam problemas
- **Validar suposições**: Regressão linear assume resíduos normalmente distribuídos

### Suposições da Regressão Linear:
1. **Linearidade**: Relação linear entre features e target
2. **Independência**: Resíduos são independentes
3. **Homocedasticidade**: Variância constante dos resíduos
4. **Normalidade**: Resíduos seguem distribuição normal

```python
# Fazendo predições no conjunto de treino
y_train_pred = pipeline.predict(X_train)

# Calculando os resíduos
residuals = y_train - y_train_pred

print(f"Estatísticas dos resíduos:")
print(f"Média: {residuals.mean():.4f}")
print(f"Desvio Padrão: {residuals.std():.4f}")
print(f"Mínimo: {residuals.min():.4f}")
print(f"Máximo: {residuals.max():.4f}")

# Criando QQ Plot para verificar normalidade
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot dos Resíduos - Verificação de Normalidade')
plt.xlabel('Quantis Teóricos')
plt.ylabel('Quantis dos Resíduos')
plt.grid(True, alpha=0.3)
plt.show()

print("\nInterpretação do QQ Plot:")
print("• Pontos na linha reta: Resíduos normais ✅")
print("• Pontos fora da linha: Resíduos não normais ❌")
```

```python
# Teste de Shapiro-Wilk para normalidade
stat, p_value = stats.shapiro(residuals)

print(f"Teste de Shapiro-Wilk para Normalidade:")
print(f"Estatística de Teste: {stat:.3f}")
print(f"Valor-p: {p_value:.3f}")

print("\nInterpretação do Teste:")
print("• Estatística próxima de 1: Dados normais ✅")
print("• Valor-p > 0.05: Não rejeitamos a hipótese de normalidade ✅")
print("• Valor-p ≤ 0.05: Rejeitamos a hipótese de normalidade ❌")

if p_value > 0.05:
    print(f"\n✅ Resultado: Resíduos são normalmente distribuídos (p = {p_value:.3f})")
else:
    print(f"\n❌ Resultado: Resíduos NÃO são normalmente distribuídos (p = {p_value:.3f})")

print("\nImplicações:")
print("• Resíduos normais: Modelo está bem especificado")
print("• Resíduos não normais: Pode indicar necessidade de transformações ou modelo diferente")
```

---

## 🚀 Modelo Final e Predições

### Pipeline Completo

Agora vamos criar o modelo final que combina pré-processamento + regressão linear em um único pipeline. Isso garante que:

- **Consistência**: Mesmas transformações aplicadas em treino e teste
- **Simplicidade**: Um único objeto para fazer predições
- **Robustez**: Menos chance de erros

### Como Funciona:
1. **Entrada**: Dados brutos (tamanho, localização, quartos)
2. **Pré-processamento**: Normalização + One-hot encoding
3. **Modelo**: Regressão linear
4. **Saída**: Preço predito

```python
# Criando o pipeline final completo
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['tamanho', 'quantidade_quartos']), # Features numéricas
        ('cat', OneHotEncoder(), ['localizacao']) # Features categóricas
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Primeiro: pré-processamento
    ('model', LinearRegression())    # Segundo: modelo
])

print("Pipeline final criado!")
print("\nEtapas do pipeline:")
print("1️⃣  Pré-processamento: StandardScaler + OneHotEncoder")
print("2️⃣  Modelo: LinearRegression")

# Carregando dados e preparando para treino
data = pd.read_csv('house_prices.csv', sep='|')
X = data.drop('price', axis=1)  # Features
y = data['price']               # Target

# Dividindo dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o pipeline completo
pipeline.fit(X_train, y_train)

print("\n✅ Pipeline treinado com sucesso!")
print(f"📊 Dados de treino: {X_train.shape[0]} amostras")
print(f"📊 Dados de teste: {X_test.shape[0]} amostras")
```

```python
# Fazendo uma predição de exemplo
features = [400, 'Centro', 2]
features_df = pd.DataFrame([features], columns=['tamanho', 'localizacao', 'quantidade_quartos'])

# Fazendo a predição
price = pipeline.predict(features_df)[0]

print("🏠 Predição de Preço de Imóvel")
print("=" * 40)
print(f"📍 Características:")
print(f"   • Tamanho: {features[0]} m²")
print(f"   • Localização: {features[1]}")
print(f"   • Quartos: {features[2]}")
print(f"\n💰 Preço Predito: R$ {price:,.2f}")

print("\n🎯 Como interpretar:")
print("• O modelo analisou padrões nos dados de treino")
print("• Identificou relações entre características e preços")
print("• Aplicou essas relações para prever o preço do novo imóvel")
print("• Quanto mais dados de treino, melhor a predição")
```

---

## 🎓 Resumo do que Aprendemos

### 📊 **Limpeza e Pré-processamento de Dados**
- **Importância**: Dados limpos = modelos melhores
- **Técnicas**: Normalização, One-hot encoding
- **Ferramentas**: StandardScaler, OneHotEncoder

### 🤖 **Machine Learning**
- **Algoritmo**: Regressão Linear
- **Objetivo**: Predizer valores contínuos
- **Aplicação**: Preços de imóveis

### 🔍 **Validação de Modelo**
- **Análise de resíduos**: Verificar qualidade
- **Testes estatísticos**: Shapiro-Wilk
- **Visualizações**: QQ Plot

### 🚀 **Próximos Passos**
- Experimentar outros algoritmos (Random Forest, XGBoost)
- Feature engineering mais avançado
- Validação cruzada
- Otimização de hiperparâmetros

### 💡 **Conceitos-Chave**
1. **Dados são o combustível**: Qualidade dos dados determina qualidade do modelo
2. **Pré-processamento é crucial**: Transformações adequadas melhoram performance
3. **Validação é essencial**: Sempre verificar se o modelo faz sentido
4. **Interpretabilidade importa**: Modelos simples são mais confiáveis

---

**🎯 Lembre-se**: Machine Learning é uma jornada iterativa. Comece simples, valide sempre, melhore gradualmente!

---

## 📝 Código Completo

```python
# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import scipy.stats as stats

# Carregamento dos dados
data = pd.read_csv('house_prices.csv', sep='|')

# Preparação dos dados
X = data.drop('price', axis=1)
y = data['price']

# Criação do pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['tamanho', 'quantidade_quartos']),
        ('cat', OneHotEncoder(), ['localizacao'])
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
pipeline.fit(X_train, y_train)

# Predição
features = [400, 'Centro', 2]
features_df = pd.DataFrame([features], columns=['tamanho', 'localizacao', 'quantidade_quartos'])
price = pipeline.predict(features_df)[0]

print(f"Preço predito: R$ {price:,.2f}")
``` 