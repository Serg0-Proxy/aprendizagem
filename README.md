# Classificação de Imagens de Paisagens com CNN

Projeto de aprendizado de máquina que treina uma rede neural convolucional (CNN) com TensorFlow/Keras para classificar imagens em 4 categorias de paisagem.

## Classes

- Rural (Plantação, Pastagem)
- Urbano (Cidade, Rua)
- Água (Rio, Lago, Açude)
- Área Verde (Floresta, Mata)

## Estrutura do projeto

```
aprendizagem/
├── Treinamento/      # 1792 imagens de treino (uma pasta por classe)
├── Validacao/        # 420 imagens de validação (uma pasta por classe)
├── aprendizado.py    # script de treino e avaliação
└── README.md
```

## Tecnologias

- Python 3
- TensorFlow / Keras
- Pillow

## Como executar

1. Clone o repositório:

```powershell
   git clone https://github.com/Serg0-Proxy/aprendizagem.git
   cd aprendizagem
```

2. Crie e ative um ambiente virtual:

```powershell
   python -m venv aprendenv
   .\aprendenv\Scripts\Activate.ps1
```

3. Instale as dependências:

```powershell
   pip install tensorflow pillow
```

4. Execute o treinamento:

```powershell
   python aprendizado.py
```

Ao final, o modelo treinado é salvo em `modelo.keras`.

## Como funciona

- CNN com 3 blocos de convolução + max pooling, seguidos de camadas densas.
- **Data augmentation** no treino (rotação, deslocamento, zoom e espelhamento) para melhorar a generalização.
- **Dropout** para reduzir o overfitting.
- **Early stopping** que interrompe o treino quando o `val_loss` deixa de melhorar e restaura os melhores pesos.

## Resultados

| Versão                                     | Acurácia de validação | val_loss |
| ------------------------------------------ | --------------------- | -------- |
| CNN simples                                | 87,14%                | 0,767    |
| Com dropout, augmentation e early stopping | 88,81%                | 0,371    |

A versão simples apresentava overfitting (97% no treino contra 87% na validação). As técnicas de regularização eliminaram essa diferença.

## Próximos passos

- Testar transfer learning (MobileNetV2) para aumentar a acurácia.
- Criar um script para classificar imagens novas usando o `modelo.keras`.
