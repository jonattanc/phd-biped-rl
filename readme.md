# Biped RL in PyBullet
🦵 Biped RL in PyBullet — Generalização Cruzada
Este projeto implementa um framework para treinar e avaliar agentes de Aprendizado por Reforço (RL) no controle de um robô bípede em simulação física (PyBullet). O foco é estudar a generalização cruzada de políticas treinadas em diferentes ambientes perturbados.

# Objetivo Geral
Treinar 6 agentes especialistas (AE), cada um em um circuito distinto (PR, P<μ, RamA, RamD, PG, PRB), e avaliar sua capacidade de generalização quando testados nos outros 5 circuitos. A métrica principal é o Tempo Médio (Tm) para completar os 10 metros.

# Circuitos de Perturbação
PR: Piso Regular (baseline)
PBA: Piso Plano Baixo Atrito (μ=0.1)
PRA: Rampa Ascendente (8.33°)
PRD: Rampa Descendente (-8.33°)
PG: Piso Granulado (areia, 50mm)
PRB: Piso Regular com Bloqueio Articular (4 ativações aleatórias de 1.5s)

# Installation

## Pré-requisitos
Python 3.12
Git

## Configuração do Ambiente

### Clone o repositório
```
git clone https://github.com/jonattanc/phd-biped-rl.git
cd phd-biped-rl
```

### Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows
```

### Install dependencies
```
pip install -r requirements.txt
```

### Setup the development environment
It's recommended to use Visual Studio Code with the Python extension. This should be enough to activate the Black lint extension when saving files.

### Run
```
python src/main.py
```

# Fluxos de Utilização das Abas

## Aba TREINAMENTO - Desenvolvimento de Agentes Especialistas

### Objetivo
Treinar agentes especialistas (AE) em circuitos específicos para posterior avaliação de generalização.

### Fluxo Principal
1. Configuração Inicial:
Algoritmo: [TD3 ▼]   Ambiente: [PR ▼]   Robô: [robot_stage1 ▼]

2. Controles de Treinamento:
[Iniciar Treino] [Pausar] [Finalizar] [Salvar Treino] [Carregar Treino]

3. Monitoramento em Tempo Real:
Gráficos de recompensa, tempo e distância por episódio
Logs detalhados do treinamento
Métricas de performance em tempo real

4. Funcionalidades Avançadas:
Visualização do robô durante treinamento
Modo tempo real da simulação
Sistema automático de salvamento do melhor modelo
Retomada de treinamentos interrompidos

### Casos de Uso Típicos
Caso 1: Novo Treinamento:
1. Selecionar algoritmo (TD3/PPO), ambiente (PR/PBA/etc) e robô
2. Clicar "Iniciar Treino"
3. Monitorar convergência pelos gráficos
4. Pausar e "Salvar Treino" quando estável

Caso 2: Retomada de Treinamento:
1. Clicar "Carregar Treino"
2. Selecionar sessão anterior
3. Clicar "Retomar Treino"
4. Continuar de onde parou

Caso 3: Análise de Performance:
1. Usar gráficos para identificar:
   - Episódios de maior recompensa
   - Estabilidade da marcha
   - Consistência temporal
2. Exportar gráficos para relatórios

## Aba AVALIAÇÃO - Teste de Modelos Individuais

### Objetivo
Avaliar o desempenho de um modelo específico em um circuito determinado.

### Fluxo Principal
1. Configuração do Teste:
Modelo: [Procurar...]   Ambiente: [PR ▼]   Robô: [robot_stage1 ▼]
Episódios: [20]   [Modo Determinístico] ✅ [Visualizar Robô]
2. Execução e Resultados:
[Executar Avaliação] → Métricas detalhadas + Gráficos de performance
3. Métricas Coletadas:
Taxa de sucesso e tempo médio
Distribuição de tempos por episódio
Progressão temporal e métricas consolidadas
Análise de consistência e performance
4. Funcionalidades de Exportação:
Exportar resultados para CSV
Salvar gráficos como PNG
Histórico de avaliações

### Casos de Uso Típicos
Caso 1: Validação de Modelo:
1. Selecionar modelo .zip treinado
2. Definir ambiente de teste (ex: PR)
3. Executar 20 episódios determinísticos
4. Analisar taxa de sucesso e consistência

Caso 2: Teste de Robustez:
1. Usar modelo treinado em PR
2. Testar em Pμ ou PG
3. Avaliar degradação de performance
4. Comparar com especialista nativo

Caso 3: Análise Estatística:
1. Executar múltiplas avaliações
2. Exportar dados para análise externa
3. Gerar gráficos para publicação

## Aba COMPARAÇÃO - Avaliação Cruzada Completa

### Objetivo
Executar automaticamente toda a avaliação cruzada conforme requisitos RF-08 a RF-11.

### Fluxo Principal
1. Configuração da Avaliação Cruzada:
Diretório de Modelos: [training_data/] [Procurar]
Episódios por avaliação: [20]   [Modo Determinístico]
2. Execução Automática:
[Executar Avaliação Cruzada Completa] → Sistema executa automaticamente:
3. Requisitos Atendidos Automaticamente:
Avaliação de Complexidade (6×20 = 120 execuções)
Avaliação de Generalização (6×5×20 = 600 execuções)
Avaliação de Especificidade (6×6×20 = 720 execuções)
Classificação Direcional (análise automática)
4. Resultados Consolidados:
Ranking de complexidade dos circuitos
Matriz de generalização entre circuitos
Gaps de especificidade (AE vs AG)
Análise de transferências direcionais

### Casos de Uso Típicos
Caso 1: Estudo de Generalização Completo: 
1. Colocar 6 modelos especialistas em training_data/
2. Clicar "Executar Avaliação Cruzada Completa"
3. Aguardar execução automática (1.440 avaliações)
4. Analisar relatório consolidado:
   - Quais especialistas generalizam melhor?
   - Transferências ascendentes vs descendentes
   - Circuitos mais "difíceis"

Caso 2: Análise Comparativa: 
1. Executar avaliação cruzada
2. Exportar relatório JSON completo
3. Usar heatmaps para identificar padrões
4. Gerar gráficos para publicação

Caso 3: Validação Científica: 
1. Reproduzir experimentos com seeds fixas
2. Validar hipóteses sobre transferência
3. Comparar diferentes arquiteturas de RL

# Estrutura do Projeto

phd-biped-rl/
├── src/
│   ├── agent.py                # Classe agente de RL.
│   ├── best_model_tracker.py   # Tracker dos melhores modelos.
│   ├── cross_evaluation.py     # Lógica de avaliação cruzada (RF-08 a RF-11)
│   ├── environment.py          # Classe ambiente/cenário. Carrega .xacro de ambientes.
│   ├── evaluate_model.py       # Calcula métricas estatisticas e salva.
│   ├── logger.py               # Sistema de Logger.
│   ├── main.py                 # Ponto de entrada. Inicializa a GUI e processos.
│   ├── metrics_saver.py        # Sistema de métricas e exportação
│   ├── robot.py                # Classe Robot. Carrega .xacro, obtém juntas.
│   ├── simulation.py           # Lógica de simulação e cálculo de recompensa.
│   ├── tab_training.py         # Aba de treinamento
│   ├── tab_evaluation.py       # Aba de avaliação individual
│   ├── tab_comparison.py       # Aba de avaliação cruzada
│   ├── simulation.py           # Lógica de simulação e recompensas
│   ├── train_process.py        # Lógica do treino
│   └── utils.py                # Utilitários e paths
├── environments/           # Modelos .xacro dos circuitos
├── especialistas/          # Modelos dos especialistasd treinados (model_PR.zip)
├── robots/                 # Modelos .xacro dos robôs
├── training_data/          # Modelos treinados e sessões
├── tmp/                    # Arquivos temporários (URDFs)
└── logs/                   # Logs de execução e dados
    ├── cross_evaluation/   # Relatórios de avaliação cruzada
    └── data/               # CSV com métricas detalhadas

# Saídas e Métricas

## Métricas Primárias:
Tempo Médio (Tm): Principal métrica para todos os RFs
Taxa de Sucesso: % de episódios completados com sucesso

## Métricas Secundárias:
Desvio Padrão do Tempo (σ-Tm)
Custo Energético (Σ|torque × velocity| × Δt)
Estabilidade (variância da inclinação)
Regularidade da Marcha

## Estrutura de Dados
Complexidade: 6 circuitos × 20 repetições = 120 registros
Generalização: 6 origens × 5 destinos × 20 repetições = 600 registros
Especificidade: 6 circuitos × 6 avatares × 20 repetições = 720 registros

# Aplicações de Pesquisa
Este framework permite:
Estudo sistemático de generalização em RL
Análise de transferência entre domínios perturbados
Comparação de algoritmos de RL em tarefas dinâmicas
Validação reprodutível com seeds fixas
Geração de dados para publicações científicas

# Licença
Para mais detalhes, consulte o arquivo LICENSE.
