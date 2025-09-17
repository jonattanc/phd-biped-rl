# Biped RL in PyBullet

🦵 Biped RL in PyBullet — Generalização Cruzada
Este projeto implementa um framework para treinar e avaliar agentes de Aprendizado por Reforço (RL) no controle de um robô bípede em simulação física (PyBullet). O foco é estudar a generalização cruzada de políticas treinadas em diferentes ambientes perturbados.

Atualmente, o sistema está na Semana 2: Validação do Pipeline de Treinamento, com infraestrutura completa para simulação, logging, interface gráfica e registro de métricas.

# Objetivo Geral

Treinar 6 agentes especialistas (AE), cada um em um circuito distinto (PR, P<μ, RamA, RamD, PG, PRB), e avaliar sua capacidade de generalização quando testados nos outros 5 circuitos. A métrica principal é o Tempo Médio (Tm) para completar os 10 metros.

## Installation

### Install Python

This project is being developped with python 3.12.

### Clone the repository

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

### Interface Gráfica
Inicie o Treinamento: Selecione "PR" e "robot_stage1", clique em "Iniciar Treinamento".
Monitore em Tempo Real: Gráficos de Recompensa, Duração e Distância por episódio.
Controle: Pausar, Finalizar, Salvar Snapshot (metadados).
Logs: A caixa de texto exibe os logs mais recentes.

### Estrutura de Arquivos
phd-biped-rl/
├── src/
│   ├── agent.py          # Classe agente de RL.
│   ├── environment.py    # Classe ambiente/cenário. Carrega .xacro de ambientes.
│   ├── evaluate_model.py # Calcula métricas estatisticas e salva.
│   ├── gui.py            # Interface Tkinter + Gráficos Matplotlib + Logger integrado.
│   ├── gym_env.py        # Ambiente no padrão gym. Implementa step e reset.
│   ├── logger.py         # EpisodeLogger. Salva CSVs com metadados por episódio.
│   ├── main.py           # Ponto de entrada. Inicializa a GUI e processos.
│   ├── robot.py          # Classe Robot. Carrega .xacro, obtém juntas.
│   ├── simulation.py     # Lógica principal de simulação e cálculo de recompensa.
│   └── utils.py          # Utilidades gerais, como manipulação de arquivos e paths.
│── robots/               # Modelos de robôs .xacro.
│── environments/         # Modelos de ambientes/cenários .xacro.
├── tmp/                  # Diversos arquivos temporários, como URDFs gerados em tempo de execução.
└── logs/
    ├── log__{log_description}__proc{proc_num}.txt  # Log principal de cada processo.
    └── data/                                       # CSVs com metadados de cada episódio.

### Cronograma Implementado

Fundação: Avatar, Ambiente PR, Agente Aleatório.
Robô se move alguns cm antes de cair.

Validação do Pipeline: GUI, Gráficos, Logging, CSV.
Interface funcional, dados salvos corretamente.

➡️ Próximo
Primeiro Treinamento com RL (PPO).
Robô completa 10m com >10% de sucesso.

➡️ Próximo
Ajuste de Parâmetros e Primeiros Resultados Estatísticos.
/PR.csv
com Tm, σ-Tm, sucesso.

➡️ Futuro
Testar Generalização: Criar novos ambientes, treinar 6 AEs.
Avaliação cruzada, cálculo de ΔTm.

➡️ Futuro
Análise de Resultados e Elaboração de Tese.
Gráficos, conclusões sobre generalização.

### Licença
Para mais detalhes, consulte o arquivo LICENSE.
