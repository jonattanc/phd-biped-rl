# Biped RL in PyBullet

This project trains reinforcement learning agents to control a simple biped robot in PyBullet.  
The robot has joint motors and an inertial sensor.  
Agents are trained on different terrains (flat, slope, steps, rough) and later tested across all scenarios to study generalization.

## Installation

### Install Python

This project is being developped with python 3.12.7.

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

## Usage

### Activate the virtual environment

```
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows
```

### Run

```
python main.py
```

### Estrutura de Arquivos
```
PHD-BIPED-RL/
├── src/
│   ├── main.py
│   ├── robot.py
│   └── simulation.py
├── envs/ 
│   └── exoskeleton_pr_stage1.py 
├── models/
│   ├── robots/
│   │   ├── robot_stage1.xacro      # Avatar Minimalista
│   │   ├── robot_stage2.xacro      # Avatar Básico
│   │   └── robot_stage3.xacro      # Avatar Completo
│   └── environments/
│       ├── PR.urdf"     # Plano Regular
│       ├── PBA.urdf"    # Plano Baixo Atrito
│       ├── PRA.urdf"    # Plano Rampa Ascendente
│       ├── PRD.urdf"    # Plano Rampa Descendente
│       ├── PG.urdf"     # Plano Granular
│       └── PRB.urdf"    # Plano Regular com Bloqueio
├── tmp/
└── logs/
```