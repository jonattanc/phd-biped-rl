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
cd phd_biped_rl
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
