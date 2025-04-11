# Visual Recognition Package

## Installation

#### 1. Clone repository
```bash
git clone https://github.com/HRI-EU/carma.git
cd carma
```

##### 2. Install Dependencies 

```bash
python -m venv venv
source venv/bin/activate # Linux
source venv/Scripts/activate # Windows
pip install -e .
```

## Examples

```bash
source venv/bin/activate # Linux
source venv/Scripts/activate # Windows
```

#### Carma
``` bash
# run experiment
python -m examples.carma
```
``` bash
# run evaluation
python -m evaluation.evaluate_carma
```