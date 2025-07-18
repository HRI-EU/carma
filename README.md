# CARMA: Context-Aware Situational Grounding Combining Vision-Language Models with Object and Action Recognition

## Installation

The framework is set up for Unix systems.

#### 1. Set OpenAI key
To use GPT4, you need to set
```bash
export OPENAI_API_KEY="53CRE7_KEY"
```

#### 2. Clone repository
```bash
git clone https://github.com/HRI-EU/carma.git
cd carma
```

##### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
# activate virtual environment
source venv/bin/activate # Linux
call venv\Scripts\activate # Windows
# Install dependencies
pip install -e .
```

## Example

```bash
# activate virtual environment
source venv/bin/activate # Linux
call venv\Scripts\activate # Windows
python -m examples.carma
```
The default experiment is Sorting Fruits including one person, according to the paper.
You can change the system configuration modifying the main.py here: \
https://github.com/HRI-EU/carma/blob/main/examples/carma.py#L302 \
The first boolean value switches on/off the usage of the action label, the second boolean controls the action trigger and 
the last boolean allows to use the previous triplet in the prompt or not.

Same as for running carma, you can evaluate the different runs, using the boolean set again at: \
https://github.com/HRI-EU/carma/blob/main/src/evaluation/evaluate_carma.py#L377 \
The default experiment is again Sorting Fruits including one person.

``` bash
# run evaluation
python -m evaluation.evaluate_carma
```
