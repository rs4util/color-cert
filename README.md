# COLOR-CERT: Context-Aware Randomized Smoothing via Colorization-Based Entropy

This is a repository containing the code of the paper "COLOR-CERT: Context-Aware Randomized Smoothing via Colorization-Based Entropy".
The repository heritates heavily from [Interactive Deep Colorization](https://github.com/junyanz/interactive-deep-colorization) and [Macer](https://github.com/RuntianZ/macer).

Clone the repository 
```
git clone https://github.com/rs4util/color-cert.git
cd color-cert
```

## Install dependencies.

1. Create and activate Python 3 virtual environment
```
Python3 -m venv env
source env/bin/activate
```
2. Installing packages
```
pip install --upgrade pip
pip install -r requirements.txt
```
3. Download the colorization model by running the following script
```
./download.sh
```
## Training
Start training by running
```
./train.sh
```

## Test
Start testing/certifying by running
```
./test.sh
```
