These instructions were tested for Ubuntu 20.04

# Install apt dependencies

```bash
$ sudo apt install build-essential python3-venv python3-pip
```

# Create virtual environment

```bash
$ python3 -m venv <virtual environment directory>
$ source <virtual environment directory>/bin/activate
$ pip install --upgrade pip
$ pip install wheel
$ pip install -r requirements.txt
```
