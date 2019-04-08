import subprocess


def test_classification():
    assert subprocess.run('python tests/main_classification.py', shell=True, check=True)


def test_autoencoder():
    assert subprocess.run('python tests/main_autoencoder.py', shell=True, check=True)

