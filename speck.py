from os import urandom

import numpy as np

# Constants
WORD_SIZE: int = 16
ALPHA: int = 7
BETA: int = 2
MASK_VAL: int = 2 ** WORD_SIZE - 1


def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE - k))


def ror(x, k):
    return (x >> k) | ((x << (WORD_SIZE - k)) & MASK_VAL)


def encrypt_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA)
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA)
    c1 = c1 ^ c0
    return (c0, c1)


def decrypt_one_round(c, k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA)
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA)
    return (c0, c1)


def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k) - 1]
    l = list(reversed(k[: len(k) - 1]))
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = encrypt_one_round((l[i % 3], ks[i]), i)
    return ks


def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x, y = encrypt_one_round((x, y), k)
    return (x, y)


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = decrypt_one_round((x, y), k)
    return (x, y)


def check_testvector():
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694C)
    ks = expand_key(key, 22)
    ct = encrypt(pt, ks)
    if ct == (0xA868, 0x42F2):
        print("Testvector verified.")
        return True
    else:
        print("Testvector not verified.")
        return False


def convert_to_binary(arr):
    """
    Takes an array of ciphertext pairs. The array's rows contain lefthand and
    righthand sides of the ciphertexts in alternating fashion. In other words,
    the first row contains the lefthand side of a ciphertext. The second row
    contains the righthand side of a ciphertext. The third row contains the
    lefthand side of a ciphertext and so on.

    An array of bit vectors containing the same data is returned.
    """
    X = np.zeros((4 * WORD_SIZE, len(arr[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE):
        index = i // WORD_SIZE
        offset = WORD_SIZE - (i % WORD_SIZE) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X


def readcsv(datei):
    """
    Takes a text file that contains encrypted block0, block1, true diff prob,
    real or random data. Samples are line separated, the above items
    whitespace-separated returns train data, ground truth, optimal DDT
    prediction.
    """
    data = np.genfromtxt(
        datei, delimiter=" ", converters={x: lambda s: int(s, 16) for x in range(2)}
    )
    X0 = [data[i][0] for i in range(len(data))]
    X1 = [data[i][1] for i in range(len(data))]
    Y = [data[i][3] for i in range(len(data))]
    Z = [data[i][2] for i in range(len(data))]
    ct0a = [X0[i] >> 16 for i in range(len(data))]
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))]
    ct0b = [X1[i] >> 16 for i in range(len(data))]
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))]
    ct0a = np.array(ct0a, dtype=np.uint16)
    ct1a = np.array(ct1a, dtype=np.uint16)
    ct0b = np.array(ct0b, dtype=np.uint16)
    ct1b = np.array(ct1b, dtype=np.uint16)

    # X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b])
    Y = np.array(Y, dtype=np.uint8)
    Z = np.array(Z)
    return (X, Y, Z)


def make_train_data(n, nr, diff=(0x0040, 0)):
    """Baseline training data generator."""
    # Generate labels
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1

    # Generate keys
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)

    # Generate plaintexts
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    # Apply input difference
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)

    plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)

    # Expand keys & encrypt plaintexts
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)

    # Convert to input format for neural networks
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    return X, Y


def real_differences_data(n, nr, diff=(0x0040, 0)):
    """Real differences data generator."""
    # Generate labels
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1

    # Generate keys
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)

    # Generate plaintexts
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    # Apply input difference
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)

    # Expand keys and encrypt
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)

    # Generate blinding values
    k0 = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    k1 = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)

    # Apply blinding to the samples labelled as random
    ctdata0l[Y == 0] = ctdata0l[Y == 0] ^ k0
    ctdata0r[Y == 0] = ctdata0r[Y == 0] ^ k1
    ctdata1l[Y == 0] = ctdata1l[Y == 0] ^ k0
    ctdata1r[Y == 0] = ctdata1r[Y == 0] ^ k1

    # Convert to input data for neural networks
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    return X, Y
