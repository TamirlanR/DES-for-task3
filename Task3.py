import random

def bstr(b: bytes) -> str:
    return " ".join(f"{byte:08b}" for byte in b)

def flip_bit_in_bytes(b: bytes, bit_index: int) -> bytes:

    if len(b) != 8:
        raise ValueError("Expected 8 bit.")
    byte_idx = bit_index // 8
    
    bit_pos_in_byte = bit_index % 8
    mask = 1 << (7 - bit_pos_in_byte)
    ba = bytearray(b)
    ba[byte_idx] ^= mask
    return bytes(ba)

def hamming_distance(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError("lengths are does not match ")
    dist = 0
    for x, y in zip(a, b):
        v = x ^ y
        dist += v.bit_count()  # Python 3.8+: count bits set to 1
    return dist

IP = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]

FP = [
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41,  9, 49, 17, 57, 25
]

E = [
    32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9,10,11,12,13,
    12,13,14,15,16,17,
    16,17,18,19,20,21,
    20,21,22,23,24,25,
    24,25,26,27,28,29,
    28,29,30,31,32, 1
]

P = [
    16, 7, 20, 21, 29, 12, 28, 17,
     1,15, 23, 26,  5, 18, 31, 10,
     2, 8, 24, 14, 32, 27,  3,  9,
    19,13, 30,  6, 22, 11,  4, 25
]

PC1 = [
    57,49,41,33,25,17, 9,
     1,58,50,42,34,26,18,
    10, 2,59,51,43,35,27,
    19,11, 3,60,52,44,36,
    63,55,47,39,31,23,15,
     7,62,54,46,38,30,22,
    14, 6,61,53,45,37,29,
    21,13, 5,28,20,12, 4
]

PC2 = [
    14,17,11,24, 1, 5,
     3,28,15, 6,21,10,
    23,19,12, 4,26, 8,
    16, 7,27,20,13, 2,
    41,52,31,37,47,55,
    30,40,51,45,33,48,
    44,49,39,56,34,53,
    46,42,50,36,29,32
]

# S-boxes (8 boxes, 4x16 each)
SBOX = [
    # S1
    [
        [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
        [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
        [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
        [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13],
    ],
    # S2
    [
        [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
        [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
        [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
        [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9],
    ],
    # S3
    [
        [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
        [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
        [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
        [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12],
    ],
    # S4
    [
        [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
        [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
        [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
        [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14],
    ],
    # S5
    [
        [2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
        [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
        [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
        [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3],
    ],
    # S6
    [
        [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
        [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
        [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
        [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13],
    ],
    # S7
    [
        [4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
        [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
        [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
        [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12],
    ],
    # S8
    [
        [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
        [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
        [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
        [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11],
    ],
]

# --- helpers ---
def bytes_to_bits(b: bytes) -> list[int]:
    return [(byte >> (7 - i)) & 1 for byte in b for i in range(8)]

def bits_to_bytes(bits: list[int]) -> bytes:
    return bytes(int(''.join(str(bit) for bit in bits[i:i+8]), 2) for i in range(0, len(bits), 8))

def permute(bits: list[int], table: list[int]) -> list[int]:
    # DES tables are 1-based
    return [bits[i-1] for i in table]

def left_rotate(lst: list[int], n: int) -> list[int]:
    return lst[n:] + lst[:n]

def xor_bits(a: list[int], b: list[int]) -> list[int]:
    return [x ^ y for x, y in zip(a, b)]

def sbox_substitution(bits48: list[int]) -> list[int]:
    out = []
    for box in range(8):
        chunk = bits48[box*6:(box+1)*6]
        row = (chunk[0] << 1) | chunk[5]
        col = (chunk[1] << 3) | (chunk[2] << 2) | (chunk[3] << 1) | chunk[4]
        val = SBOX[box][row][col]
        out.extend([(val >> 3) & 1, (val >> 2) & 1, (val >> 1) & 1, val & 1])
    return out

# round shifts schedule
SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

def key_schedule(key64_bits: list[int]) -> list[list[int]]:
    # drop parity bits -> 56 bits
    key56 = permute(key64_bits, PC1)
    C = key56[:28]
    D = key56[28:]
    subkeys = []
    for s in SHIFTS:
        C = left_rotate(C, s)
        D = left_rotate(D, s)
        cd = C + D
        k48 = permute(cd, PC2)
        subkeys.append(k48)
    return subkeys

def feistel(right32: list[int], subkey48: list[int]) -> list[int]:
    expanded = permute(right32, E)                # 32 -> 48
    xored = xor_bits(expanded, subkey48)          # + subkey
    sboxed = sbox_substitution(xored)             # 48 -> 32
    return permute(sboxed, P)                     # permute P

def des_encrypt_block(block8: bytes, key8: bytes) -> bytes:
    # 1) initial permutation
    block_bits = bytes_to_bits(block8)
    ip = permute(block_bits, IP)
    L = ip[:32]
    R = ip[32:]

    # 2) key schedule
    key_bits = bytes_to_bits(key8)
    subkeys = key_schedule(key_bits)

    # 3) 16 rounds
    for i in range(16):
        f = feistel(R, subkeys[i])
        L, R = R, xor_bits(L, f)

    # 4) preoutput (R16, L16), final permutation
    rl = R + L
    fp = permute(rl, FP)
    return bits_to_bytes(fp)


def block_from_A(A: str) -> bytes:
    b = A.encode('utf-8')
    return b[:8] if len(b) >= 8 else b + b'\x00' * (8 - len(b))

def block_from_B(B: int) -> bytes:
    # 8-byte big-endian
    return B.to_bytes(8, 'big', signed=False)

A = "Tamirlan Ramazanov"
B = 20250704


plaintext_block = block_from_A(A)  
key_block = block_from_B(B)  


ct0 = des_encrypt_block(plaintext_block, key_block)

print("Plaintext (hex):", plaintext_block.hex())
print("Key       (hex):", key_block.hex())
print("CT base   (hex):", ct0.hex())


random.seed(42)  
bit_pt = random.randrange(64) 
pt_flipped = flip_bit_in_bytes(plaintext_block, bit_pt)
ct_ptflip = des_encrypt_block(pt_flipped, key_block)

hd_pt = hamming_distance(ct0, ct_ptflip)

print("\n[Plaintext bit flip]")
print(f"Flipped bit index in PT: {bit_pt}")
print("PT flipped (hex):", pt_flipped.hex())
print("PT flipped (bin):", bstr(pt_flipped))
print("CT after PT flip (hex):", ct_ptflip.hex())
print("CT after PT flip (bin):", bstr(ct_ptflip))
print("Changed bits in ciphertext:", hd_pt)


bit_k = random.randrange(64)
key_flipped = flip_bit_in_bytes(key_block, bit_k)
ct_kflip = des_encrypt_block(plaintext_block, key_flipped)

hd_k = hamming_distance(ct0, ct_kflip)

print("\n[Key bit flip]")
print(f"Flipped bit index in KEY: {bit_k}")
print("KEY flipped (hex):", key_flipped.hex())
print("KEY flipped (bin):", bstr(key_flipped))
print("CT after KEY flip (hex):", ct_kflip.hex())
print("CT after KEY flip (bin):", bstr(ct_kflip))
print("Changed bits in ciphertext:", hd_k)


