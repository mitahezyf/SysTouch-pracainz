#kciuk
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

#wskazujacy
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

#srodkowy
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

#serdeczny
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

#maly
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

#nadgarstek
WRIST = 0

#slownik

#zbiory calych palcow
FINGERS = {
    "thumb": [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    "index": [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    "middle": [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    "ring": [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    "pinky": [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
}

#opuszki palcow
FINGER_TIPS = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}

#pierwszy staw liczac od opuszka
FINGER_DIPS = {
    "thumb": THUMB_CMC,
    "index": INDEX_DIP,
    "middle": MIDDLE_DIP,
    "ring": RING_DIP,
    "pinky": PINKY_DIP,
}

#druku staw liczac od opuszka
FINGER_PIPS = {
    "index": INDEX_PIP,
    "middle": MIDDLE_PIP,
    "ring": RING_PIP,
    "pinky": PINKY_PIP,
}

#miejsce gdzie palec laczy sie z nadgarstkiem
FINGER_MCPS = {
    "thumb": THUMB_MCP,
    "index": INDEX_MCP,
    "middle": MIDDLE_MCP,
    "ring": RING_MCP,
    "pinky": PINKY_MCP,
}
WHOLE_HAND = list(range(21))

