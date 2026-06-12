from transfer_attack_core import ALL_ATTACKS

print(ALL_ATTACKS)

if "FAUG" in ALL_ATTACKS:
    print("FAUG registered successfully")
else:
    print("FAUG NOT FOUND")