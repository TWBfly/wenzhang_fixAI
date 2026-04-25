import sys
import os
sys.path.append('/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI')
from advanced_analyzers import ShowDontTellTransformer, IdentitySummaryDetector, ConjunctionNeutering

def test_analyzers():
    sdt = ShowDontTellTransformer()
    identity = IdentitySummaryDetector()
    conj = ConjunctionNeutering()
    
    # Test 1: SDT and Abstract Lyricism
    text1 = "他感到极其恐惧，陷入了无尽的深渊。由于他很害怕，他意识到这是一抹诡异的色彩。"
    flags1 = sdt.check(text1)
    print(f"SDT Flags: {flags1}")
    
    # Test 2: Identity Summary
    text2 = "他是一个极其出色的猎手，也是个准备割开自己颈动脉的疯子。"
    flags2 = identity.check(text2)
    print(f"Identity Flags: {flags2}")
    
    # Test 3: Conjunction Neutering
    text3 = "随着时间的推移，他不仅感到了寒冷，而且发现自己无路可走。然而，他并没有放弃，从而坚持了下去。"
    flags3 = conj.check(text3)
    print(f"Conjunction Flags: {[f['hit'] for f in flags3]}")

if __name__ == "__main__":
    test_analyzers()
