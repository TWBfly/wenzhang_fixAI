"""测试叙事因果链完整性审计器 (NarrativeCausalityAuditor)"""
import sys
sys.path.insert(0, '/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI')

from advanced_analyzers import NarrativeCausalityAuditor

auditor = NarrativeCausalityAuditor()

# ============================================
# 测试1: 对 1_ai_fix.md 楔子的断层检测
# ============================================
print("=" * 60)
print("测试1: 1_ai_fix.md 楔子 (应该检测出断层)")
print("=" * 60)

ai_fix_opening = """指尖触到那叠泛黄的保帖时，宁禾的指腹猛地一僵——质地不对。

那是上好的宣纸，却没有她惯用的竹香。她抽出纸页，一滩殷红的朱砂印在烛火下狰狞如血。

"无限连带。"

四个字像一根烧红的铁钉，狠狠钻进她的瞳仁。宁禾听见自己喉咙里挤出极轻的一声冷笑，撕碎明细的纸屑落入火盆，火苗舔舐着炭黑，升起一股呛人的焦臭。铁皮柜的锁扣在指尖冰凉滑过，她反手锁死，钥匙攥进掌心，硌得骨头生疼。"""

report1 = auditor.audit_chapter(ai_fix_opening, "楔子")
print(f"\n严重级别: {report1['severity']}")
print(f"总问题数: {report1['total_issues']}")

if report1["opening_issues"]:
    print("\n🔴 开篇因果审计:")
    for oi in report1["opening_issues"]:
        print(f"  - {oi}")

if report1["dangling_entities"]:
    print("\n🟡 悬空实体:")
    for de in report1["dangling_entities"]:
        print(f"  - {de['reason']}")

if report1["causal_gaps"]:
    print("\n🟡 因果断裂:")
    for cg in report1["causal_gaps"]:
        print(f"  - {cg['reason']}")

# ============================================
# 测试2: 对 1.md 原文开头的检测 (应该通过)
# ============================================
print("\n" + "=" * 60)
print("测试2: 1.md 原文开头 (应该基本通过)")
print("=" * 60)

original_opening = """堂外的雨越落越急，砸在岁验堂的重檐上，溅起一地冰冷的碎玉。堂内烧着几盆银丝碳，却驱不散空气中那股潮湿的墨香与陈年纸张的腐气。

宁禾站在长案前，右手悬腕，最后一点朱砂在盐引联保账的末尾落下。朱砂鲜艳，却压不住她心底翻涌的那层凉意。她的指腹压着笔杆，指节处因为长久握笔而留下的薄茧，此刻在微微发麻。这支紫毫笔是她十七岁那年，师父临终前亲手传下的。

"宁丫头，这是最后一笔了？"

屏风后传来一声略显苍老的询问，伴随着盖碗拨动瓷缘的细响。

蒋承德慢条斯理地走了出来。作为青州商会的会首，他今日穿了一件暗枣红色的缂丝长衫，腰间的玉带钩在昏暗的烛火下泛着内敛的清光。他走到宁禾身边，目光在密密麻麻的账页上扫过，眼神里透着几分难言的复杂。"""

report2 = auditor.audit_chapter(original_opening, "第一章")
print(f"\n严重级别: {report2['severity']}")
print(f"总问题数: {report2['total_issues']}")

if report2["opening_issues"]:
    print("\n🔴 开篇因果审计:")
    for oi in report2["opening_issues"]:
        print(f"  - {oi}")
else:
    print("\n✅ 开篇因果完整性审计通过")

if report2["dangling_entities"]:
    print("\n🟡 悬空实体:")
    for de in report2["dangling_entities"]:
        print(f"  - {de['reason']}")

# ============================================
# 测试3: 对比总结
# ============================================
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)
print(f"1_ai_fix.md 楔子: 严重级别={report1['severity']}, 问题数={report1['total_issues']}")
print(f"1.md 原文开头:   严重级别={report2['severity']}, 问题数={report2['total_issues']}")

if report1['severity'] == 'P0' and report2['severity'] != 'P0':
    print("\n✅ 审计器正确区分了断层文本和正常文本！")
elif report1['total_issues'] > report2['total_issues']:
    print("\n✅ 审计器检测到 AI 修改版有更多问题（符合预期）")
else:
    print("\n⚠️ 审计器结果需要进一步调优")
