import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from duanpian_fixAI_1 import EnhancedNovelAISurgery, NovelContext, SentenceChange


class WenzhangFixAITest(unittest.TestCase):
    def make_engine(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        return EnhancedNovelAISurgery(root_dir=root, db_path=root / "db" / "test_logic.db")

    def tearDown(self):
        tmp = getattr(self, "tmp", None)
        if tmp:
            tmp.cleanup()

    def test_split_sentences_keeps_punctuation(self):
        engine = self.make_engine()
        self.assertEqual(
            engine.split_sentences("她停住了。风钻进袖口！还走吗？走。"),
            ["她停住了。", "风钻进袖口！", "还走吗？", "走。"],
        )

    def test_rule_detection_finds_ai_pattern(self):
        engine = self.make_engine()
        hits = engine.detect_rules_in_sentence("测试", 0, 0, "这意味着她必须重新判断局势。")
        self.assertTrue(any(hit.rule_name == "这意味着" for hit in hits))

    def test_explanatory_tail_tag_detector(self):
        engine = self.make_engine()
        issues = engine.tail_tag_detector.check("大腿根部的肌肉在打颤——那是原主身体残余的恐惧。")
        self.assertTrue(issues)
        self.assertIn("解释性尾注", issues[0]["reason"])

    def test_modern_to_ancient_mapping(self):
        engine = self.make_engine()
        applied = []
        revised = engine.map_modern_to_ancient("他拿出手机看了一眼。", "Historical", applied)
        self.assertIn("书信", revised)
        self.assertIn("词汇映射:手机->书信", applied)

    def test_analyze_chapter_is_pure_by_default(self):
        engine = self.make_engine()
        calls = []

        def fake_extract(chapter, text, characters=None):
            calls.append((chapter, text, characters))

        engine.logic_auditor.extract_facts = fake_extract
        context = NovelContext("book", "Unknown", "Unknown", "Serious", 0)
        engine.analyze_chapter("一", "她走进屋里。", context)
        self.assertEqual(calls, [])
        engine.analyze_chapter("一", "她走进屋里。", context, persist=True)
        self.assertEqual(len(calls), 1)

    def test_export_tasks_only_does_not_call_llm_and_uses_excerpt_by_default(self):
        engine = self.make_engine()
        text = "她推开门，冷风钻进袖口。\n\n" + "这是一个需要保留的长文本标记。\n" * 300
        context = NovelContext("book", "Unknown", "Historical", "Serious", len(text), [])
        with patch("duanpian_fixAI_1.call_llm_with_fallback") as mocked_llm:
            revised, _ = engine.rewrite_chapter(
                "长文测试",
                text,
                context=context,
                enable_semantic_opt=True,
                export_tasks_only=True,
            )
        mocked_llm.assert_not_called()
        self.assertIn("需要保留的长文本标记", revised)
        task_path = Path(engine.root_dir) / "agent_tasks" / "refine_长文测试.md"
        task_text = task_path.read_text(encoding="utf-8")
        self.assertIn("默认不内联全文", task_text)
        self.assertLess(task_text.count("需要保留的长文本标记"), 80)
        self.assertNotIn("```text\n" + text, task_text)

    def test_task_full_text_is_explicit_opt_in(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        engine = EnhancedNovelAISurgery(
            root_dir=root,
            db_path=root / "db" / "test_logic.db",
            task_include_full_text=True,
        )
        text = "她推开门，冷风钻进袖口。\n\n" + "这是一个需要保留的长文本标记。\n" * 20
        context = NovelContext("book", "Unknown", "Historical", "Serious", len(text), [])
        with patch("duanpian_fixAI_1.call_llm_with_fallback") as mocked_llm:
            engine.rewrite_chapter("全文任务测试", text, context=context, export_tasks_only=True)
        mocked_llm.assert_not_called()
        task_path = Path(engine.root_dir) / "agent_tasks" / "refine_全文任务测试.md"
        self.assertIn("## 参考全文内容", task_path.read_text(encoding="utf-8"))

    def test_llm_final_text_validation_rejects_truncation(self):
        engine = self.make_engine()
        original = "张三推开门，冷风钻进袖口。" * 80
        context = NovelContext("book", "Unknown", "Historical", "Serious", len(original), [{"name": "张三"}])
        ok, reason = engine.validate_llm_final_text(original, "张三推开门。", context)
        self.assertFalse(ok)
        self.assertIn("word_count_out_of_range", reason)

    def test_long_titled_chapter_is_chunked(self):
        engine = self.make_engine()
        text = "第一章\n" + ("她推开门，冷风钻进袖口。\n\n" * 160)
        chunks = engine.split_book_by_chapters(text, max_chunk_size=800)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(body) <= 900 for _, body in chunks))

    def test_persisted_logic_audit_does_not_call_llm_by_default(self):
        engine = self.make_engine()
        context = NovelContext("book", "Unknown", "Unknown", "Serious", 0, [])
        text = "她拿走了钱。\n" + "她走进屋里。" * 120
        with patch("duanpian_fixAI_1.generate_text_safe") as mocked_generate:
            engine.analyze_chapter("一", text, context, persist=True)
        mocked_generate.assert_not_called()

    def test_targeted_llm_patch_sends_only_selected_paragraphs(self):
        engine = self.make_engine()
        engine.finalize_llm_caller = lambda prompt, sys_prompt: '[{"paragraph": 2, "text": "她把账册重新翻到末页，先前少掉的一笔银两终于对上了。"}]'
        context = NovelContext("book", "Unknown", "Historical", "Serious", 0, [{"name": "她"}])
        text = "\n\n".join([
            "堂外雨声压着窗纸。",
            "这意味着她必须重新判断局势。",
            "她把灯芯拨亮，继续往下查。",
            "门外有人低声咳了一下。",
        ])
        changes = [
            SentenceChange(
                "章",
                2,
                1,
                "这意味着她必须重新判断局势。",
                "[审计建议: 局部修复，不自动改写]",
                ["这意味着", "P1"],
                "low",
            )
        ]
        patched = engine.execute_targeted_llm_patches("章", text, context, changes, [])
        self.assertIn("少掉的一笔银两终于对上", patched)
        self.assertIn("堂外雨声压着窗纸", patched)
        self.assertIn("门外有人低声咳了一下", patched)

    def test_patch_prompt_respects_token_budget(self):
        engine = self.make_engine()
        engine.llm_token_budget = 900
        engine.llm_max_patches = 10
        context = NovelContext("book", "Unknown", "Historical", "Serious", 0, [])
        text = "\n\n".join([f"第{i}段。" + "她推开门，冷风钻进袖口。" * 20 for i in range(1, 12)])
        changes = [
            SentenceChange("章", i, 1, "这意味着她必须重新判断局势。", "[审计建议: 局部修复，不自动改写]", ["这意味着", "P1"], "low")
            for i in range(1, 12)
        ]
        prompt, targets = engine.build_targeted_llm_patch_prompt("章", text, context, changes, [], list(range(1, 12)))
        self.assertTrue(targets)
        self.assertLessEqual(engine.estimate_tokens(prompt), engine.llm_token_budget)

    def test_evolution_rules_enter_pending_before_activation(self):
        engine = self.make_engine()
        engine.evolution_engine.add_evolution_rule("测试候选规则", r"玄妙测试词", 1.0, "test", "P1")
        with sqlite3.connect(engine.db_path) as conn:
            pending = conn.execute(
                "SELECT status FROM pending_evolution_rules WHERE name='测试候选规则'"
            ).fetchone()
            active = conn.execute(
                "SELECT name FROM evolution_rules WHERE name='测试候选规则'"
            ).fetchone()
        self.assertEqual(pending[0], "pending")
        self.assertIsNone(active)

        promoted = engine.evolution_engine.approve_pending_rules(
            positive_samples=["这里有玄妙测试词。"],
            negative_samples=["她把茶杯放回桌上。"],
        )
        self.assertEqual(promoted, 1)
        with sqlite3.connect(engine.db_path) as conn:
            active = conn.execute(
                "SELECT name FROM evolution_rules WHERE name='测试候选规则'"
            ).fetchone()
        self.assertEqual(active[0], "测试候选规则")


if __name__ == "__main__":
    unittest.main()
