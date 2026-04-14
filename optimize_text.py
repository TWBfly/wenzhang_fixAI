#!/usr/bin/env python3
"""
文本优化脚本 - 根据fixAI_prompt.md规则优化原文
"""

import re
from pathlib import Path


def load_text(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def save_text(file_path: str, text: str) -> None:
    Path(file_path).write_text(text, encoding="utf-8")


def optimize_text(text: str) -> str:
    """
    根据fixAI_prompt.md规则优化文本
    """
    
    # 1. 替换"微微" - 改为具体动作或删除
    text = re.sub(r'眼眶微微发胀', '眼眶发胀', text)
    text = re.sub(r'微微侧了侧脸', '侧过脸', text)
    text = re.sub(r'心里微微一暖', '心里一暖', text)
    text = re.sub(r'微微发颤', '发颤', text)
    
    # 2. 替换"缓缓" - 改为直接动作
    text = re.sub(r'缓缓站起身', '站起身', text)
    text = re.sub(r'缓缓开口', '开口', text)
    
    # 3. 替换"顿了顿" - 直接切入
    text = re.sub(r'她顿了顿，', '', text)
    text = re.sub(r'他顿了顿，', '', text)
    text = re.sub(r'脚步顿了顿', '脚步一停', text)
    
    # 4. 替换"仿佛" - 改为具体描述
    text = re.sub(r'神色淡然得仿佛刚才那个要杀人的人不是她', 
                  '神色淡然。刚才要杀人的不像她', text)
    text = re.sub(r'像是一根缝衣针在脑浆里使劲儿搅合', 
                  '像缝衣针在脑浆里搅', text)
    
    # 5. 替换"显然" - 改为动作表现
    text = re.sub(r'崔老夫人显然没料到', '崔老夫人没料到', text)
    
    # 6. 替换"不是A而是B"句式
    text = re.sub(r'这种冷，不是因为天气的寒，而是因为这屋子里没多少人气儿',
                  '屋里冷。不是天气的寒，是没人气儿', text)
    text = re.sub(r'指尖触到的不是温热的余烬，而是那种黏糊糊、带着湿气的黑灰',
                  '指尖触到黏糊糊的黑灰，带着湿气。余烬早凉了', text)
    
    # 7. 替换"她知道/他明白"等解释型旁白
    text = re.sub(r'她知道，林挽枝说的是实话', '王嬷嬷信了', text)
    
    # 8. 替换"目光落在" 
    text = re.sub(r'目光落在了', '盯着', text)
    text = re.sub(r'目光落在', '视线扫过', text)
    
    # 9. 替换"取而代之"
    text = re.sub(r'取而代之的是', '换上', text)
    
    # 10. 替换"沉默片刻"等过渡
    text = re.sub(r'沉默片刻', '沉默', text)
    text = re.sub(r'静了片刻', '安静下来', text)
    
    # 11. 打破对称句式
    text = re.sub(r'既有([^，]+)也有([^，]+)', r'\1。\2也不少', text)
    
    # 12. 删除百科式套话
    text = re.sub(r'总的来说', '', text)
    text = re.sub(r'值得注意的是', '', text)
    text = re.sub(r'不可忽视的是', '', text)
    text = re.sub(r'综上所述', '', text)
    text = re.sub(r'显而易见', '', text)
    
    # 13. 优化"这意味着"
    text = re.sub(r'这意味着', '说明', text)
    
    # 14. 替换"本质上"
    text = re.sub(r'本质上', '说到底', text)
    
    # 15. 优化情绪标签 - 改为具体动作
    text = re.sub(r'眼神里闪过一抹极其复杂的情绪', 
                  '眼神复杂。嫌恶、怀疑，还有迟疑', text)
    
    # 16. 替换"仿佛"的其他用法
    text = re.sub(r'像是一尊石刻的塑像', '像尊石刻', text)
    text = re.sub(r'像是一把绷紧的弦', '像根绷紧的弦', text)
    
    # 17. 删除冗余的"似乎"
    text = re.sub(r'似乎隐约传来', '隐约传来', text)
    text = re.sub(r'似乎已经', '已经', text)
    
    # 18. 优化"感觉到"
    text = re.sub(r'她感觉到', '她感到', text)
    text = re.sub(r'他感觉到', '他感到', text)
    
    # 19. 打破长句，增加短句节奏
    # 在适当位置断句
    text = re.sub(r'，那股子([^，]{10,20}?)的([^，]{10,20}?)，直往', 
                  r'。\1的\2直往', text)
    
    # 20. 优化"那种"
    text = re.sub(r'那种([^，]{5,15}?)的感觉', r'\1的劲儿', text)
    text = re.sub(r'那种([^，]{5,15}?)的样子', r'\1的样', text)
    
    return text


def main():
    input_path = "/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_zhengwen/2/原文.md"
    output_path = "/Users/tang/PycharmProjects/pythonProject/wenzhang_fixAI/duanpian_zhengwen/2/原文_qwen优化.md"
    
    print(f"读取原文: {input_path}")
    original_text = load_text(input_path)
    
    print("开始优化...")
    optimized_text = optimize_text(original_text)
    
    print(f"保存优化文本: {output_path}")
    save_text(output_path, optimized_text)
    
    print("优化完成!")
    print(f"原文长度: {len(original_text)} 字符")
    print(f"优化后长度: {len(optimized_text)} 字符")


if __name__ == "__main__":
    main()
