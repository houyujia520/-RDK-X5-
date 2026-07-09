#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import easyocr

# ---------- 配置 ----------
USE_GPU = False
OCR_CONFIDENCE_THRESHOLD = 0.3
TXT_PATH = os.path.join(os.path.dirname(__file__), "med-txt.txt")

# ---------- 全局变量 ----------
reader = None
drug_database = {}

def load_drug_database():
    global drug_database
    if not os.path.exists(TXT_PATH):
        print(f"警告: 未找到文件 {TXT_PATH}，用量建议功能将不可用。")
        return

    try:
        print("正在加载药品数据库...")
        with open(TXT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 16:
                    continue
                drug_name = parts[2].strip()
                advice = parts[15].strip() if len(parts) > 15 else "暂无详细建议"
                if drug_name:
                    drug_database[drug_name] = advice
        print(f"数据库加载完成，共收录 {len(drug_database)} 种药品。")
    except Exception as e:
        print(f"加载数据库失败: {e}")

def init_ocr():
    global reader
    print("正在加载 OCR 模型... (首次运行可能较慢)")
    reader = easyocr.Reader(['ch_sim'], gpu=USE_GPU, verbose=False)
    print("OCR 模型加载完成。")
    load_drug_database()

def filter_drug_text(raw_results):
    filtered = []
    for (bbox, text, confidence) in raw_results:
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            continue
        clean_text = re.sub(r'[^\w\u4e00-\u9fff\s\./\(\)\-]', '', text).strip()
        if not clean_text or len(clean_text) < 2:
            continue
        filtered.append({
            "text": clean_text,
            "confidence": confidence,
            "bbox": bbox
        })
    filtered.sort(key=lambda x: x['confidence'], reverse=True)
    return filtered

def match_drug_advice(ocr_results):
    if not drug_database:
        return None, None
    for item in ocr_results:
        ocr_text = item['text']
        for drug_name, advice in drug_database.items():
            if drug_name in ocr_text or ocr_text in drug_name:
                return drug_name, advice
    return None, None