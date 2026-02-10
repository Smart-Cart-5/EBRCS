---
title: EBRCS Smart Checkout
emoji: 🛒
colorFrom: orange
colorTo: red
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
---

# 🛒 EBRCS 스마트 체크아웃 시스템

AI 기반 실시간 상품 인식 자동 체크아웃 데모

## 🎯 기능

- **실시간 카메라 인식**: 라이브 카메라로 상품 자동 인식
- **영상 업로드**: 녹화된 영상 분석
- **ROI 설정**: 관심 영역 지정으로 정확도 향상
- **상품 등록**: 관리자용 새 상품 추가
- **영수증 검수**: 인식된 상품 수정 및 확인

## 🤖 AI 모델

- **DINOv3** (facebook/dinov3-vitl16-pretrain-lvd1689m) - 가중치 0.7
- **CLIP** (openai/clip-vit-base-patch32) - 가중치 0.3
- **FAISS** IndexFlatIP - 빠른 유사도 검색

## 📱 사용 방법

### 데스크톱
1. 홈 화면에서 "체크아웃 시작" 클릭
2. 카메라 또는 영상 업로드 선택
3. ROI 설정 (선택 사항)
4. 상품을 카메라에 보여주면 자동 인식
5. "영수증 검수" 페이지에서 확인 및 수정

### 모바일
1. 좌측 사이드바에서 "📱 모바일 체크아웃" 선택
2. 카메라 인덱스 선택 (기본: 0)
3. ROI 설정 후 체크아웃 시작
4. 영수증 검수 및 확인

## 🔧 기술 스택

- Streamlit
- PyTorch
- Transformers (Hugging Face)
- FAISS
- OpenCV
- PEFT (LoRA)

## 📊 성능

- 추론 속도: ~200-500ms/프레임 (T4 GPU)
- 매칭 임계값: 0.62
- 쿨다운: 1초

## 🎓 프로젝트 정보

멋쟁이사자처럼 대학교 최종 프로젝트 - AI 기반 무인 체크아웃 시스템
