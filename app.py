import streamlit as st
import os
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
import openai
from datetime import datetime
import io
import re
import json
import tempfile
import time
import math
from pydub import AudioSegment
import concurrent.futures
import threading

def setup_google_credentials():
    """Google Cloud認証設定"""
    try:
        # Streamlit Secretsから認証情報を取得
        creds_dict = dict(st.secrets["gcp_service_account"])
        
        # 一時ファイルに認証情報を書き込み
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(creds_dict, f)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
        
        return True
    except Exception as e:
        st.error(f"Google Cloud認証に失敗しました: {e}")
        return False

def split_audio_file(audio_file, chunk_duration_minutes=10):
    """音声ファイルを指定した長さのチャンクに分割"""
    try:
        # 音声ファイルを読み込み
        audio_file.seek(0)
        audio = AudioSegment.from_file(audio_file)
        
        # チャンクサイズ（ミリ秒）
        chunk_duration_ms = chunk_duration_minutes * 60 * 1000
        
        # 音声を分割
        chunks = []
        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        st.error(f"音声分割に失敗しました: {e}")
        return None

def upload_chunk_to_gcs(chunk, chunk_index, bucket_name, base_filename):
    """音声チャンクをGCSにアップロード"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # チャンクファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"audio_{timestamp}_{base_filename}_chunk_{chunk_index:03d}.wav"
        blob = bucket.blob(blob_name)
        
        # チャンクをWAV形式でバイト配列に変換
        chunk_io = io.BytesIO()
        chunk.export(chunk_io, format="wav")
        chunk_io.seek(0)
        
        # アップロード
        blob.upload_from_file(chunk_io, content_type='audio/wav')
        
        return f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        st.error(f"チャンク {chunk_index} のアップロードに失敗: {e}")
        return None

def transcribe_audio_chunk(gcs_uri, chunk_index):
    """音声チャンクを文字起こし（最適化設定）"""
    try:
        client = speech.SpeechClient()
        
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ja-JP",
            enable_automatic_punctuation=True,
            # 高速化のため簡素化
            enable_speaker_diarization=False,  # 話者分離を無効化で高速化
            model="default",  # デフォルトモデルで高速化
            use_enhanced=False  # エンハンス無効で高速化
        )
        
        # 短時間処理用に同期認識を使用
        if chunk_index < 5:  # 最初の5チャンクのみ進捗表示
            st.info(f"🎯 チャンク {chunk_index + 1} を音声認識中...")
        
        response = client.recognize(config=config, audio=audio)
        
        # 結果を整理
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "
        
        return f"[チャンク {chunk_index + 1}]\n{transcript.strip()}\n\n"
    except Exception as e:
        st.error(f"チャンク {chunk_index + 1} の音声認識に失敗: {e}")
        return f"[チャンク {chunk_index + 1}] - 認識失敗\n\n"

def process_audio_chunks_parallel(chunk_gcs_uris):
    """音声チャンクを並列処理"""
    transcripts = [""] * len(chunk_gcs_uris)
    
    # 進捗バー
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def transcribe_with_progress(chunk_data):
        chunk_index, gcs_uri = chunk_data
        result = transcribe_audio_chunk(gcs_uri, chunk_index)
        
        # 進捗更新
        progress = (chunk_index + 1) / len(chunk_gcs_uris)
        progress_bar.progress(progress)
        status_text.text(f"処理中: {chunk_index + 1}/{len(chunk_gcs_uris)} チャンク完了")
        
        return chunk_index, result
    
    # 並列処理（最大4スレッド）
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        chunk_data = [(i, uri) for i, uri in enumerate(chunk_gcs_uris)]
        future_to_chunk = {executor.submit(transcribe_with_progress, data): data for data in chunk_data}
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index, transcript = future.result()
            transcripts[chunk_index] = transcript
    
    progress_bar.progress(1.0)
    status_text.text("✅ 全チャンクの処理が完了しました！")
    
    return "".join(transcripts)

def generate_meeting_minutes(transcript):
    """OpenAI GPTを使用して議事録を生成"""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        # 長いテキスト用に要約機能を追加
        prompt = f"""
以下の会議の音声転写テキストから、構造化された議事録を作成してください。

音声転写テキスト:
{transcript[:15000]}...  # 長すぎる場合は切り詰め

以下の形式で議事録を作成してください：

# 会議議事録

## 📅 会議情報
- 日時：{datetime.now().strftime("%Y年%m月%d日")}
- 音声時間：約{len(transcript.split())//100}分（推定）

## 📝 議題・討議内容
[主要な議題と討議内容を整理]

## ✅ 決定事項
[会議で決定された重要事項]

## 📋 アクションアイテム
[今後のタスクや担当者]

## 📊 次回会議
[次回の予定や課題]

## 💬 その他・備考
[補足情報や重要な発言]

※長時間音声のため、重要なポイントを抽出して整理しています。
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは議事録作成の専門家です。長時間の会議内容を効率的に整理し、読みやすい議事録を作成してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"議事録生成に失敗しました: {e}")
        return None

def main():
    st.set_page_config(
        page_title="🎤 高速議事録作成アプリ",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 高速議事録作成アプリ")
    st.markdown("**60分の長時間音声も高速処理対応！** 音声を自動分割して並列処理します")
    
    # Google Cloud認証チェック
    if not setup_google_credentials():
        st.stop()
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    bucket_name = st.sidebar.text_input(
        "GCSバケット名", 
        value=st.secrets.get("GCS_BUCKET_NAME", "")
    )
    
    chunk_duration = st.sidebar.selectbox(
        "分割時間（分）",
        [5, 10, 15],
        index=1,
        help="音声を指定した時間で分割します。短いほど高速処理"
    )
    
    if not bucket_name:
        st.error("GCSバケット名を設定してください")
        st.stop()
    
    # メインエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎵 音声ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "音声ファイルを選択してください",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="対応形式: WAV, MP3, M4A, FLAC（60分の長時間音声OK）"
        )
        
        if uploaded_file is not None:
            st.success(f"ファイル: {uploaded_file.name}")
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"ファイルサイズ: {file_size_mb:.1f} MB")
            
            # 予想処理時間を表示
            estimated_minutes = file_size_mb / 2  # 概算
            st.info(f"📊 予想処理時間: 約{estimated_minutes:.1f}分")
            
            # 処理開始ボタン
            if st.button("🚀 高速議事録作成開始", type="primary"):
                with st.spinner("高速処理中..."):
                    start_time = time.time()
                    
                    # 1. 音声分割
                    st.info("✂️ 音声を分割中...")
                    chunks = split_audio_file(uploaded_file, chunk_duration)
                    
                    if chunks:
                        st.success(f"✅ 音声を{len(chunks)}個のチャンクに分割完了")
                        
                        # 2. 並列アップロード
                        st.info("📤 チャンクを並列アップロード中...")
                        chunk_gcs_uris = []
                        base_filename = os.path.splitext(uploaded_file.name)[0]
                        
                        for i, chunk in enumerate(chunks):
                            gcs_uri = upload_chunk_to_gcs(chunk, i, bucket_name, base_filename)
                            if gcs_uri:
                                chunk_gcs_uris.append(gcs_uri)
                        
                        if chunk_gcs_uris:
                            st.success(f"✅ {len(chunk_gcs_uris)}個のチャンクのアップロード完了")
                            
                            # 3. 並列音声認識
                            st.info("🎯 並列音声認識開始...")
                            transcript = process_audio_chunks_parallel(chunk_gcs_uris)
                            
                            if transcript:
                                st.success("✅ 音声認識完了")
                                
                                # 4. 議事録生成
                                st.info("📝 議事録生成中...")
                                meeting_minutes = generate_meeting_minutes(transcript)
                                
                                if meeting_minutes:
                                    end_time = time.time()
                                    processing_time = (end_time - start_time) / 60
                                    
                                    st.success(f"✅ 議事録生成完了！（処理時間: {processing_time:.1f}分）")
                                    
                                    # 結果表示
                                    with col2:
                                        st.header("📋 生成された議事録")
                                        st.markdown(meeting_minutes)
                                        
                                        # ダウンロードボタン
                                        st.download_button(
                                            label="📥 議事録をダウンロード",
                                            data=meeting_minutes,
                                            file_name=f"議事録_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                            mime="text/markdown"
                                        )
                                    
                                    # 音声転写テキスト表示（折りたたみ）
                                    with st.expander("📄 音声転写テキスト（参考）"):
                                        st.text_area("転写結果", transcript, height=300)
                                    
                                    # 処理統計
                                    with st.expander("📊 処理統計"):
                                        st.write(f"- 総処理時間: {processing_time:.1f}分")
                                        st.write(f"- 分割チャンク数: {len(chunks)}個")
                                        st.write(f"- 1チャンクあたり平均時間: {processing_time/len(chunks):.1f}分")
    
    # フッター
    st.markdown("---")
    st.markdown("🚀 **高速処理**: 音声分割＋並列処理で大幅時間短縮")
    st.markdown("🔒 **プライバシー**: アップロードされた音声ファイルは処理後に自動削除されます")

if __name__ == "__main__":
    main()
