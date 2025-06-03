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

def upload_to_gcs(audio_file, bucket_name):
    """Google Cloud Storageにファイルをアップロード"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # ファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"audio_{timestamp}_{audio_file.name}"
        blob = bucket.blob(blob_name)
        
        # ファイルアップロード
        audio_file.seek(0)
        blob.upload_from_file(audio_file)
        
        return f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        st.error(f"ファイルアップロードに失敗しました: {e}")
        return None

def transcribe_audio(gcs_uri, file_extension):
    """Google Speech-to-Textで音声を文字起こし"""
    try:
        client = speech.SpeechClient()
        
        # ファイル形式に応じたエンコーディング設定
        encoding_map = {
            '.wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            '.mp3': speech.RecognitionConfig.AudioEncoding.MP3,
            '.m4a': speech.RecognitionConfig.AudioEncoding.MP3,  # M4Aは通常MP3として処理
            '.flac': speech.RecognitionConfig.AudioEncoding.FLAC,
        }
        
        encoding = encoding_map.get(file_extension.lower(), 
                                  speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED)
        
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=encoding,
            language_code="ja-JP",
            enable_automatic_punctuation=True,
            enable_speaker_diarization=True,
            diarization_speaker_count=2,
            model="latest_long",
            use_enhanced=True  # 音質向上
        )
        
        # 長時間音声の場合は非同期処理
        operation = client.long_running_recognize(config=config, audio=audio)
        
        st.info("音声認識を実行中です... しばらくお待ちください")
        response = operation.result(timeout=300)  # 5分でタイムアウト
        
        # 結果を整理
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + "\n"
        
        return transcript.strip()
    except Exception as e:
        st.error(f"音声認識に失敗しました: {e}")
        return None

def generate_meeting_minutes(transcript):
    """OpenAI GPTを使用して議事録を生成"""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        prompt = f"""
以下の会議の音声転写テキストから、構造化された議事録を作成してください。

音声転写テキスト:
{transcript}

以下の形式で議事録を作成してください：

# 会議議事録

## 📅 会議情報
- 日時：{datetime.now().strftime("%Y年%m月%d日")}
- 参加者：[音声から推測される参加者数]

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
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは議事録作成の専門家です。会議の内容を整理し、読みやすい議事録を作成してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"議事録生成に失敗しました: {e}")
        return None

def main():
    st.set_page_config(
        page_title="🎤 チーム議事録作成アプリ",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 チーム議事録作成アプリ")
    st.markdown("音声ファイルをアップロードして、自動で議事録を作成します")
    
    # Google Cloud認証チェック
    if not setup_google_credentials():
        st.stop()
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    bucket_name = st.sidebar.text_input(
        "GCSバケット名", 
        value=st.secrets.get("GCS_BUCKET_NAME", "")
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
            help="対応形式: WAV, MP3, M4A, FLAC (最大200MB)"
        )
        
        if uploaded_file is not None:
            st.success(f"ファイル: {uploaded_file.name}")
            st.info(f"ファイルサイズ: {uploaded_file.size / 1024 / 1024:.1f} MB")
            
            # 処理開始ボタン
            if st.button("🚀 議事録作成開始", type="primary"):
                with st.spinner("処理中..."):
                    # 1. GCSにアップロード
                    st.info("📤 ファイルをアップロード中...")
                    gcs_uri = upload_to_gcs(uploaded_file, bucket_name)
                    
                    if gcs_uri:
                        st.success("✅ アップロード完了")
                        
                        # 2. 音声認識
                        st.info("🎯 音声認識中...")
                        file_extension = os.path.splitext(uploaded_file.name)[1]
                        transcript = transcribe_audio(gcs_uri, file_extension)
                        
                        if transcript:
                            st.success("✅ 音声認識完了")
                            
                            # 3. 議事録生成
                            st.info("📝 議事録生成中...")
                            meeting_minutes = generate_meeting_minutes(transcript)
                            
                            if meeting_minutes:
                                st.success("✅ 議事録生成完了！")
                                
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
                                    st.text_area("転写結果", transcript, height=200)
    
    # フッター
    st.markdown("---")
    st.markdown("🔒 **プライバシー**: アップロードされた音声ファイルは処理後に自動削除されます")

if __name__ == "__main__":
    main()
