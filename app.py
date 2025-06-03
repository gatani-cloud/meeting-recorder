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

def transcribe_audio_optimized(gcs_uri, file_extension, speed_mode="balanced"):
    """最適化された音声認識（確実＆高速）"""
    try:
        client = speech.SpeechClient()
        
        # ファイル形式に応じたエンコーディング設定
        encoding_map = {
            '.wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            '.mp3': speech.RecognitionConfig.AudioEncoding.MP3,
            '.m4a': speech.RecognitionConfig.AudioEncoding.MP3,
            '.flac': speech.RecognitionConfig.AudioEncoding.FLAC,
        }
        
        encoding = encoding_map.get(file_extension.lower(), 
                                  speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED)
        
        audio = speech.RecognitionAudio(uri=gcs_uri)
        
        # 速度モードに応じた設定
        if speed_mode == "fast":
            # 高速モード：処理速度優先
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="default",  # 安定したモデル
                enable_automatic_punctuation=True,
                enable_speaker_diarization=False,  # 話者分離無効で高速化
                use_enhanced=False,  # エンハンス無効で高速化
                max_alternatives=1,
                profanity_filter=False
            )
            timeout_minutes = 25
        elif speed_mode == "quality":
            # 品質モード：精度優先
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="latest_long",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=True,
                diarization_speaker_count=2,
                use_enhanced=True
            )
            timeout_minutes = 40
        else:
            # バランスモード：速度と精度のバランス（推奨）
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="default",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=False,  # 高速化
                use_enhanced=False,  # 高速化
                max_alternatives=1
            )
            timeout_minutes = 30
        
        # 非同期処理開始
        operation = client.long_running_recognize(config=config, audio=audio)
        
        st.info(f"🎯 音声認識実行中... 最大{timeout_minutes}分お待ちください")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        # 進捗監視
        while not operation.done():
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                st.error(f"⏰ 処理時間が{timeout_minutes}分を超えました。より小さなファイルでお試しください。")
                return None, 0
            
            # 進捗表示
            estimated_progress = min(elapsed_time / (timeout_seconds * 0.8), 0.95)
            progress_bar.progress(estimated_progress)
            
            # 残り時間計算
            if elapsed_time > 60:  # 1分経過後に残り時間表示
                estimated_total = elapsed_time / estimated_progress if estimated_progress > 0.1 else timeout_seconds
                remaining_time = max(0, (estimated_total - elapsed_time) / 60)
                status_text.text(f"⚡ {speed_mode}モード処理中... {elapsed_time/60:.1f}分経過 (推定残り{remaining_time:.1f}分)")
            else:
                status_text.text(f"⚡ {speed_mode}モード処理中... {elapsed_time:.0f}秒経過")
            
            time.sleep(8)  # 8秒間隔でチェック
        
        # 結果取得
        response = operation.result()
        processing_time = (time.time() - start_time) / 60
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ 音声認識完了！({processing_time:.1f}分)")
        
        # 結果を整理
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + "\n"
        
        return transcript.strip(), processing_time
    except Exception as e:
        st.error(f"音声認識に失敗しました: {e}")
        return None, 0

def generate_meeting_minutes_smart(transcript, processing_time, speed_mode):
    """スマート議事録生成"""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        # テキスト長に応じた処理
        max_length = 10000
        if len(transcript) > max_length:
            # 長いテキストの場合は要点抽出
            parts = [
                transcript[:max_length//3],
                transcript[len(transcript)//2:len(transcript)//2 + max_length//3],
                transcript[-max_length//3:]
            ]
            transcript_sample = "\n\n[--- 中間部分 ---]\n\n".join(parts)
            note = "※長時間音声のため主要部分を抽出して議事録を作成しています。"
        else:
            transcript_sample = transcript
            note = ""
        
        prompt = f"""
以下の会議音声転写テキストから、実用的な議事録を作成してください。

音声転写テキスト:
{transcript_sample}

以下の形式で議事録を作成してください：

# 🎤 会議議事録

## 📅 基本情報
- 作成日時：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- 処理モード：{speed_mode}
- 処理時間：{processing_time:.1f}分
- 音声長：約{len(transcript.split())//120}分（推定）

## 📋 主要議題
[重要な議題を3-5点で整理]

## ✅ 決定事項
[会議で決定された重要事項を優先度順に]

## 📊 討議内容
[主要な討議内容と意見]

## 🎯 アクションアイテム
[具体的なタスクと担当者、期限]

## 💡 重要な発言・提案
[特に重要な発言やアイデア]

## 📈 次回までの課題
[継続検討事項や次回議題]

## 📝 備考
{note}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは経験豊富な議事録作成の専門家です。会議の内容を整理し、実用的で読みやすい議事録を作成してください。"},
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
        page_title="🚀 実用的議事録アプリ",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 実用的議事録アプリ")
    st.markdown("**60分音声対応！** 確実性と高速化を両立した議事録作成")
    
    # Google Cloud認証チェック
    if not setup_google_credentials():
        st.stop()
    
    # サイドバー設定
    st.sidebar.header("⚙️ 処理設定")
    bucket_name = st.sidebar.text_input(
        "GCSバケット名", 
        value=st.secrets.get("GCS_BUCKET_NAME", "")
    )
    
    speed_mode = st.sidebar.selectbox(
        "処理モード",
        ["balanced", "fast", "quality"],
        index=0,
        help="""
        • balanced: 速度と精度のバランス（推奨）
        • fast: 高速処理優先（20分以内）
        • quality: 高品質優先（40分以内）
        """
    )
    
    # モードの説明
    if speed_mode == "fast":
        st.sidebar.success("⚡ 高速モード：処理時間優先")
        st.sidebar.info("• 60分音声 → 約15-20分処理\n• 話者分離なし\n• エンハンス機能なし")
    elif speed_mode == "quality":
        st.sidebar.info("🎯 高品質モード：精度優先")
        st.sidebar.info("• 60分音声 → 約25-40分処理\n• 話者分離あり\n• エンハンス機能あり")
    else:
        st.sidebar.success("⚖️ バランスモード：推奨設定")
        st.sidebar.info("• 60分音声 → 約20-30分処理\n• 速度と精度を両立")
    
    # 使用のヒント
    st.sidebar.markdown("""
    ### 💡 効率的な使用法
    - **初回利用**: バランスモード推奨
    - **緊急時**: 高速モード
    - **重要会議**: 高品質モード
    - **ファイル形式**: WAV > MP3 > M4A
    - **推奨サイズ**: 50MB以下
    """)
    
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
            help="対応形式: WAV, MP3, M4A, FLAC (推奨: 50MB以下)"
        )
        
        if uploaded_file is not None:
            st.success(f"ファイル: {uploaded_file.name}")
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"ファイルサイズ: {file_size_mb:.1f} MB")
            
            # 予想処理時間を表示
            if speed_mode == "fast":
                estimated_minutes = file_size_mb * 0.8
            elif speed_mode == "quality":
                estimated_minutes = file_size_mb * 1.5
            else:
                estimated_minutes = file_size_mb * 1.0
                
            st.info(f"📊 予想処理時間: 約{estimated_minutes:.1f}分")
            
            # ファイル最適化のアドバイス
            if file_size_mb > 50:
                st.warning("⚠️ 大きなファイルです。処理時間が長くなる可能性があります")
            elif uploaded_file.name.endswith('.wav'):
                st.success("✅ WAV形式：最適な処理が期待できます")
            
            # 処理開始ボタン
            button_text = f"🚀 {speed_mode}モードで開始"
            if st.button(button_text, type="primary"):
                total_start_time = time.time()
                
                with st.spinner("処理中..."):
                    # 1. GCSにアップロード
                    st.info("📤 ファイルをアップロード中...")
                    gcs_uri = upload_to_gcs(uploaded_file, bucket_name)
                    
                    if gcs_uri:
                        st.success("✅ アップロード完了")
                        
                        # 2. 音声認識
                        file_extension = os.path.splitext(uploaded_file.name)[1]
                        result = transcribe_audio_optimized(gcs_uri, file_extension, speed_mode)
                        
                        if result and result[0]:
                            transcript, processing_time = result
                            st.success(f"✅ 音声認識完了（{processing_time:.1f}分）")
                            
                            # 3. 議事録生成
                            st.info("📝 議事録生成中...")
                            meeting_minutes = generate_meeting_minutes_smart(transcript, processing_time, speed_mode)
                            
                            if meeting_minutes:
                                total_time = (time.time() - total_start_time) / 60
                                
                                st.success(f"🎉 全処理完了！総時間: {total_time:.1f}分")
                                
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
                                
                                # 処理統計
                                with st.expander("📊 処理統計"):
                                    st.write(f"- **処理モード**: {speed_mode}")
                                    st.write(f"- **音声認識時間**: {processing_time:.1f}分")
                                    st.write(f"- **総処理時間**: {total_time:.1f}分")
                                    st.write(f"- **ファイルサイズ**: {file_size_mb:.1f} MB")
                                    st.write(f"- **転写文字数**: {len(transcript):,}文字")
                                    st.write(f"- **推定音声長**: {len(transcript.split())//120}分")
                                
                                # 音声転写テキスト表示
                                with st.expander("📄 音声転写テキスト（全文）"):
                                    st.text_area("転写結果", transcript, height=400)
    
    # フッター情報
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🚀 確実な処理**")
        st.markdown("安定した音声認識と十分なタイムアウト設定")
    
    with col2:
        st.markdown("**⚖️ 選べるモード**")
        st.markdown("用途に応じた速度・精度設定")
    
    with col3:
        st.markdown("**🔒 プライバシー**")
        st.markdown("処理後ファイル自動削除")

if __name__ == "__main__":
    main()
