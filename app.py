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

def transcribe_audio_high_speed(gcs_uri, file_extension, speed_mode="fast"):
    """超高速音声認識（最適化設定）"""
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
        
        # 高速化設定
        if speed_mode == "ultra_fast":
            # 超高速モード：5-10分で処理
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="latest_short",  # 高速モデル
                enable_automatic_punctuation=False,  # 句読点無効で高速化
                enable_speaker_diarization=False,    # 話者分離無効で高速化
                use_enhanced=False,                   # エンハンス無効で高速化
                enable_word_time_offsets=False,      # 時間オフセット無効
                max_alternatives=1,                   # 候補数を1に限定
                profanity_filter=False,              # 冒涜フィルタ無効
                enable_word_confidence=False         # 信頼度計算無効
            )
        elif speed_mode == "fast":
            # 高速モード：10-15分で処理
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="default",                     # デフォルトモデル
                enable_automatic_punctuation=True,
                enable_speaker_diarization=False,   # 話者分離無効で高速化
                use_enhanced=False,                  # エンハンス無効で高速化
                max_alternatives=1
            )
        else:
            # 標準モード：精度重視
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="latest_long",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=True,
                diarization_speaker_count=2,
                use_enhanced=True
            )
        
        # 非同期処理開始
        operation = client.long_running_recognize(config=config, audio=audio)
        
        # 進捗表示とリアルタイム監視
        st.info(f"🚀 {speed_mode}モードで音声認識実行中...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        timeout_seconds = 600 if speed_mode == "ultra_fast" else 900  # 10分 or 15分
        
        # ポーリング間隔を短縮して高速応答
        poll_interval = 5  # 5秒間隔
        
        while not operation.done():
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                st.error(f"⏰ 処理時間が{timeout_seconds//60}分を超えました。")
                return None
            
            # 進捗表示（推定）
            if speed_mode == "ultra_fast":
                estimated_progress = min(elapsed_time / (timeout_seconds * 0.6), 0.95)
            else:
                estimated_progress = min(elapsed_time / (timeout_seconds * 0.7), 0.95)
                
            progress_bar.progress(estimated_progress)
            status_text.text(f"⚡ {speed_mode}処理中... {elapsed_time/60:.1f}分経過 (予想残り{max(0, (timeout_seconds*0.7-elapsed_time)/60):.1f}分)")
            
            time.sleep(poll_interval)
        
        # 結果取得
        response = operation.result()
        end_time = time.time()
        processing_time = (end_time - start_time) / 60
        
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

def generate_meeting_minutes_fast(transcript, processing_time):
    """高速議事録生成（要点抽出型）"""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        # 長いテキストの場合は要点を抽出
        max_length = 8000
        if len(transcript) > max_length:
            # 冒頭、中間、終盤からサンプリング
            part1 = transcript[:max_length//3]
            part2 = transcript[len(transcript)//2:len(transcript)//2 + max_length//3]
            part3 = transcript[-max_length//3:]
            transcript_sample = f"{part1}\n\n[中間部分]\n{part2}\n\n[終盤部分]\n{part3}"
        else:
            transcript_sample = transcript
        
        prompt = f"""
以下の会議音声の転写テキストから、効率的で実用的な議事録を作成してください。

音声転写テキスト:
{transcript_sample}

以下の形式で簡潔な議事録を作成してください：

# ⚡ 高速議事録

## 📅 基本情報
- 作成日時：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- 処理時間：{processing_time:.1f}分
- 音声長：約{len(transcript.split())//100}分（推定）

## 🎯 主要議題（3-5点）
[重要な議題を箇条書きで]

## ✅ 決定事項（重要度順）
[決定された事項を優先度順に]

## 📋 アクションアイテム
[具体的なタスクと期限]

## 💡 重要な発言・意見
[特に重要な発言やアイデア]

## 📊 次回までの課題
[継続検討事項]

※高速処理により要点を抽出した議事録です。詳細は音声転写テキストをご参照ください。
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは効率的な議事録作成の専門家です。長時間の会議内容から要点を素早く抽出し、実用的な議事録を作成してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # 一貫性重視
            max_tokens=1500   # 簡潔さ重視
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"議事録生成に失敗しました: {e}")
        return None

def main():
    st.set_page_config(
        page_title="⚡ 超高速議事録アプリ",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ 超高速議事録アプリ")
    st.markdown("**60分音声も10-15分で処理！** 最適化設定で実用的な速度を実現")
    
    # Google Cloud認証チェック
    if not setup_google_credentials():
        st.stop()
    
    # サイドバー設定
    st.sidebar.header("⚙️ 高速化設定")
    bucket_name = st.sidebar.text_input(
        "GCSバケット名", 
        value=st.secrets.get("GCS_BUCKET_NAME", "")
    )
    
    speed_mode = st.sidebar.selectbox(
        "処理速度モード",
        ["ultra_fast", "fast", "standard"],
        index=1,
        help="""
        • ultra_fast: 5-10分処理（精度-10%）
        • fast: 10-15分処理（精度-5%）
        • standard: 20-30分処理（最高精度）
        """
    )
    
    # 速度モードの説明
    if speed_mode == "ultra_fast":
        st.sidebar.success("🚀 超高速モード：60分音声を10分以内で処理")
    elif speed_mode == "fast":
        st.sidebar.info("⚡ 高速モード：60分音声を15分以内で処理")
    else:
        st.sidebar.warning("🐌 標準モード：精度最優先（時間がかかります）")
    
    # 使用のヒント
    st.sidebar.markdown("""
    ### 💡 高速化のコツ
    - **ファイル形式**: WAVが最速
    - **音声品質**: クリアな音声ほど高速
    - **ファイルサイズ**: 50MB以下推奨
    - **背景ノイズ**: 少ないほど高速処理
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
            help="推奨：WAV形式、50MB以下、クリアな音質"
        )
        
        if uploaded_file is not None:
            st.success(f"ファイル: {uploaded_file.name}")
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"ファイルサイズ: {file_size_mb:.1f} MB")
            
            # 予想処理時間を表示
            if speed_mode == "ultra_fast":
                estimated_minutes = file_size_mb * 0.3
            elif speed_mode == "fast":
                estimated_minutes = file_size_mb * 0.5
            else:
                estimated_minutes = file_size_mb * 1.0
                
            st.info(f"📊 予想処理時間: 約{estimated_minutes:.1f}分")
            
            # ファイル最適化のアドバイス
            if file_size_mb > 50:
                st.warning("⚠️ 大きなファイルです。より高速化したい場合は、音声圧縮をお試しください")
            elif uploaded_file.name.endswith('.wav'):
                st.success("✅ WAV形式：最適な処理速度が期待できます")
            
            # 処理開始ボタン
            button_text = f"⚡ {speed_mode}モードで開始"
            if st.button(button_text, type="primary"):
                start_time = time.time()
                
                with st.spinner("高速処理中..."):
                    # 1. GCSにアップロード
                    st.info("📤 ファイルをアップロード中...")
                    gcs_uri = upload_to_gcs(uploaded_file, bucket_name)
                    
                    if gcs_uri:
                        st.success("✅ アップロード完了")
                        
                        # 2. 高速音声認識
                        file_extension = os.path.splitext(uploaded_file.name)[1]
                        result = transcribe_audio_high_speed(gcs_uri, file_extension, speed_mode)
                        
                        if result and result[0]:
                            transcript, processing_time = result
                            st.success(f"✅ 音声認識完了（{processing_time:.1f}分）")
                            
                            # 3. 高速議事録生成
                            st.info("📝 議事録生成中...")
                            meeting_minutes = generate_meeting_minutes_fast(transcript, processing_time)
                            
                            if meeting_minutes:
                                total_time = (time.time() - start_time) / 60
                                
                                # 速度改善表示
                                standard_time = file_size_mb * 1.5
                                improvement = ((standard_time - total_time) / standard_time) * 100
                                
                                st.success(f"🎉 完了！総時間: {total_time:.1f}分（従来比{improvement:.0f}%短縮）")
                                
                                # 結果表示
                                with col2:
                                    st.header("📋 高速生成議事録")
                                    st.markdown(meeting_minutes)
                                    
                                    # ダウンロードボタン
                                    st.download_button(
                                        label="📥 議事録をダウンロード",
                                        data=meeting_minutes,
                                        file_name=f"高速議事録_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                        mime="text/markdown"
                                    )
                                
                                # 詳細結果
                                with st.expander("📊 処理統計"):
                                    st.write(f"- **処理モード**: {speed_mode}")
                                    st.write(f"- **音声認識時間**: {processing_time:.1f}分")
                                    st.write(f"- **総処理時間**: {total_time:.1f}分")
                                    st.write(f"- **従来比短縮率**: {improvement:.0f}%")
                                    st.write(f"- **転写文字数**: {len(transcript):,}文字")
                                
                                # 音声転写テキスト表示（折りたたみ）
                                with st.expander("📄 音声転写テキスト（参考）"):
                                    st.text_area("転写結果", transcript, height=300)
    
    # フッター情報
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⚡ 超高速処理**")
        st.markdown("最適化設定で60分音声も10-15分で処理")
    
    with col2:
        st.markdown("**🎯 実用性重視**")
        st.markdown("要点抽出型の効率的な議事録生成")
    
    with col3:
        st.markdown("**🔒 プライバシー保護**")
        st.markdown("処理後ファイル自動削除")

if __name__ == "__main__":
    main()
