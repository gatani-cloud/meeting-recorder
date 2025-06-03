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

def setup_google_credentials():
    """Google Cloud認証設定"""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(creds_dict, f)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
        return True
    except Exception as e:
        st.error(f"Google Cloud認証に失敗しました: {e}")
        return False

def split_audio_to_chunks(audio_file, chunk_size_mb=1):
    """音声ファイルを指定サイズのチャンクに分割（超小型）"""
    try:
        file_size_mb = audio_file.size / 1024 / 1024
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        
        chunks = []
        audio_file.seek(0)
        
        chunk_num = 0
        while True:
            chunk_data = audio_file.read(chunk_size_bytes)
            if not chunk_data:
                break
                
            chunk_num += 1
            chunks.append({
                'number': chunk_num,
                'data': chunk_data,
                'size_mb': len(chunk_data) / 1024 / 1024
            })
        
        return chunks
    except Exception as e:
        st.error(f"音声分割に失敗しました: {e}")
        return None

def upload_chunk_to_gcs(chunk_data, chunk_number, original_filename, bucket_name):
    """音声チャンクをGCSにアップロード"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(original_filename)[1]
        blob_name = f"chunk_{timestamp}_{chunk_number:03d}_{original_filename}"
        blob = bucket.blob(blob_name)
        
        blob.upload_from_string(chunk_data)
        
        return f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        st.error(f"チャンク {chunk_number} のアップロードに失敗しました: {e}")
        return None

def transcribe_chunk(gcs_uri, file_extension, chunk_number, speed_mode):
    """単一チャンクの音声認識（小さなチャンク用）"""
    try:
        client = speech.SpeechClient()
        
        encoding_map = {
            '.wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            '.mp3': speech.RecognitionConfig.AudioEncoding.MP3,
            '.m4a': speech.RecognitionConfig.AudioEncoding.MP3,
            '.flac': speech.RecognitionConfig.AudioEncoding.FLAC,
        }
        
        encoding = encoding_map.get(file_extension.lower(), 
                                  speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED)
        
        audio = speech.RecognitionAudio(uri=gcs_uri)
        
        # 速度モードに応じた設定（簡素化）
        config = speech.RecognitionConfig(
            encoding=encoding,
            language_code="ja-JP",
            model="default",
            enable_automatic_punctuation=True,
            enable_speaker_diarization=False,  # 小さなチャンクでは無効
            use_enhanced=False,  # 高速化のため無効
            max_alternatives=1
        )
        
        # 小さなチャンク（3MB以下）なので同期認識を試行
        try:
            response = client.recognize(config=config, audio=audio)
        except Exception as sync_error:
            # 同期認識が失敗した場合は非同期認識にフォールバック
            st.warning(f"チャンク {chunk_number}: 非同期認識に切り替え")
            operation = client.long_running_recognize(config=config, audio=audio)
            
            # 非同期認識の結果待機（最大5分）
            start_time = time.time()
            while not operation.done():
                if time.time() - start_time > 300:  # 5分でタイムアウト
                    return f"[チャンク {chunk_number}: タイムアウト]"
                time.sleep(10)
            
            response = operation.result()
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "
        
        return transcript.strip() if transcript.strip() else f"[チャンク {chunk_number}: 音声なし]"
    except Exception as e:
        st.error(f"チャンク {chunk_number} の音声認識に失敗: {e}")
        return f"[チャンク {chunk_number}: 認識失敗 - {str(e)[:50]}]"

def process_chunks_sequentially(chunks, original_filename, bucket_name, speed_mode):
    """チャンクを順次処理（スリープ対応）"""
    try:
        file_extension = os.path.splitext(original_filename)[1]
        transcripts = []
        
        st.info(f"🔄 {len(chunks)}個のチャンクを順次処理します")
        
        # 進捗管理
        progress_bar = st.progress(0)
        status_container = st.empty()
        transcript_container = st.empty()
        
        total_transcript = ""
        
        for i, chunk in enumerate(chunks):
            # 進捗更新
            progress = (i) / len(chunks)
            progress_bar.progress(progress)
            status_container.info(f"📍 チャンク {chunk['number']}/{len(chunks)} を処理中... ({chunk['size_mb']:.1f}MB)")
            
            # チャンクアップロード
            gcs_uri = upload_chunk_to_gcs(chunk['data'], chunk['number'], original_filename, bucket_name)
            
            if gcs_uri:
                # 音声認識実行
                transcript = transcribe_chunk(gcs_uri, file_extension, chunk['number'], speed_mode)
                transcripts.append(f"[チャンク {chunk['number']}]\n{transcript}\n")
                
                # リアルタイム結果表示
                total_transcript += transcript + " "
                transcript_container.text_area(
                    f"📝 処理済みテキスト（チャンク {chunk['number']}まで）", 
                    total_transcript[:1000] + "..." if len(total_transcript) > 1000 else total_transcript,
                    height=100
                )
                
                # 短時間休憩（システム負荷軽減）
                time.sleep(2)
            else:
                transcripts.append(f"[チャンク {chunk['number']}: アップロード失敗]\n")
        
        # 完了
        progress_bar.progress(1.0)
        status_container.success("✅ 全チャンクの処理が完了しました！")
        
        return "\n".join(transcripts)
    except Exception as e:
        st.error(f"チャンク処理でエラー: {e}")
        return None

def generate_meeting_minutes(transcript, processing_time, speed_mode):
    """議事録生成（APIキーチェック付き）"""
    try:
        # OpenAI APIキーの存在確認
        if "OPENAI_API_KEY" not in st.secrets:
            st.warning("⚠️ OpenAI APIキーが設定されていません。音声転写結果のみ表示します。")
            return f"""
# 🎤 音声転写結果

## 📅 基本情報
- 生成日時：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- 処理時間：{processing_time:.1f}分
- 処理方式：チャンク分割処理
- 品質モード：{speed_mode}

## 📄 音声転写テキスト
{transcript}

---
※OpenAI APIキーが設定されていないため、議事録の自動生成ができませんでした。
上記の転写テキストを元に手動で議事録を作成してください。

### 🔧 OpenAI APIキー設定方法
1. https://platform.openai.com でAPIキーを取得
2. Streamlit CloudのSecrets設定でOPENAI_API_KEYを追加
"""
        
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        # 長いテキストの処理
        max_length = 12000
        if len(transcript) > max_length:
            # 均等に抽出
            parts = []
            part_size = max_length // 3
            parts.append(transcript[:part_size])
            parts.append(transcript[len(transcript)//2:len(transcript)//2 + part_size])
            parts.append(transcript[-part_size:])
            transcript_sample = "\n\n[--- 中間部分 ---]\n\n".join(parts)
            note = "※長時間音声のため、重要部分を抽出して議事録を作成しています。"
        else:
            transcript_sample = transcript
            note = ""
        
        prompt = f"""
以下の会議音声転写テキストから、実用的な議事録を作成してください。

音声転写テキスト:
{transcript_sample}

以下の形式で議事録を作成してください：

# 🎤 会議議事録（分割処理版）

## 📅 基本情報
- 生成日時：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- 処理時間：{processing_time:.1f}分
- 処理方式：チャンク分割処理
- 品質モード：{speed_mode}

## 📋 主要議題
[重要な議題を3-5点で整理]

## ✅ 決定事項
[決定された重要事項を優先度順に]

## 📊 討議内容
[主要な討議内容と参加者の意見]

## 🎯 アクションアイテム
[具体的なタスク、担当者、期限]

## 💡 重要な発言・提案
[特に重要な発言やアイデア]

## 📈 継続検討事項
[次回会議への持ち越し課題]

## 📝 備考
{note}

---
※この議事録は音声を分割処理して生成されました。各チャンクの音声認識結果を統合しています。
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは議事録作成の専門家です。分割処理された音声認識結果から、統合された実用的な議事録を作成してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"議事録生成に失敗しました: {e}")
        return f"""
# 🎤 音声転写結果（議事録生成失敗）

## 📅 基本情報
- 生成日時：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- 処理時間：{processing_time:.1f}分
- エラー：{str(e)}

## 📄 音声転写テキスト
{transcript}

---
※議事録の自動生成に失敗しました。上記の転写テキストを元に手動で議事録を作成してください。
"""

def main():
    st.set_page_config(
        page_title="🔄 分割処理議事録アプリ",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 分割処理議事録アプリ")
    st.markdown("**スリープ対応！** 音声を小さなチャンクに分割して確実に処理する議事録システム")
    
    if not setup_google_credentials():
        st.stop()
    
    # サイドバー設定
    st.sidebar.header("⚙️ 分割処理設定")
    bucket_name = st.sidebar.text_input(
        "GCSバケット名", 
        value=st.secrets.get("GCS_BUCKET_NAME", "")
    )
    
    chunk_size = st.sidebar.selectbox(
        "チャンクサイズ",
        [1, 2, 3],
        index=0,
        help="音声ファイルを指定サイズ(MB)で分割（1MBが最も確実）"
    )
    
    speed_mode = st.sidebar.selectbox(
        "処理品質",
        ["balanced", "fast", "quality"],
        index=0
    )
    
    st.sidebar.markdown(f"""
    ### 🔄 分割処理の特徴
    - **確実性**: 小さなチャンクで確実処理
    - **進捗表示**: リアルタイム結果確認
    - **中断耐性**: 途中で止まっても部分結果あり
    - **スリープ対応**: 処理中にPCスリープ可能
    
    ### ⚙️ 現在の設定
    - チャンクサイズ: {chunk_size}MB
    - 処理品質: {speed_mode}
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
            help="推奨: WAV/MP3形式、100MB以下"
        )
        
        if uploaded_file is not None:
            st.success(f"📁 ファイル: {uploaded_file.name}")
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"📊 ファイルサイズ: {file_size_mb:.1f} MB")
            
            # 分割予測
            estimated_chunks = math.ceil(file_size_mb / chunk_size)
            estimated_time = estimated_chunks * 2  # チャンクあたり約2分
            
            st.info(f"🔄 予想分割数: {estimated_chunks}チャンク")
            st.info(f"⏱️ 予想処理時間: 約{estimated_time}分")
            
            # 処理開始ボタン
            if st.button("🚀 分割処理開始", type="primary"):
                start_time = time.time()
                
                # Step 1: 音声分割
                st.info("✂️ 音声ファイルを分割中...")
                chunks = split_audio_to_chunks(uploaded_file, chunk_size)
                
                if chunks:
                    st.success(f"✅ {len(chunks)}個のチャンクに分割完了")
                    
                    # Step 2: 順次処理
                    transcript = process_chunks_sequentially(chunks, uploaded_file.name, bucket_name, speed_mode)
                    
                    if transcript:
                        processing_time = (time.time() - start_time) / 60
                        st.success(f"✅ 音声認識完了！（{processing_time:.1f}分）")
                        
                        # Step 3: 議事録生成
                        st.info("📝 議事録生成中...")
                        meeting_minutes = generate_meeting_minutes(transcript, processing_time, speed_mode)
                        
                        if meeting_minutes:
                            st.success("🎉 議事録生成完了！")
                            
                            # 結果表示
                            with col2:
                                st.header("📋 生成された議事録")
                                st.markdown(meeting_minutes)
                                
                                # ダウンロードボタン
                                st.download_button(
                                    label="📥 議事録をダウンロード",
                                    data=meeting_minutes,
                                    file_name=f"分割処理議事録_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
                            
                            # 詳細結果
                            with st.expander("📊 処理統計"):
                                total_time = (time.time() - start_time) / 60
                                st.write(f"- **総処理時間**: {total_time:.1f}分")
                                st.write(f"- **チャンク数**: {len(chunks)}個")
                                st.write(f"- **平均チャンク処理時間**: {total_time/len(chunks):.1f}分")
                                st.write(f"- **転写文字数**: {len(transcript):,}文字")
                            
                            # 全文表示
                            with st.expander("📄 音声転写テキスト（全文）"):
                                st.text_area("転写結果", transcript, height=400)
    
    with col2:
        if 'uploaded_file' not in locals() or uploaded_file is None:
            st.header("💡 分割処理の仕組み")
            st.markdown("""
            ### 🔄 処理フロー
            1. **音声分割**: ファイルを小さなチャンクに分割
            2. **順次処理**: 各チャンクを個別に音声認識
            3. **結果統合**: 全チャンクの結果を統合
            4. **議事録生成**: 統合結果から議事録作成
            
            ### ✅ メリット
            - **確実性**: 小さなファイルで処理失敗リスク軽減
            - **進捗確認**: リアルタイムで処理状況表示
            - **中断耐性**: 途中で止まっても部分結果保持
            - **スリープ対応**: 処理中のPCスリープ可能
            
            ### ⚙️ 推奨設定
            - **ファイル形式**: WAV > MP3 > M4A
            - **チャンクサイズ**: 10MB（バランス重視）
            - **処理品質**: balanced（推奨）
            """)
    
    # フッター
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔄 分割処理**")
        st.markdown("音声を小分けして確実に処理")
    
    with col2:
        st.markdown("**📊 進捗表示**")
        st.markdown("リアルタイムで結果確認")
    
    with col3:
        st.markdown("**💤 スリープ対応**")
        st.markdown("処理中にPCスリープ可能")

if __name__ == "__main__":
    main()
