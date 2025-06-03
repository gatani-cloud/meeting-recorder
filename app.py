import streamlit as st
import os
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.cloud import firestore
import openai
from datetime import datetime, timedelta
import io
import re
import json
import tempfile
import time
import threading
import uuid
import hashlib

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

class AsyncJobManager:
    """非同期ジョブ管理システム"""
    
    def __init__(self):
        self.db = firestore.Client()
        self.jobs_collection = "meeting_recorder_jobs"
    
    def create_job(self, file_info, settings):
        """新しいジョブを作成"""
        job_id = str(uuid.uuid4())[:8]
        job_data = {
            'job_id': job_id,
            'status': 'created',
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'file_info': file_info,
            'settings': settings,
            'progress': 0,
            'current_step': 'waiting',
            'result': None,
            'error': None
        }
        
        self.db.collection(self.jobs_collection).document(job_id).set(job_data)
        return job_id
    
    def update_job_status(self, job_id, status, progress=None, current_step=None, result=None, error=None):
        """ジョブステータスを更新"""
        update_data = {
            'status': status,
            'updated_at': datetime.now()
        }
        
        if progress is not None:
            update_data['progress'] = progress
        if current_step:
            update_data['current_step'] = current_step
        if result:
            update_data['result'] = result
        if error:
            update_data['error'] = error
            
        self.db.collection(self.jobs_collection).document(job_id).update(update_data)
    
    def get_job_status(self, job_id):
        """ジョブステータスを取得"""
        doc = self.db.collection(self.jobs_collection).document(job_id).get()
        return doc.to_dict() if doc.exists else None
    
    def cleanup_old_jobs(self, days=7):
        """古いジョブを削除"""
        cutoff_date = datetime.now() - timedelta(days=days)
        old_jobs = self.db.collection(self.jobs_collection).where('created_at', '<', cutoff_date).get()
        for job in old_jobs:
            job.reference.delete()

def upload_to_gcs(audio_file, bucket_name):
    """Google Cloud Storageにファイルをアップロード"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"audio_{timestamp}_{audio_file.name}"
        blob = bucket.blob(blob_name)
        
        audio_file.seek(0)
        blob.upload_from_file(audio_file)
        
        return f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        st.error(f"ファイルアップロードに失敗しました: {e}")
        return None

def process_audio_async(job_id, gcs_uri, file_extension, settings):
    """非同期音声処理（バックエンドで実行）"""
    job_manager = AsyncJobManager()
    
    try:
        # ステップ1: 音声認識開始
        job_manager.update_job_status(job_id, 'processing', 10, '音声認識を開始')
        
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
        
        # 設定に応じた音声認識設定
        speed_mode = settings.get('speed_mode', 'balanced')
        if speed_mode == 'fast':
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="default",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=False,
                use_enhanced=False,
                max_alternatives=1
            )
        elif speed_mode == 'quality':
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="latest_long",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=True,
                diarization_speaker_count=2,
                use_enhanced=True
            )
        else:
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="default",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=False,
                use_enhanced=False
            )
        
        # 非同期音声認識開始
        operation = client.long_running_recognize(config=config, audio=audio)
        job_manager.update_job_status(job_id, 'processing', 20, '音声認識を実行中')
        
        # バックエンドで処理継続（ブラウザ不要）
        start_time = time.time()
        max_wait_time = 3600  # 1時間
        
        while not operation.done():
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                job_manager.update_job_status(job_id, 'failed', error='タイムアウト（1時間）')
                return
            
            # 進捗更新（バックエンドで自動実行）
            progress = min(20 + (elapsed_time / max_wait_time) * 60, 80)
            job_manager.update_job_status(job_id, 'processing', progress, '音声認識処理中')
            
            time.sleep(30)  # 30秒間隔でチェック
        
        # ステップ2: 結果取得
        response = operation.result()
        processing_time = (time.time() - start_time) / 60
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + "\n"
        
        job_manager.update_job_status(job_id, 'processing', 85, '議事録生成中')
        
        # ステップ3: 議事録生成
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        prompt = f"""
以下の会議音声転写テキストから実用的な議事録を作成してください。

音声転写テキスト:
{transcript[:8000]}...

以下の形式で議事録を作成してください：

# 🎤 会議議事録（自動生成）

## 📅 基本情報
- 生成日時：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- 処理時間：{processing_time:.1f}分
- 処理モード：{speed_mode}

## 📋 主要議題
[重要な議題を整理]

## ✅ 決定事項
[決定された重要事項]

## 📊 討議内容
[主要な討議内容]

## 🎯 アクションアイテム
[具体的なタスクと期限]

## 💡 重要な発言
[特に重要な発言]

## 📈 継続課題
[次回への持ち越し事項]
"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "議事録作成の専門家として、実用的な議事録を作成してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        meeting_minutes = response.choices[0].message.content
        
        # 結果保存
        result_data = {
            'transcript': transcript,
            'meeting_minutes': meeting_minutes,
            'processing_time': processing_time,
            'completed_at': datetime.now()
        }
        
        job_manager.update_job_status(job_id, 'completed', 100, '完了', result_data)
        
    except Exception as e:
        job_manager.update_job_status(job_id, 'failed', error=str(e))

def start_background_processing(job_id, gcs_uri, file_extension, settings):
    """バックエンド処理を開始"""
    thread = threading.Thread(
        target=process_audio_async,
        args=(job_id, gcs_uri, file_extension, settings),
        daemon=True
    )
    thread.start()

def main():
    st.set_page_config(
        page_title="🔄 非同期議事録アプリ",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 非同期議事録アプリ")
    st.markdown("**スリープしても大丈夫！** ブラウザを閉じても処理が継続される議事録作成システム")
    
    if not setup_google_credentials():
        st.stop()
    
    job_manager = AsyncJobManager()
    
    # セッション状態の初期化
    if 'current_job_id' not in st.session_state:
        st.session_state.current_job_id = None
    
    # サイドバー設定
    st.sidebar.header("⚙️ システム設定")
    bucket_name = st.sidebar.text_input(
        "GCSバケット名", 
        value=st.secrets.get("GCS_BUCKET_NAME", "")
    )
    
    speed_mode = st.sidebar.selectbox(
        "処理モード",
        ["balanced", "fast", "quality"],
        index=0
    )
    
    st.sidebar.markdown("""
    ### 🎯 非同期処理の特徴
    - **ブラウザ閉じてもOK**: 処理は継続
    - **スリープしてもOK**: バックエンドで実行
    - **進捗確認**: いつでも状況チェック
    - **自動復旧**: エラー時の再開機能
    """)
    
    if not bucket_name:
        st.error("GCSバケット名を設定してください")
        st.stop()
    
    # タブ構成
    tab1, tab2, tab3 = st.tabs(["🚀 新規処理", "📊 進捗確認", "📋 完了済み"])
    
    with tab1:
        st.header("🎵 音声ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "音声ファイルを選択してください",
            type=['wav', 'mp3', 'm4a', 'flac']
        )
        
        if uploaded_file is not None:
            st.success(f"ファイル: {uploaded_file.name}")
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"ファイルサイズ: {file_size_mb:.1f} MB")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 バックエンド処理開始", type="primary"):
                    with st.spinner("ファイルアップロード中..."):
                        # ファイルアップロード
                        gcs_uri = upload_to_gcs(uploaded_file, bucket_name)
                        
                        if gcs_uri:
                            # ジョブ作成
                            file_info = {
                                'name': uploaded_file.name,
                                'size': file_size_mb,
                                'gcs_uri': gcs_uri
                            }
                            settings = {
                                'speed_mode': speed_mode
                            }
                            
                            job_id = job_manager.create_job(file_info, settings)
                            st.session_state.current_job_id = job_id
                            
                            # バックエンド処理開始
                            file_extension = os.path.splitext(uploaded_file.name)[1]
                            start_background_processing(job_id, gcs_uri, file_extension, settings)
                            
                            st.success(f"✅ バックエンド処理を開始しました！")
                            st.info(f"🆔 ジョブID: **{job_id}**")
                            st.warning("💡 ブラウザを閉じても処理は継続されます。進捗確認タブで状況をチェックできます。")
            
            with col2:
                st.markdown("""
                **🔄 非同期処理の流れ**
                1. ファイルアップロード
                2. バックエンド処理開始
                3. ブラウザを閉じても継続
                4. 完了通知（進捗確認で確認）
                """)
    
    with tab2:
        st.header("📊 処理進捗確認")
        
        # 現在のジョブがある場合
        if st.session_state.current_job_id:
            job_id = st.session_state.current_job_id
            st.info(f"現在のジョブID: **{job_id}**")
            
            if st.button("🔄 最新状況を確認"):
                job_status = job_manager.get_job_status(job_id)
                
                if job_status:
                    st.json(job_status)
                    
                    # プログレスバー表示
                    if job_status['status'] == 'processing':
                        st.progress(job_status['progress'] / 100)
                        st.info(f"📍 {job_status['current_step']}")
                    elif job_status['status'] == 'completed':
                        st.success("🎉 処理完了！")
                        st.balloons()
                        
                        # 結果表示
                        result = job_status['result']
                        if result and 'meeting_minutes' in result:
                            st.markdown("### 📋 生成された議事録")
                            st.markdown(result['meeting_minutes'])
                            
                            # ダウンロード
                            st.download_button(
                                label="📥 議事録をダウンロード",
                                data=result['meeting_minutes'],
                                file_name=f"議事録_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                    elif job_status['status'] == 'failed':
                        st.error(f"❌ 処理失敗: {job_status.get('error', '不明なエラー')}")
                else:
                    st.warning("ジョブが見つかりません")
        
        # 手動ジョブID入力
        st.markdown("---")
        manual_job_id = st.text_input("🆔 ジョブIDを入力して確認", placeholder="例: abc12345")
        if manual_job_id and st.button("確認"):
            job_status = job_manager.get_job_status(manual_job_id)
            if job_status:
                st.json(job_status)
            else:
                st.error("ジョブが見つかりません")
    
    with tab3:
        st.header("📋 完了済みジョブ一覧")
        st.info("過去7日間の完了ジョブを表示（今後実装予定）")
    
    # フッター
    st.markdown("---")
    st.markdown("""
    ### 🎯 システムの特徴
    - **非同期処理**: ブラウザを閉じても処理継続
    - **進捗追跡**: いつでも処理状況を確認
    - **自動保存**: 結果は自動でクラウドに保存
    - **長時間対応**: 最大1時間の音声処理に対応
    """)

if __name__ == "__main__":
    main()
