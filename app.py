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
import uuid

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

class SimpleJobManager:
    """シンプルなジョブ管理（GCS保存）"""
    
    def __init__(self, bucket_name):
        self.storage_client = storage.Client()
        self.bucket_name = bucket_name
        self.jobs_prefix = "job_status/"
    
    def create_job(self, file_info, settings):
        """新しいジョブを作成"""
        job_id = str(uuid.uuid4())[:8]
        job_data = {
            'job_id': job_id,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'file_info': file_info,
            'settings': settings,
            'progress': 0,
            'current_step': 'waiting',
            'result': None,
            'error': None
        }
        
        self._save_job_data(job_id, job_data)
        return job_id
    
    def update_job_status(self, job_id, status, progress=None, current_step=None, result=None, error=None):
        """ジョブステータスを更新"""
        try:
            job_data = self._load_job_data(job_id)
            if not job_data:
                return False
                
            job_data['status'] = status
            job_data['updated_at'] = datetime.now().isoformat()
            
            if progress is not None:
                job_data['progress'] = progress
            if current_step:
                job_data['current_step'] = current_step
            if result:
                job_data['result'] = result
            if error:
                job_data['error'] = error
                
            self._save_job_data(job_id, job_data)
            return True
        except Exception as e:
            print(f"ステータス更新エラー: {e}")
            return False
    
    def get_job_status(self, job_id):
        """ジョブステータスを取得"""
        return self._load_job_data(job_id)
    
    def _save_job_data(self, job_id, job_data):
        """ジョブデータをGCSに保存"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob_name = f"{self.jobs_prefix}{job_id}.json"
            blob = bucket.blob(blob_name)
            
            json_data = json.dumps(job_data, ensure_ascii=False, indent=2)
            blob.upload_from_string(json_data, content_type='application/json')
        except Exception as e:
            print(f"ジョブデータ保存エラー: {e}")
    
    def _load_job_data(self, job_id):
        """ジョブデータをGCSから読み込み"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob_name = f"{self.jobs_prefix}{job_id}.json"
            blob = bucket.blob(blob_name)
            
            if blob.exists():
                json_data = blob.download_as_text()
                return json.loads(json_data)
            return None
        except Exception as e:
            print(f"ジョブデータ読み込みエラー: {e}")
            return None

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

def process_audio_async(job_id, gcs_uri, file_extension, settings, bucket_name):
    """非同期音声処理（バックエンドで実行）"""
    job_manager = SimpleJobManager(bucket_name)
    
    try:
        # ステップ1: 音声認識開始
        job_manager.update_job_status(job_id, 'processing', 10, '音声認識を開始中...')
        
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
            max_wait_time = 1500  # 25分
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
            max_wait_time = 2400  # 40分
        else:
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="ja-JP",
                model="default",
                enable_automatic_punctuation=True,
                enable_speaker_diarization=False,
                use_enhanced=False
            )
            max_wait_time = 1800  # 30分
        
        # 非同期音声認識開始
        job_manager.update_job_status(job_id, 'processing', 20, '音声認識API呼び出し中...')
        operation = client.long_running_recognize(config=config, audio=audio)
        
        # バックエンドで処理継続（ブラウザ不要）
        start_time = time.time()
        job_manager.update_job_status(job_id, 'processing', 25, f'音声認識実行中（最大{max_wait_time//60}分）...')
        
        # 進捗更新ループ
        while not operation.done():
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                job_manager.update_job_status(job_id, 'failed', error=f'タイムアウト（{max_wait_time//60}分）')
                return
            
            # 進捗更新（30秒おき）
            progress = min(25 + (elapsed_time / max_wait_time) * 60, 85)
            remaining_minutes = max(0, (max_wait_time - elapsed_time) / 60)
            job_manager.update_job_status(
                job_id, 'processing', progress, 
                f'音声認識処理中... 残り約{remaining_minutes:.0f}分'
            )
            
            time.sleep(30)  # 30秒間隔でチェック
        
        # ステップ2: 結果取得
        job_manager.update_job_status(job_id, 'processing', 85, '音声認識結果を取得中...')
        response = operation.result()
        processing_time = (time.time() - start_time) / 60
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + "\n"
        
        if not transcript.strip():
            job_manager.update_job_status(job_id, 'failed', error='音声が認識されませんでした')
            return
        
        # ステップ3: 議事録生成
        job_manager.update_job_status(job_id, 'processing', 90, '議事録生成中...')
        
        try:
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            
            # 長いテキストの場合は要約
            max_length = 8000
            if len(transcript) > max_length:
                transcript_sample = transcript[:max_length] + "...\n\n[注：長時間音声のため一部抜粋]"
            else:
                transcript_sample = transcript
            
            prompt = f"""
以下の会議音声転写テキストから実用的な議事録を作成してください。

音声転写テキスト:
{transcript_sample}

以下の形式で議事録を作成してください：

# 🎤 会議議事録（自動生成）

## 📅 基本情報
- 生成日時：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- 処理時間：{processing_time:.1f}分
- 処理モード：{speed_mode}
- ジョブID：{job_id}

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

---
※この議事録は音声認識AIにより自動生成されました。
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "議事録作成の専門家として、実用的で読みやすい議事録を作成してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            meeting_minutes = response.choices[0].message.content
            
        except Exception as e:
            meeting_minutes = f"議事録生成でエラーが発生しました: {e}\n\n音声転写テキスト:\n{transcript}"
        
        # 結果保存
        result_data = {
            'transcript': transcript,
            'meeting_minutes': meeting_minutes,
            'processing_time': processing_time,
            'completed_at': datetime.now().isoformat(),
            'stats': {
                'characters': len(transcript),
                'estimated_duration': f"{len(transcript.split())//120}分"
            }
        }
        
        job_manager.update_job_status(job_id, 'completed', 100, '完了', result_data)
        
    except Exception as e:
        job_manager.update_job_status(job_id, 'failed', error=f'処理エラー: {str(e)}')

def start_background_processing(job_id, gcs_uri, file_extension, settings, bucket_name):
    """バックエンド処理を開始"""
    thread = threading.Thread(
        target=process_audio_async,
        args=(job_id, gcs_uri, file_extension, settings, bucket_name),
        daemon=True
    )
    thread.start()

def main():
    st.set_page_config(
        page_title="🔄 スリープ対応議事録アプリ",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 スリープ対応議事録アプリ")
    st.markdown("**PCがスリープしても大丈夫！** バックエンド処理で継続実行される議事録システム")
    
    if not setup_google_credentials():
        st.stop()
    
    # サイドバー設定
    st.sidebar.header("⚙️ システム設定")
    bucket_name = st.sidebar.text_input(
        "GCSバケット名", 
        value=st.secrets.get("GCS_BUCKET_NAME", "")
    )
    
    speed_mode = st.sidebar.selectbox(
        "処理モード",
        ["balanced", "fast", "quality"],
        index=0,
        help="""
        • balanced: 25-30分処理（推奨）
        • fast: 20-25分処理（高速）
        • quality: 35-40分処理（高品質）
        """
    )
    
    st.sidebar.markdown(f"""
    ### 🎯 {speed_mode}モードの特徴
    {"**⚡ 高速処理**: 話者分離なし、シンプル設定" if speed_mode == "fast" else "**🎨 高品質**: 話者分離あり、エンハンス機能" if speed_mode == "quality" else "**⚖️ バランス**: 速度と精度を両立"}
    
    ### 💡 スリープ対応
    - **処理継続**: PCスリープ中も継続
    - **進捗確認**: いつでも状況チェック  
    - **自動保存**: GCSに結果保存
    - **復旧可能**: ジョブIDで後から取得
    """)
    
    if not bucket_name:
        st.error("GCSバケット名を設定してください")
        st.stop()
    
    # セッション状態の初期化
    if 'current_job_id' not in st.session_state:
        st.session_state.current_job_id = None
    
    # タブ構成
    tab1, tab2 = st.tabs(["🚀 新規処理", "📊 進捗確認"])
    
    with tab1:
        st.header("🎵 音声ファイルアップロード")
        
        uploaded_file = st.file_uploader(
            "音声ファイルを選択してください",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="推奨: WAV形式、50MB以下"
        )
        
        if uploaded_file is not None:
            st.success(f"📁 ファイル: {uploaded_file.name}")
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"📊 ファイルサイズ: {file_size_mb:.1f} MB")
            
            # 予想処理時間
            if speed_mode == "fast":
                estimated_time = file_size_mb * 1.2
            elif speed_mode == "quality":
                estimated_time = file_size_mb * 2.0
            else:
                estimated_time = file_size_mb * 1.5
            
            st.info(f"⏱️ 予想処理時間: 約{estimated_time:.0f}分")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🚀 バックエンド処理開始", type="primary", use_container_width=True):
                    with st.spinner("ファイルアップロード中..."):
                        # ファイルアップロード
                        gcs_uri = upload_to_gcs(uploaded_file, bucket_name)
                        
                        if gcs_uri:
                            # ジョブ作成
                            job_manager = SimpleJobManager(bucket_name)
                            
                            file_info = {
                                'name': uploaded_file.name,
                                'size_mb': file_size_mb,
                                'gcs_uri': gcs_uri
                            }
                            settings = {
                                'speed_mode': speed_mode
                            }
                            
                            job_id = job_manager.create_job(file_info, settings)
                            st.session_state.current_job_id = job_id
                            
                            # バックエンド処理開始
                            file_extension = os.path.splitext(uploaded_file.name)[1]
                            start_background_processing(job_id, gcs_uri, file_extension, settings, bucket_name)
                            
                            st.success("✅ バックエンド処理を開始しました！")
                            st.info(f"🆔 ジョブID: **{job_id}**")
                            st.warning("💡 **重要**: PCをスリープさせても処理は継続されます。「進捗確認」タブで状況をチェックしてください。")
                            
                            # 自動的に進捗確認タブに移動するための情報表示
                            st.markdown("---")
                            st.markdown("### 📱 次のステップ")
                            st.markdown("1. **「進捗確認」タブ**をクリック")
                            st.markdown("2. **ジョブIDをコピー**して保存")
                            st.markdown(f"3. **約{estimated_time:.0f}分後**に結果確認")
                            st.markdown("4. **PCをスリープ**させても大丈夫！")
            
            with col2:
                st.markdown("### 🔄 処理の流れ")
                st.markdown(f"""
                1. **ファイルアップロード** (1分)
                2. **音声認識開始** ({estimated_time:.0f}分)
                3. **議事録生成** (2分)
                4. **結果保存** (1分)
                
                **💻 PCをスリープさせてOK！**
                処理はクラウドで継続されます。
                """)
    
    with tab2:
        st.header("📊 処理進捗確認")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 現在のジョブ確認
            if st.session_state.current_job_id:
                current_job_id = st.session_state.current_job_id
                st.info(f"📍 現在のジョブ: **{current_job_id}**")
                
                if st.button("🔄 最新状況を確認", type="primary"):
                    job_manager = SimpleJobManager(bucket_name)
                    job_status = job_manager.get_job_status(current_job_id)
                    
                    if job_status:
                        # ステータス表示
                        status = job_status['status']
                        if status == 'processing':
                            st.info(f"⚡ 処理中: {job_status.get('current_step', 'Unknown')}")
                            st.progress(job_status.get('progress', 0) / 100)
                        elif status == 'completed':
                            st.success("🎉 処理完了！")
                            st.balloons()
                            
                            # 結果表示
                            result = job_status.get('result', {})
                            if 'meeting_minutes' in result:
                                st.markdown("### 📋 生成された議事録")
                                st.markdown(result['meeting_minutes'])
                                
                                # ダウンロード
                                st.download_button(
                                    label="📥 議事録をダウンロード",
                                    data=result['meeting_minutes'],
                                    file_name=f"議事録_{current_job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
                                
                                # 統計情報
                                with st.expander("📊 処理統計"):
                                    st.json(result.get('stats', {}))
                        elif status == 'failed':
                            st.error(f"❌ 処理失敗: {job_status.get('error', '不明なエラー')}")
                        else:
                            st.warning(f"⏳ ステータス: {status}")
                        
                        # 詳細情報
                        with st.expander("🔍 詳細情報"):
                            st.json(job_status)
                    else:
                        st.error("ジョブが見つかりません")
        
        with col2:
            st.markdown("### 💡 使い方")
            st.markdown("""
            1. **自動更新**: 「最新状況を確認」で進捗チェック
            2. **ジョブID**: 後で確認する場合は保存
            3. **完了通知**: 処理完了時に結果表示
            4. **ダウンロード**: 議事録ファイル取得
            """)
        
        # 手動ジョブID確認
        st.markdown("---")
        st.markdown("### 🆔 手動ジョブID確認")
        manual_job_id = st.text_input("ジョブIDを入力", placeholder="例: abc12345")
        
        if manual_job_id and st.button("🔍 このジョブを確認"):
            job_manager = SimpleJobManager(bucket_name)
            job_status = job_manager.get_job_status(manual_job_id)
            
            if job_status:
                st.session_state.current_job_id = manual_job_id
                st.success(f"ジョブ {manual_job_id} が見つかりました！上の「最新状況を確認」で詳細を確認してください。")
            else:
                st.error("指定されたジョブIDが見つかりません")
    
    # フッター
    st.markdown("---")
    st.markdown("### 🎯 システムの特徴")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔄 スリープ対応**")
        st.markdown("PCがスリープしても処理継続")
    
    with col2:
        st.markdown("**📊 進捗追跡**")
        st.markdown("いつでも処理状況を確認")
    
    with col3:
        st.markdown("**🔒 自動保存**")
        st.markdown("結果はクラウドに安全保存")

if __name__ == "__main__":
    main()
